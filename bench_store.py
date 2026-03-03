"""
Benchmark: SharedStore put/get latency and cross-process throughput.

Measures:
  - put() latency:  time to copy a numpy array into the store (key lookup + alloc + memcpy)
  - get() latency:  time to get a zero-copy ndarray view (key lookup + pointer math)
  - get_bytes():    time to get a zero-copy memoryview of raw bytes
  - Cross-process:  producer puts frames, consumer gets them via SharedStore

Image sizes tested:
  - 720p  (1280x720x3)   ~2.7 MB
  - 1080p (1920x1080x3)  ~6.2 MB
  - 4K    (3840x2160x3)  ~24.9 MB

Usage:
  uv run python bench_store.py
"""

import os
import time
import statistics
import multiprocessing as mp

import numpy as np
from shmem import SharedStore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_SIZES = {
    "720p": (720, 1280, 3),
    "1080p": (1080, 1920, 3),
    "4K": (2160, 3840, 3),
}

DTYPE = np.uint8
WARMUP_ITERS = 5
BENCH_ITERS = 100
THROUGHPUT_FRAMES = 500
STORE_NAME = "bench_store"
STORE_SIZE_MB = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_image(shape):
    return np.random.randint(0, 255, size=shape, dtype=DTYPE)


def percentile(data, p):
    s = sorted(data)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


def format_us(val):
    if val >= 1_000_000:
        return f"{val / 1_000_000:>10.2f} s "
    elif val >= 1_000:
        return f"{val / 1_000:>10.2f} ms"
    else:
        return f"{val:>10.2f} us"


# ---------------------------------------------------------------------------
# 1. Single-process put/get latency
# ---------------------------------------------------------------------------


def bench_put_get_latency(shape, image):
    store = SharedStore.create(STORE_NAME, chunk_size_mb=STORE_SIZE_MB, max_entries=64)

    put_times = []
    get_times = []

    try:
        # Warmup
        for i in range(WARMUP_ITERS):
            store.put("warmup", image)
            store.get("warmup")
        store.delete("warmup")

        # Bench put
        for i in range(BENCH_ITERS):
            key = f"frame_{i % 4}"
            t0 = time.perf_counter_ns()
            store.put(key, image)
            t1 = time.perf_counter_ns()
            put_times.append((t1 - t0) / 1000)

        # Bench get (key is already stored)
        store.put("bench_read", image)
        for _ in range(BENCH_ITERS):
            t0 = time.perf_counter_ns()
            arr = store.get("bench_read")
            t1 = time.perf_counter_ns()
            get_times.append((t1 - t0) / 1000)
        del arr

        # Clean up all keys so no views remain
        for key in store.keys():
            store.delete(key)

    finally:
        store.destroy()

    return put_times, get_times


# ---------------------------------------------------------------------------
# 2. get_bytes latency for various payload sizes
# ---------------------------------------------------------------------------


def bench_get_bytes_latency():
    store = SharedStore.create(STORE_NAME, chunk_size_mb=STORE_SIZE_MB, max_entries=64)
    sizes = [64, 1024, 64 * 1024, 1024 * 1024]
    results = {}

    try:
        for sz in sizes:
            data = bytes(range(256)) * (sz // 256 + 1)
            data = data[:sz]
            store.put_bytes("raw", data)

            times = []
            for _ in range(WARMUP_ITERS):
                store.get_bytes("raw")

            for _ in range(BENCH_ITERS):
                t0 = time.perf_counter_ns()
                mv = store.get_bytes("raw")
                t1 = time.perf_counter_ns()
                times.append((t1 - t0) / 1000)
            del mv

            results[sz] = times
            store.delete("raw")

    finally:
        store.destroy()

    return results


# ---------------------------------------------------------------------------
# 3. Cross-process throughput
# ---------------------------------------------------------------------------


def _producer(store_name, locks, shape, ready, start, n):
    store = SharedStore.connect(store_name, locks)
    frame = make_image(shape)
    ready.set()
    start.wait()
    for i in range(n):
        store.put("frame", frame)
    store.close()


def _consumer(store_name, locks, shape, ready, start, done, n):
    store = SharedStore.connect(store_name, locks)
    ready.set()
    start.wait()
    for i in range(n):
        while True:
            arr = store.get("frame")
            if arr is not None:
                break
    done.set()
    store.close()


def bench_throughput(shape):
    nbytes = int(np.prod(shape)) * np.dtype(DTYPE).itemsize
    store = SharedStore.create(STORE_NAME, chunk_size_mb=STORE_SIZE_MB, max_entries=64)

    n = THROUGHPUT_FRAMES
    prod_ready, cons_ready = mp.Event(), mp.Event()
    start, done = mp.Event(), mp.Event()

    try:
        p = mp.Process(
            target=_producer,
            args=(STORE_NAME, store.locks(), shape, prod_ready, start, n),
        )
        c = mp.Process(
            target=_consumer,
            args=(STORE_NAME, store.locks(), shape, cons_ready, start, done, n),
        )
        p.start()
        c.start()
        prod_ready.wait()
        cons_ready.wait()

        t0 = time.perf_counter()
        start.set()
        done.wait()
        t1 = time.perf_counter()

        p.join(timeout=10)
        c.join(timeout=10)
        elapsed = t1 - t0
    finally:
        store.destroy()

    return n / elapsed, elapsed, n


# ---------------------------------------------------------------------------
# 4. Multi-chunk auto-growth
# ---------------------------------------------------------------------------

# Chunk sizes chosen so that each image size will trigger growth:
#   720p  (2.8 MB): 4 MB chunk → ~1 image/chunk → growth after 1st unique key
#   1080p (6.2 MB): 8 MB chunk → ~1 image/chunk → growth after 1st unique key
#   4K   (24.9 MB): 32 MB chunk → ~1 image/chunk → growth after 1st unique key
GROWTH_CHUNK_SIZES = {
    "720p": 4,
    "1080p": 8,
    "4K": 32,
}

GROWTH_N_KEYS = 20  # store this many unique keys (will span many chunks)


def bench_multi_chunk_put(shape, image, chunk_size_mb):
    """Store GROWTH_N_KEYS unique images, measuring each put(). Returns
    (put_times, n_chunks_created, total_data_mb)."""
    store = SharedStore.create(STORE_NAME, chunk_size_mb=chunk_size_mb, max_entries=256)

    put_times = []

    try:
        # Warmup
        for i in range(WARMUP_ITERS):
            store.put("warmup", image)
        store.delete("warmup")

        for i in range(GROWTH_N_KEYS):
            t0 = time.perf_counter_ns()
            store.put(f"img_{i}", image)
            t1 = time.perf_counter_ns()
            put_times.append((t1 - t0) / 1000)

        info = store.info()
        n_chunks = info["chunk_count"]
        total_data_mb = info["total_bytes"] / 1e6

        # Read back from all chunks
        get_times = []
        for i in range(GROWTH_N_KEYS):
            t0 = time.perf_counter_ns()
            arr = store.get(f"img_{i}")
            t1 = time.perf_counter_ns()
            get_times.append((t1 - t0) / 1000)
        del arr

        for i in range(GROWTH_N_KEYS):
            store.delete(f"img_{i}")

    finally:
        store.destroy()

    return put_times, get_times, n_chunks, total_data_mb


def _growth_producer(store_name, locks, shape, n_keys, ready, start):
    store = SharedStore.connect(store_name, locks)
    frame = make_image(shape)
    ready.set()
    start.wait()
    for i in range(n_keys):
        store.put(f"frame_{i}", frame)
    store.close()


def _growth_consumer(store_name, locks, n_keys, ready, start, done):
    store = SharedStore.connect(store_name, locks)
    ready.set()
    start.wait()
    # Wait for all keys to appear, then read them all
    for i in range(n_keys):
        while True:
            arr = store.get(f"frame_{i}")
            if arr is not None:
                break
    done.set()
    store.close()


def bench_multi_chunk_throughput(shape, chunk_size_mb):
    nbytes = int(np.prod(shape)) * np.dtype(DTYPE).itemsize
    n_keys = GROWTH_N_KEYS
    store = SharedStore.create(STORE_NAME, chunk_size_mb=chunk_size_mb, max_entries=256)

    prod_ready, cons_ready = mp.Event(), mp.Event()
    start, done = mp.Event(), mp.Event()

    try:
        p = mp.Process(
            target=_growth_producer,
            args=(STORE_NAME, store.locks(), shape, n_keys, prod_ready, start),
        )
        c = mp.Process(
            target=_growth_consumer,
            args=(STORE_NAME, store.locks(), n_keys, cons_ready, start, done),
        )
        p.start()
        c.start()
        prod_ready.wait()
        cons_ready.wait()

        t0 = time.perf_counter()
        start.set()
        done.wait()
        t1 = time.perf_counter()

        p.join(timeout=30)
        c.join(timeout=30)
        elapsed = t1 - t0

        info = store.info()
        n_chunks = info["chunk_count"]
    finally:
        store.destroy()

    fps = n_keys / elapsed
    return fps, elapsed, n_keys, n_chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_latency():
    print("=" * 80)
    print("SharedStore PUT/GET LATENCY  (single process)")
    print(f"  Warmup: {WARMUP_ITERS}  |  Iterations: {BENCH_ITERS}")
    print("=" * 80)

    for size_name, shape in IMAGE_SIZES.items():
        nbytes = int(np.prod(shape)) * np.dtype(DTYPE).itemsize
        image = make_image(shape)
        print(f"\n  {size_name} ({shape[1]}x{shape[0]}x{shape[2]}) = {nbytes / 1e6:.1f} MB\n")

        put_t, get_t = bench_put_get_latency(shape, image)

        header = f"  {'Operation':<25} {'Median':>12} {'p99':>12} {'Min':>12}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for label, t in [("put()", put_t), ("get() [zero-copy]", get_t)]:
            med = statistics.median(t)
            p99 = percentile(t, 99)
            mn = min(t)
            print(f"  {label:<25} {format_us(med)} {format_us(p99)} {format_us(mn)}")


def run_get_bytes():
    print("\n" + "=" * 80)
    print("SharedStore GET_BYTES LATENCY  (zero-copy memoryview)")
    print(f"  Warmup: {WARMUP_ITERS}  |  Iterations: {BENCH_ITERS}")
    print("=" * 80)

    results = bench_get_bytes_latency()

    header = f"  {'Payload Size':<25} {'Median':>12} {'p99':>12} {'Min':>12}"
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    for sz, times in results.items():
        if sz < 1024:
            label = f"{sz} B"
        elif sz < 1024 * 1024:
            label = f"{sz // 1024} KB"
        else:
            label = f"{sz // (1024 * 1024)} MB"
        med = statistics.median(times)
        p99 = percentile(times, 99)
        mn = min(times)
        print(f"  {label:<25} {format_us(med)} {format_us(p99)} {format_us(mn)}")


def run_throughput():
    print("\n" + "=" * 80)
    print("SharedStore CROSS-PROCESS THROUGHPUT  (producer -> consumer)")
    print(f"  Frames: {THROUGHPUT_FRAMES}")
    print("=" * 80)

    for size_name, shape in IMAGE_SIZES.items():
        nbytes = int(np.prod(shape)) * np.dtype(DTYPE).itemsize
        print(f"\n  {size_name} ({shape[1]}x{shape[0]}x{shape[2]}) = {nbytes / 1e6:.1f} MB\n")

        header = f"  {'Method':<25} {'FPS':>10} {'Time (s)':>10} {'Bandwidth':>14}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        fps, elapsed, n = bench_throughput(shape)
        bw = fps * nbytes / 1e9
        print(f"  {'SharedStore':<25} {fps:>10.1f} {elapsed:>10.3f} {bw:>11.2f} GB/s")


def run_multi_chunk():
    print("\n" + "=" * 80)
    print("MULTI-CHUNK AUTO-GROWTH  (small chunks, many unique keys)")
    print(f"  Keys: {GROWTH_N_KEYS}  |  Warmup: {WARMUP_ITERS}")
    print("=" * 80)

    for size_name, shape in IMAGE_SIZES.items():
        nbytes = int(np.prod(shape)) * np.dtype(DTYPE).itemsize
        chunk_mb = GROWTH_CHUNK_SIZES[size_name]
        image = make_image(shape)
        print(f"\n  {size_name} ({shape[1]}x{shape[0]}x{shape[2]}) = {nbytes / 1e6:.1f} MB"
              f"  |  chunk_size = {chunk_mb} MB\n")

        put_t, get_t, n_chunks, total_mb = bench_multi_chunk_put(shape, image, chunk_mb)
        print(f"  Chunks allocated: {n_chunks}  |  Total data region: {total_mb:.1f} MB\n")

        header = f"  {'Operation':<25} {'Median':>12} {'p99':>12} {'Min':>12}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for label, t in [("put() [with growth]", put_t), ("get() [cross-chunk]", get_t)]:
            med = statistics.median(t)
            p99 = percentile(t, 99)
            mn = min(t)
            print(f"  {label:<25} {format_us(med)} {format_us(p99)} {format_us(mn)}")


def run_multi_chunk_throughput():
    print("\n" + "=" * 80)
    print("MULTI-CHUNK CROSS-PROCESS THROUGHPUT  (auto-growth under load)")
    print(f"  Keys: {GROWTH_N_KEYS}")
    print("=" * 80)

    for size_name, shape in IMAGE_SIZES.items():
        nbytes = int(np.prod(shape)) * np.dtype(DTYPE).itemsize
        chunk_mb = GROWTH_CHUNK_SIZES[size_name]
        print(f"\n  {size_name} ({shape[1]}x{shape[0]}x{shape[2]}) = {nbytes / 1e6:.1f} MB"
              f"  |  chunk_size = {chunk_mb} MB\n")

        header = f"  {'Method':<25} {'FPS':>10} {'Time (s)':>10} {'Chunks':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        fps, elapsed, n, n_chunks = bench_multi_chunk_throughput(shape, chunk_mb)
        print(f"  {'SharedStore (growth)':<25} {fps:>10.1f} {elapsed:>10.3f} {n_chunks:>8}")


def main():
    print()
    print("  SharedStore Benchmark")
    print(f"  Python {os.sys.version.split()[0]} | NumPy {np.__version__}")
    print(f"  PID: {os.getpid()} | CPUs: {os.cpu_count()}")
    print()

    run_latency()
    run_get_bytes()
    run_throughput()
    run_multi_chunk()
    run_multi_chunk_throughput()

    print("\n" + "=" * 80)
    print("NOTES")
    print("=" * 80)
    print("""
  - put()  = key lookup + alloc + memcpy (dominated by memcpy for large arrays)
  - get()  = key lookup + pointer math, returns zero-copy ndarray view
  - get_bytes() = key lookup + slice, returns zero-copy memoryview
  - Throughput = producer calls put(), consumer calls get() in a loop.
    Includes lock acquisition overhead for each operation.
  - Multi-chunk: small chunk sizes force auto-growth; each put() that exceeds
    the current chunk creates a new shared memory segment on demand.
""")


if __name__ == "__main__":
    main()
