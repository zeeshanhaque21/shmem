"""
Benchmark: Zero-Copy Shared Memory Libraries for Python

Compares:
  1. multiprocessing.shared_memory (stdlib)
  2. SharedArray (C extension)

Measures:
  - Write latency: time to copy a numpy array into shared memory
  - Read latency: time to create a zero-copy view on shared memory
  - Cross-process throughput: producer writes, consumer reads via shared mem

Image sizes tested:
  - 720p  (1280x720x3)   ~2.7 MB
  - 1080p (1920x1080x3)  ~6.2 MB
  - 4K    (3840x2160x3)  ~24.9 MB

Usage:
  uv run python benchmark.py
"""

import os
import time
import statistics
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

try:
    import SharedArray as sa

    HAS_SHARED_ARRAY = True
except ImportError:
    HAS_SHARED_ARRAY = False


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
# 1. multiprocessing.shared_memory latency
# ---------------------------------------------------------------------------


def bench_stdlib_shm_latency(shape, image):
    nbytes = int(np.prod(shape)) * np.dtype(DTYPE).itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes, name="bench_stdlib")

    write_times = []
    read_times = []

    try:
        for _ in range(WARMUP_ITERS):
            arr = np.ndarray(shape, dtype=DTYPE, buffer=shm.buf)
            arr[:] = image

        for _ in range(BENCH_ITERS):
            arr = np.ndarray(shape, dtype=DTYPE, buffer=shm.buf)
            t0 = time.perf_counter_ns()
            arr[:] = image
            t1 = time.perf_counter_ns()
            write_times.append((t1 - t0) / 1000)

        for _ in range(BENCH_ITERS):
            t0 = time.perf_counter_ns()
            arr = np.ndarray(shape, dtype=DTYPE, buffer=shm.buf)
            t1 = time.perf_counter_ns()
            read_times.append((t1 - t0) / 1000)
    finally:
        shm.close()
        shm.unlink()

    return write_times, read_times


# ---------------------------------------------------------------------------
# 2. SharedArray latency
# ---------------------------------------------------------------------------


def bench_shared_array_latency(shape, image):
    sa_name = "shm://bench_sa"
    try:
        sa.delete(sa_name)
    except Exception:
        pass

    sa.create(sa_name, shape=shape, dtype=DTYPE)
    write_times = []
    read_times = []

    try:
        for _ in range(WARMUP_ITERS):
            arr = sa.attach(sa_name)
            arr[:] = image

        for _ in range(BENCH_ITERS):
            arr = sa.attach(sa_name)
            t0 = time.perf_counter_ns()
            arr[:] = image
            t1 = time.perf_counter_ns()
            write_times.append((t1 - t0) / 1000)

        for _ in range(BENCH_ITERS):
            t0 = time.perf_counter_ns()
            arr = sa.attach(sa_name)
            t1 = time.perf_counter_ns()
            read_times.append((t1 - t0) / 1000)
    finally:
        sa.delete(sa_name)

    return write_times, read_times


# ---------------------------------------------------------------------------
# 3. Cross-process throughput
# ---------------------------------------------------------------------------


def _shm_producer(shm_name, shape, ready, start, n):
    shm = shared_memory.SharedMemory(name=shm_name, create=False, track=False)
    arr = np.ndarray(shape, dtype=DTYPE, buffer=shm.buf)
    frame = make_image(shape)
    ready.set()
    start.wait()
    for _ in range(n):
        arr[:] = frame
    shm.close()


def _shm_consumer(shm_name, shape, ready, start, done, n):
    shm = shared_memory.SharedMemory(name=shm_name, create=False, track=False)
    arr = np.ndarray(shape, dtype=DTYPE, buffer=shm.buf)
    sink = np.empty_like(arr)
    ready.set()
    start.wait()
    for _ in range(n):
        np.copyto(sink, arr)  # force full read of shared memory
    done.set()
    shm.close()


def bench_shm_throughput(shape):
    nbytes = int(np.prod(shape)) * np.dtype(DTYPE).itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes, name="bench_tp")

    n = THROUGHPUT_FRAMES
    prod_ready, cons_ready = mp.Event(), mp.Event()
    start, done = mp.Event(), mp.Event()

    try:
        p = mp.Process(target=_shm_producer, args=(shm.name, shape, prod_ready, start, n))
        c = mp.Process(target=_shm_consumer, args=(shm.name, shape, cons_ready, start, done, n))
        p.start()
        c.start()
        prod_ready.wait()
        cons_ready.wait()

        t0 = time.perf_counter()
        start.set()
        done.wait()
        t1 = time.perf_counter()

        p.join(timeout=5)
        c.join(timeout=5)
        elapsed = t1 - t0
    finally:
        shm.close()
        shm.unlink()

    return n / elapsed, elapsed, n


def _sa_producer(sa_name, shape, ready, start, n):
    arr = sa.attach(sa_name)
    frame = make_image(shape)
    ready.set()
    start.wait()
    for _ in range(n):
        arr[:] = frame


def _sa_consumer(sa_name, shape, ready, start, done, n):
    arr = sa.attach(sa_name)
    sink = np.empty_like(arr)
    ready.set()
    start.wait()
    for _ in range(n):
        np.copyto(sink, arr)  # force full read of shared memory
    done.set()


def bench_sa_throughput(shape):
    sa_name = "shm://bench_tp_sa"
    try:
        sa.delete(sa_name)
    except Exception:
        pass
    sa.create(sa_name, shape=shape, dtype=DTYPE)

    n = THROUGHPUT_FRAMES
    prod_ready, cons_ready = mp.Event(), mp.Event()
    start, done = mp.Event(), mp.Event()

    try:
        p = mp.Process(target=_sa_producer, args=(sa_name, shape, prod_ready, start, n))
        c = mp.Process(target=_sa_consumer, args=(sa_name, shape, cons_ready, start, done, n))
        p.start()
        c.start()
        prod_ready.wait()
        cons_ready.wait()

        t0 = time.perf_counter()
        start.set()
        done.wait()
        t1 = time.perf_counter()

        p.join(timeout=5)
        c.join(timeout=5)
        elapsed = t1 - t0
    finally:
        sa.delete(sa_name)

    return n / elapsed, elapsed, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_latency():
    print("=" * 80)
    print("LATENCY BENCHMARK  (single process, in-memory)")
    print(f"  Warmup: {WARMUP_ITERS}  |  Iterations: {BENCH_ITERS}")
    print("=" * 80)

    for size_name, shape in IMAGE_SIZES.items():
        nbytes = int(np.prod(shape)) * np.dtype(DTYPE).itemsize
        image = make_image(shape)
        print(f"\n  {size_name} ({shape[1]}x{shape[0]}x{shape[2]}) = {nbytes / 1e6:.1f} MB\n")

        methods = {}

        w, r = bench_stdlib_shm_latency(shape, image)
        methods["stdlib shared_memory"] = (w, r)

        if HAS_SHARED_ARRAY:
            w, r = bench_shared_array_latency(shape, image)
            methods["SharedArray"] = (w, r)

        header = f"  {'Method':<25} {'Write (med)':>12} {'Write (p99)':>12} {'Read (med)':>12} {'Read (p99)':>12}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for method, (wt, rt) in methods.items():
            wm, rm = statistics.median(wt), statistics.median(rt)
            wp, rp = percentile(wt, 99), percentile(rt, 99)
            print(f"  {method:<25} {format_us(wm)} {format_us(wp)} {format_us(rm)} {format_us(rp)}")


def run_throughput():
    print("\n" + "=" * 80)
    print("CROSS-PROCESS THROUGHPUT  (producer -> consumer)")
    print(f"  Frames: {THROUGHPUT_FRAMES}")
    print("=" * 80)

    for size_name, shape in IMAGE_SIZES.items():
        nbytes = int(np.prod(shape)) * np.dtype(DTYPE).itemsize
        print(f"\n  {size_name} ({shape[1]}x{shape[0]}x{shape[2]}) = {nbytes / 1e6:.1f} MB\n")

        header = f"  {'Method':<25} {'FPS':>10} {'Time (s)':>10} {'Bandwidth':>14}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        fps, elapsed, n = bench_shm_throughput(shape)
        bw = fps * nbytes / 1e9
        print(f"  {'stdlib shared_memory':<25} {fps:>10.1f} {elapsed:>10.3f} {bw:>11.2f} GB/s")

        if HAS_SHARED_ARRAY:
            fps, elapsed, n = bench_sa_throughput(shape)
            bw = fps * nbytes / 1e9
            print(f"  {'SharedArray':<25} {fps:>10.1f} {elapsed:>10.3f} {bw:>11.2f} GB/s")


def main():
    print()
    print("  Shared Memory Benchmark — Zero-Copy Libraries")
    print(f"  Python {os.sys.version.split()[0]} | NumPy {np.__version__}")
    print(f"  PID: {os.getpid()} | CPUs: {os.cpu_count()}")
    if HAS_SHARED_ARRAY:
        print("  SharedArray: available")
    else:
        print("  SharedArray: NOT installed (skipping)")
    print()

    run_latency()
    run_throughput()

    print("\n" + "=" * 80)
    print("NOTES")
    print("=" * 80)
    print("""
  - 'Write' = time to memcpy a numpy array INTO the shared region
  - 'Read'  = time to create a zero-copy ndarray VIEW on shared memory
    (no data is copied or deserialized — it's just a pointer + metadata)
  - Throughput = producer writes full frames, consumer touches data
    to confirm it's accessible. No synchronization between them.
""")


if __name__ == "__main__":
    main()
