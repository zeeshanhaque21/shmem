"""Multi-process concurrency tests for SharedStore."""

import multiprocessing
import numpy as np
import pytest
from shmem import SharedStore

STORE_NAME = "test_concurrent"


@pytest.fixture
def store():
    s = SharedStore.create(STORE_NAME, size_mb=4, max_entries=256)
    yield s
    s.destroy()


def _reader_worker(name, locks, max_entries, key, expected_sum, results, idx):
    """Child process: connect, read a key, verify checksum."""
    try:
        s = SharedStore.connect(name, locks, max_entries)
        arr = s.get(key)
        if arr is not None and arr.sum() == expected_sum:
            results[idx] = 1
        else:
            results[idx] = 0
        s.close()
    except Exception:
        results[idx] = -1


def test_concurrent_readers(store):
    """Multiple readers should be able to read simultaneously without corruption."""
    arr = np.arange(1000, dtype=np.int64)
    store.put("shared_data", arr)
    expected_sum = arr.sum()

    n_readers = 4
    results = multiprocessing.Array("i", n_readers)

    procs = []
    for i in range(n_readers):
        p = multiprocessing.Process(
            target=_reader_worker,
            args=(STORE_NAME, store.locks(), 256, "shared_data", expected_sum, results, i),
        )
        procs.append(p)
        p.start()

    for p in procs:
        p.join(timeout=10)

    for i in range(n_readers):
        assert results[i] == 1, f"Reader {i} failed (result={results[i]})"


def _writer_worker(name, locks, max_entries, key, value, done_flag):
    """Child process: connect and write a value."""
    try:
        s = SharedStore.connect(name, locks, max_entries)
        arr = np.full(100, value, dtype=np.float64)
        s.put(key, arr)
        s.close()
        done_flag.value = 1
    except Exception:
        done_flag.value = -1


def test_writer_exclusion(store):
    """Writers should get exclusive access — sequential writes should not corrupt."""
    n_writers = 4
    done_flags = [multiprocessing.Value("i", 0) for _ in range(n_writers)]
    procs = []

    for i in range(n_writers):
        p = multiprocessing.Process(
            target=_writer_worker,
            args=(STORE_NAME, store.locks(), 256, f"key_{i}", float(i), done_flags[i]),
        )
        procs.append(p)
        p.start()

    for p in procs:
        p.join(timeout=10)

    for i in range(n_writers):
        assert done_flags[i].value == 1, f"Writer {i} failed"

    # Verify all writes are present
    for i in range(n_writers):
        arr = store.get(f"key_{i}")
        assert arr is not None
        np.testing.assert_array_equal(arr, np.full(100, float(i), dtype=np.float64))


def _stress_worker(name, locks, max_entries, worker_id, n_ops, results_flag):
    """Stress test: each worker writes and reads its own keys."""
    try:
        s = SharedStore.connect(name, locks, max_entries)
        for j in range(n_ops):
            key = f"w{worker_id}_k{j}"
            arr = np.full(50, worker_id * 1000 + j, dtype=np.int32)
            s.put(key, arr)
            result = s.get(key)
            if result is None or result[0] != worker_id * 1000 + j:
                results_flag.value = -1
                s.close()
                return
        # Clean up
        for j in range(n_ops):
            s.delete(f"w{worker_id}_k{j}")
        s.close()
        results_flag.value = 1
    except Exception:
        results_flag.value = -1


def test_stress_concurrent_rw(store):
    """Multiple processes doing interleaved reads and writes."""
    n_workers = 4
    n_ops = 20
    flags = [multiprocessing.Value("i", 0) for _ in range(n_workers)]
    procs = []

    for i in range(n_workers):
        p = multiprocessing.Process(
            target=_stress_worker,
            args=(STORE_NAME, store.locks(), 256, i, n_ops, flags[i]),
        )
        procs.append(p)
        p.start()

    for p in procs:
        p.join(timeout=30)

    for i in range(n_workers):
        assert flags[i].value == 1, f"Stress worker {i} failed"


def _child_put_and_delete(name, locks, max_entries, done_flag):
    """Child writes a key, reads it, deletes it."""
    try:
        s = SharedStore.connect(name, locks, max_entries)
        arr = np.array([42, 43, 44], dtype=np.int32)
        s.put("child_key", arr)
        result = s.get("child_key")
        assert result is not None
        assert result[0] == 42
        s.delete("child_key")
        assert s.get("child_key") is None
        s.close()
        done_flag.value = 1
    except Exception:
        done_flag.value = -1


def test_child_full_lifecycle(store):
    """A child process can put, get, and delete keys."""
    flag = multiprocessing.Value("i", 0)
    p = multiprocessing.Process(
        target=_child_put_and_delete,
        args=(STORE_NAME, store.locks(), 256, flag),
    )
    p.start()
    p.join(timeout=10)
    assert flag.value == 1
