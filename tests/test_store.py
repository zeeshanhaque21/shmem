"""Integration tests for SharedStore — put/get round-trips, zero-copy, bytes, lifecycle."""

import numpy as np
import pytest
from shmem import SharedStore, OutOfMemoryError


STORE_NAME = "test_store"


@pytest.fixture
def store():
    s = SharedStore.create(STORE_NAME, chunk_size_mb=1, max_entries=64)
    yield s
    s.destroy()


def test_put_get_roundtrip(store):
    arr = np.arange(100, dtype=np.float32)
    store.put("data", arr)
    result = store.get("data")
    assert result is not None
    np.testing.assert_array_equal(result, arr)


def test_get_returns_correct_dtype_and_shape(store):
    arr = np.zeros((10, 20, 3), dtype=np.uint8)
    store.put("img", arr)
    result = store.get("img")
    assert result.dtype == np.uint8
    assert result.shape == (10, 20, 3)


def test_get_missing_returns_none(store):
    assert store.get("nonexistent") is None


def test_get_is_read_only(store):
    arr = np.array([1, 2, 3], dtype=np.int32)
    store.put("ro", arr)
    result = store.get("ro")
    assert not result.flags.writeable


def test_get_mut_is_writable(store):
    arr = np.array([1, 2, 3], dtype=np.int32)
    store.put("rw", arr)
    result = store.get_mut("rw")
    assert result.flags.writeable
    result[0] = 99
    # Verify mutation is visible
    check = store.get("rw")
    assert check[0] == 99


def test_zero_copy_view(store):
    arr = np.arange(1000, dtype=np.float64)
    store.put("big", arr)
    v1 = store.get("big")
    v2 = store.get("big")
    # Both should point to the same memory (zero-copy)
    assert v1.ctypes.data == v2.ctypes.data


def test_overwrite_key(store):
    arr1 = np.array([1, 2, 3], dtype=np.int32)
    arr2 = np.array([4, 5, 6, 7], dtype=np.int32)
    store.put("key", arr1)
    store.put("key", arr2)
    result = store.get("key")
    np.testing.assert_array_equal(result, arr2)


def test_delete(store):
    arr = np.array([1, 2, 3], dtype=np.int32)
    store.put("del", arr)
    assert store.delete("del") is True
    assert store.get("del") is None
    assert store.delete("del") is False  # already gone


def test_put_get_bytes(store):
    data = b"hello world, this is raw bytes"
    store.put_bytes("raw", data)
    mv = store.get_bytes("raw")
    assert mv is not None
    assert bytes(mv) == data


def test_get_bytes_missing(store):
    assert store.get_bytes("nope") is None


def test_keys(store):
    store.put("a", np.array([1]))
    store.put("b", np.array([2]))
    store.put("c", np.array([3]))
    k = sorted(store.keys())
    assert k == ["a", "b", "c"]


def test_keys_after_delete(store):
    store.put("a", np.array([1]))
    store.put("b", np.array([2]))
    store.delete("a")
    assert store.keys() == ["b"]


def test_info(store):
    info = store.info()
    assert info["max_entries"] == 64
    assert info["entry_count"] == 0
    assert info["chunk_count"] == 1
    store.put("x", np.zeros(100, dtype=np.uint8))
    info = store.info()
    assert info["entry_count"] == 1
    assert info["used_blocks"] == 1


def test_connect(store):
    """Test that a second store instance can read data written by the first."""
    arr = np.arange(50, dtype=np.float32)
    store.put("shared", arr)

    store2 = SharedStore.connect(STORE_NAME, store.locks())
    result = store2.get("shared")
    np.testing.assert_array_equal(result, arr)
    store2.close()


def test_connect_can_write(store):
    """Test that a connected store can write data readable by the creator."""
    store2 = SharedStore.connect(STORE_NAME, store.locks())
    arr = np.array([10, 20, 30], dtype=np.int64)
    store2.put("from_child", arr)
    store2.close()

    result = store.get("from_child")
    np.testing.assert_array_equal(result, arr)


def test_multiple_dtypes(store):
    for dtype in [np.uint8, np.int16, np.float32, np.float64, np.int64]:
        arr = np.arange(10, dtype=dtype)
        store.put(f"dtype_{dtype.__name__}", arr)
        result = store.get(f"dtype_{dtype.__name__}")
        np.testing.assert_array_equal(result, arr)
        assert result.dtype == dtype


def test_large_array(store):
    arr = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    store.put("large", arr)
    result = store.get("large")
    np.testing.assert_array_equal(result, arr)


# -- Multi-chunk tests -------------------------------------------------------


def test_auto_growth():
    """Fill one chunk, verify that a second chunk is automatically created."""
    # Use a tiny chunk so we can fill it quickly
    store = SharedStore.create("test_growth", chunk_size_mb=1, max_entries=64)
    try:
        info = store.info()
        assert info["chunk_count"] == 1

        # Each array is ~100KB. A 1MB chunk fits ~10 of them.
        arr = np.zeros(100_000, dtype=np.uint8)  # 100KB
        for i in range(12):
            store.put(f"k{i}", arr)

        info = store.info()
        assert info["chunk_count"] >= 2
        assert info["entry_count"] == 12

        # Verify all data is readable
        for i in range(12):
            result = store.get(f"k{i}")
            assert result is not None
            assert result.shape == arr.shape
    finally:
        store.destroy()


def test_multi_chunk_get_delete():
    """Values across multiple chunks can be read and deleted."""
    store = SharedStore.create("test_mc_del", chunk_size_mb=1, max_entries=64)
    try:
        arr = np.zeros(100_000, dtype=np.uint8)
        for i in range(12):
            store.put(f"k{i}", arr)

        # Delete from the first chunk
        assert store.delete("k0") is True
        assert store.get("k0") is None

        # Delete from a later chunk
        assert store.delete("k11") is True
        assert store.get("k11") is None

        info = store.info()
        assert info["entry_count"] == 10
    finally:
        store.destroy()


def test_connect_reads_config_from_header():
    """connect() should read max_entries and chunk_size from the header."""
    store = SharedStore.create("test_mc_conn", chunk_size_mb=1, max_entries=128)
    try:
        arr = np.zeros(100_000, dtype=np.uint8)
        for i in range(12):
            store.put(f"k{i}", arr)

        # Connect without specifying max_entries
        store2 = SharedStore.connect("test_mc_conn", store.locks())
        for i in range(12):
            result = store2.get(f"k{i}")
            assert result is not None
        info = store2.info()
        assert info["max_entries"] == 128
        assert info["chunk_count"] >= 2
        store2.close()
    finally:
        store.destroy()


def test_info_across_chunks():
    """info() should aggregate stats from all chunks."""
    store = SharedStore.create("test_mc_info", chunk_size_mb=1, max_entries=64)
    try:
        arr = np.zeros(100_000, dtype=np.uint8)
        for i in range(12):
            store.put(f"k{i}", arr)

        info = store.info()
        assert info["chunk_count"] >= 2
        assert info["total_bytes"] == info["chunk_data_size"] * info["chunk_count"]
        assert info["used_blocks"] == 12
    finally:
        store.destroy()


def test_value_too_large_raises():
    """A value larger than chunk_data_size should raise OutOfMemoryError."""
    store = SharedStore.create("test_mc_big", chunk_size_mb=1, max_entries=64)
    try:
        # 1MB chunk, try to store >1MB
        huge = np.zeros(2_000_000, dtype=np.uint8)
        with pytest.raises(OutOfMemoryError, match="too large"):
            store.put("huge", huge)
    finally:
        store.destroy()
