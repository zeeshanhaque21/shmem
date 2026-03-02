"""Integration tests for SharedStore — put/get round-trips, zero-copy, bytes, lifecycle."""

import numpy as np
import pytest
from shmem import SharedStore


STORE_NAME = "test_store"


@pytest.fixture
def store():
    s = SharedStore.create(STORE_NAME, size_mb=1, max_entries=64)
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
    store.put("x", np.zeros(100, dtype=np.uint8))
    info = store.info()
    assert info["entry_count"] == 1
    assert info["used_blocks"] == 1


def test_connect(store):
    """Test that a second store instance can read data written by the first."""
    arr = np.arange(50, dtype=np.float32)
    store.put("shared", arr)

    store2 = SharedStore.connect(STORE_NAME, store.locks(), max_entries=64)
    result = store2.get("shared")
    np.testing.assert_array_equal(result, arr)
    store2.close()


def test_connect_can_write(store):
    """Test that a connected store can write data readable by the creator."""
    store2 = SharedStore.connect(STORE_NAME, store.locks(), max_entries=64)
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
