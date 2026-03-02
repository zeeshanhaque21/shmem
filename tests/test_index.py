"""Tests for the hash index (open-addressing with linear probing)."""

import pytest
from shmem.index import HashIndex, hash_key
from shmem.layout import INDEX_ENTRY_SIZE, ENTRY_OCCUPIED, ENTRY_TOMBSTONE, ENTRY_EMPTY
from shmem.errors import StoreFullError

MAX_ENTRIES = 16


def make_index(max_entries=MAX_ENTRIES):
    buf = bytearray(max_entries * INDEX_ENTRY_SIZE)
    return HashIndex(buf, max_entries)


def test_hash_key_deterministic():
    h1 = hash_key("hello")
    h2 = hash_key("hello")
    assert h1 == h2
    assert isinstance(h1, int)


def test_hash_key_different():
    assert hash_key("a") != hash_key("b")


def test_insert_and_find():
    idx = make_index()
    idx.insert("key1", data_offset=100, data_size=64, dtype_str="<u1", ndim=1, shape=(64,))
    entry = idx.find("key1")
    assert entry is not None
    assert entry.data_offset == 100
    assert entry.data_size == 64
    assert entry.ndim == 1
    assert entry.shape[0] == 64


def test_find_missing():
    idx = make_index()
    assert idx.find("nonexistent") is None


def test_insert_update():
    idx = make_index()
    idx.insert("key1", data_offset=100, data_size=64)
    idx.insert("key1", data_offset=200, data_size=128)
    entry = idx.find("key1")
    assert entry.data_offset == 200
    assert entry.data_size == 128


def test_remove():
    idx = make_index()
    idx.insert("key1", data_offset=100, data_size=64)
    old = idx.remove("key1")
    assert old is not None
    assert old.data_offset == 100
    assert idx.find("key1") is None


def test_remove_missing():
    idx = make_index()
    assert idx.remove("nonexistent") is None


def test_tombstone_allows_probe():
    """Removing a key should not break probe chains for keys after it."""
    idx = make_index(max_entries=4)
    # Insert multiple keys that might collide
    idx.insert("a", data_offset=10, data_size=1)
    idx.insert("b", data_offset=20, data_size=1)
    idx.insert("c", data_offset=30, data_size=1)
    idx.insert("d", data_offset=40, data_size=1)

    # Remove one in the middle
    idx.remove("b")

    # All remaining keys should still be findable
    assert idx.find("a") is not None
    assert idx.find("c") is not None
    assert idx.find("d") is not None
    assert idx.find("b") is None


def test_insert_into_tombstone():
    """A new insert should reuse a tombstone slot."""
    idx = make_index(max_entries=4)
    idx.insert("a", data_offset=10, data_size=1)
    idx.remove("a")
    # Re-insert should succeed (reuses tombstone)
    slot = idx.insert("a", data_offset=20, data_size=2)
    entry = idx.find("a")
    assert entry is not None
    assert entry.data_offset == 20


def test_keys():
    idx = make_index()
    idx.insert("alpha", data_offset=0, data_size=1)
    idx.insert("beta", data_offset=0, data_size=1)
    idx.insert("gamma", data_offset=0, data_size=1)
    keys = sorted(idx.keys())
    assert keys == ["alpha", "beta", "gamma"]


def test_count():
    idx = make_index()
    assert idx.count() == 0
    idx.insert("a", data_offset=0, data_size=1)
    idx.insert("b", data_offset=0, data_size=1)
    assert idx.count() == 2
    idx.remove("a")
    assert idx.count() == 1


def test_full_table():
    idx = make_index(max_entries=4)
    idx.insert("a", data_offset=0, data_size=1)
    idx.insert("b", data_offset=0, data_size=1)
    idx.insert("c", data_offset=0, data_size=1)
    idx.insert("d", data_offset=0, data_size=1)
    with pytest.raises(StoreFullError):
        idx.insert("e", data_offset=0, data_size=1)


def test_full_table_with_tombstones():
    """A full table with tombstones should still allow inserts."""
    idx = make_index(max_entries=4)
    idx.insert("a", data_offset=0, data_size=1)
    idx.insert("b", data_offset=0, data_size=1)
    idx.insert("c", data_offset=0, data_size=1)
    idx.insert("d", data_offset=0, data_size=1)
    idx.remove("b")
    # Should succeed because there's a tombstone slot
    idx.insert("e", data_offset=0, data_size=1)
    assert idx.find("e") is not None


def test_key_too_long():
    idx = make_index()
    long_key = "x" * 200
    with pytest.raises(ValueError, match="Key too long"):
        idx.insert(long_key, data_offset=0, data_size=1)


def test_find_slot():
    idx = make_index()
    idx.insert("key1", data_offset=100, data_size=64)
    slot = idx.find_slot("key1")
    assert slot >= 0
    assert idx.find_slot("missing") == -1


def test_iterate():
    idx = make_index()
    idx.insert("x", data_offset=1, data_size=1)
    idx.insert("y", data_offset=2, data_size=1)
    entries = list(idx.iterate())
    assert len(entries) == 2
    offsets = sorted(e.data_offset for e in entries)
    assert offsets == [1, 2]
