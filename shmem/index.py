"""Open-addressing hash table with linear probing over the shared memory index region.

Each slot is an IndexEntry (256 bytes).  Keys are UTF-8 strings (max 127 chars + null).
Deleted entries use a TOMBSTONE state so probe chains aren't broken.
"""

import ctypes
import hashlib
from typing import Iterator

from .layout import (
    INDEX_ENTRY_SIZE,
    MAX_KEY_LEN,
    IndexEntry,
    ENTRY_EMPTY,
    ENTRY_OCCUPIED,
    ENTRY_TOMBSTONE,
)
from .errors import StoreFullError


def hash_key(key: str) -> int:
    """Hash a string key to a 64-bit unsigned integer using MD5."""
    digest = hashlib.md5(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


class HashIndex:
    """Open-addressing hash table mapped onto a contiguous buffer.

    Parameters
    ----------
    buf : writable buffer
        The index region of the shared memory (max_entries × 256 bytes).
    max_entries : int
        Number of slots in the hash table.
    """

    def __init__(self, buf, max_entries: int):
        self._buf = buf
        self._max_entries = max_entries

    def _buf_ptr(self, offset: int) -> int:
        if isinstance(self._buf, memoryview):
            return ctypes.addressof(ctypes.c_char.from_buffer(self._buf, offset))
        elif isinstance(self._buf, bytearray):
            return ctypes.addressof((ctypes.c_char * len(self._buf)).from_buffer(self._buf)) + offset
        else:
            return ctypes.addressof(ctypes.c_char.from_buffer(self._buf, offset))

    def _read_entry(self, slot: int) -> IndexEntry:
        entry = IndexEntry()
        offset = slot * INDEX_ENTRY_SIZE
        ctypes.memmove(ctypes.addressof(entry), self._buf_ptr(offset), INDEX_ENTRY_SIZE)
        return entry

    def _write_entry(self, slot: int, entry: IndexEntry) -> None:
        offset = slot * INDEX_ENTRY_SIZE
        ctypes.memmove(self._buf_ptr(offset), ctypes.addressof(entry), INDEX_ENTRY_SIZE)

    def _key_bytes(self, key: str) -> bytes:
        encoded = key.encode("utf-8")
        if len(encoded) >= MAX_KEY_LEN:
            raise ValueError(f"Key too long: {len(encoded)} bytes (max {MAX_KEY_LEN - 1})")
        return encoded

    def find(self, key: str) -> IndexEntry | None:
        """Look up a key.  Returns the IndexEntry if found, else None."""
        key_b = self._key_bytes(key)
        start = hash_key(key) % self._max_entries
        for i in range(self._max_entries):
            slot = (start + i) % self._max_entries
            entry = self._read_entry(slot)
            if entry.state == ENTRY_EMPTY:
                return None
            if entry.state == ENTRY_OCCUPIED and entry.key.rstrip(b"\x00") == key_b:
                return entry
            # TOMBSTONE: keep probing
        return None

    def find_slot(self, key: str) -> int:
        """Return the slot index for an occupied key, or -1 if not found."""
        key_b = self._key_bytes(key)
        start = hash_key(key) % self._max_entries
        for i in range(self._max_entries):
            slot = (start + i) % self._max_entries
            entry = self._read_entry(slot)
            if entry.state == ENTRY_EMPTY:
                return -1
            if entry.state == ENTRY_OCCUPIED and entry.key.rstrip(b"\x00") == key_b:
                return slot
        return -1

    def insert(
        self,
        key: str,
        data_offset: int,
        data_size: int,
        dtype_str: str = "",
        ndim: int = 0,
        shape: tuple[int, ...] = (),
    ) -> int:
        """Insert or update a key.  Returns the slot index used.

        Raises StoreFullError if no slot is available.
        """
        key_b = self._key_bytes(key)
        start = hash_key(key) % self._max_entries
        first_tombstone = -1

        for i in range(self._max_entries):
            slot = (start + i) % self._max_entries
            entry = self._read_entry(slot)

            if entry.state == ENTRY_EMPTY:
                # Use tombstone slot if we passed one, otherwise use this empty slot
                target = first_tombstone if first_tombstone >= 0 else slot
                self._write_new_entry(target, key_b, data_offset, data_size, dtype_str, ndim, shape)
                return target

            if entry.state == ENTRY_TOMBSTONE:
                if first_tombstone < 0:
                    first_tombstone = slot
                continue

            if entry.key.rstrip(b"\x00") == key_b:
                # Update existing entry
                self._write_new_entry(slot, key_b, data_offset, data_size, dtype_str, ndim, shape)
                return slot

        # All slots probed — use tombstone if available
        if first_tombstone >= 0:
            self._write_new_entry(first_tombstone, key_b, data_offset, data_size, dtype_str, ndim, shape)
            return first_tombstone

        raise StoreFullError(f"Hash table is full ({self._max_entries} entries)")

    def _write_new_entry(
        self,
        slot: int,
        key_b: bytes,
        data_offset: int,
        data_size: int,
        dtype_str: str,
        ndim: int,
        shape: tuple[int, ...],
    ) -> None:
        entry = IndexEntry()
        entry.key = key_b
        entry.data_offset = data_offset
        entry.data_size = data_size
        entry.dtype_str = dtype_str.encode("utf-8") if dtype_str else b""
        entry.ndim = ndim
        for j, s in enumerate(shape):
            entry.shape[j] = s
        entry.state = ENTRY_OCCUPIED
        self._write_entry(slot, entry)

    def remove(self, key: str) -> IndexEntry | None:
        """Remove a key by marking its slot as TOMBSTONE.

        Returns the old entry (with data_offset/data_size for deallocation),
        or None if the key was not found.
        """
        key_b = self._key_bytes(key)
        start = hash_key(key) % self._max_entries
        for i in range(self._max_entries):
            slot = (start + i) % self._max_entries
            entry = self._read_entry(slot)
            if entry.state == ENTRY_EMPTY:
                return None
            if entry.state == ENTRY_OCCUPIED and entry.key.rstrip(b"\x00") == key_b:
                # Mark as tombstone
                tomb = IndexEntry()
                tomb.state = ENTRY_TOMBSTONE
                self._write_entry(slot, tomb)
                return entry
        return None

    def iterate(self) -> Iterator[IndexEntry]:
        """Yield all occupied entries."""
        for slot in range(self._max_entries):
            entry = self._read_entry(slot)
            if entry.state == ENTRY_OCCUPIED:
                yield entry

    def keys(self) -> list[str]:
        """Return all occupied keys as strings."""
        result = []
        for entry in self.iterate():
            result.append(entry.key.rstrip(b"\x00").decode("utf-8"))
        return result

    def count(self) -> int:
        """Count occupied entries."""
        n = 0
        for slot in range(self._max_entries):
            entry = self._read_entry(slot)
            if entry.state == ENTRY_OCCUPIED:
                n += 1
        return n
