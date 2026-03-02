"""Open-addressing hash table with linear probing over the shared memory index region.

Each slot is an IndexEntry (256 bytes).  Keys are UTF-8 strings (max 127 chars + null).
Deleted entries use a TOMBSTONE state so probe chains aren't broken.

Probing uses direct buffer reads (state byte + key slice) to avoid
copying the full 256-byte entry on every probe.  Only the matching
entry's fields are read via struct.unpack_from.
"""

import ctypes
import hashlib
import struct
from typing import Iterator, NamedTuple

from .layout import (
    INDEX_ENTRY_SIZE,
    MAX_KEY_LEN,
    MAX_SHAPE_DIMS,
    IndexEntry,
    ENTRY_EMPTY,
    ENTRY_OCCUPIED,
    ENTRY_TOMBSTONE,
)
from .errors import StoreFullError

# Field offsets within a 256-byte IndexEntry (_pack_=1):
#   key:          0   (128 bytes)
#   data_offset:  128 (uint64, 8 bytes)
#   data_size:    136 (uint64, 8 bytes)
#   dtype_str:    144 (8 bytes)
#   ndim:         152 (uint32, 4 bytes)
#   shape:        156 (uint64 × 8 = 64 bytes)
#   state:        220 (uint8, 1 byte)
#   _pad:         221 (35 bytes)
_F_KEY = 0
_F_DATA_OFFSET = 128
_F_DATA_SIZE = 136
_F_DTYPE_STR = 144
_F_NDIM = 152
_F_SHAPE = 156
_F_STATE = 220

# struct format for reading data fields in one call:
# data_offset(Q) + data_size(Q) + dtype_str(8s) + ndim(I)
_DATA_FIELDS_FMT = struct.Struct("<QQ8sI")
_DATA_FIELDS_SIZE = _DATA_FIELDS_FMT.size  # 28

# struct format for writing entry fields (excluding key and state):
_SHAPE_FMT = struct.Struct(f"<{MAX_SHAPE_DIMS}Q")


def hash_key(key: str) -> int:
    """Hash a string key to a 64-bit unsigned integer using MD5."""
    digest = hashlib.md5(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


class FoundEntry(NamedTuple):
    """Lightweight result from index lookups — avoids 256B ctypes memmove."""

    data_offset: int
    data_size: int
    dtype_str: bytes   # raw bytes, caller decodes
    ndim: int
    shape: tuple[int, ...]


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

    # -- fast direct buffer access for probing --------------------------------

    def _slot_offset(self, slot: int) -> int:
        return slot * INDEX_ENTRY_SIZE

    def _read_state(self, slot: int) -> int:
        return self._buf[self._slot_offset(slot) + _F_STATE]

    def _read_key_bytes(self, slot: int) -> memoryview | bytes:
        off = self._slot_offset(slot)
        return self._buf[off + _F_KEY : off + _F_KEY + MAX_KEY_LEN]

    def _key_matches(self, slot: int, key_b: bytes) -> bool:
        """Check if the key at *slot* matches *key_b* without copying the full entry."""
        off = self._slot_offset(slot) + _F_KEY
        key_len = len(key_b)
        # Compare the key bytes, then check null terminator
        if self._buf[off : off + key_len] != key_b:
            return False
        # Ensure the next byte is null (not a prefix match)
        if key_len < MAX_KEY_LEN and self._buf[off + key_len] != 0:
            return False
        return True

    def _read_data_fields(self, slot: int) -> FoundEntry:
        """Read just the data fields of a matched entry (28 + shape bytes)."""
        off = self._slot_offset(slot) + _F_DATA_OFFSET
        data_offset, data_size, dtype_str, ndim = _DATA_FIELDS_FMT.unpack_from(
            self._buf, off
        )
        # Read shape dimensions
        shape_off = self._slot_offset(slot) + _F_SHAPE
        shape = struct.unpack_from(f"<{ndim}Q", self._buf, shape_off) if ndim > 0 else ()
        return FoundEntry(data_offset, data_size, dtype_str, ndim, shape)

    # -- ctypes-based full entry read/write (used for writes and iteration) ---

    def _buf_ptr(self, offset: int) -> int:
        if isinstance(self._buf, memoryview):
            return ctypes.addressof(ctypes.c_char.from_buffer(self._buf, offset))
        elif isinstance(self._buf, bytearray):
            return ctypes.addressof(
                (ctypes.c_char * len(self._buf)).from_buffer(self._buf)
            ) + offset
        else:
            return ctypes.addressof(ctypes.c_char.from_buffer(self._buf, offset))

    def _read_entry(self, slot: int) -> IndexEntry:
        entry = IndexEntry()
        offset = self._slot_offset(slot)
        ctypes.memmove(ctypes.addressof(entry), self._buf_ptr(offset), INDEX_ENTRY_SIZE)
        return entry

    def _write_entry(self, slot: int, entry: IndexEntry) -> None:
        offset = self._slot_offset(slot)
        ctypes.memmove(self._buf_ptr(offset), ctypes.addressof(entry), INDEX_ENTRY_SIZE)

    def _key_bytes(self, key: str) -> bytes:
        encoded = key.encode("utf-8")
        if len(encoded) >= MAX_KEY_LEN:
            raise ValueError(f"Key too long: {len(encoded)} bytes (max {MAX_KEY_LEN - 1})")
        return encoded

    # -- public API -----------------------------------------------------------

    def find(self, key: str) -> FoundEntry | None:
        """Look up a key.  Returns a FoundEntry if found, else None."""
        key_b = self._key_bytes(key)
        start = hash_key(key) % self._max_entries
        for i in range(self._max_entries):
            slot = (start + i) % self._max_entries
            state = self._read_state(slot)
            if state == ENTRY_EMPTY:
                return None
            if state == ENTRY_OCCUPIED and self._key_matches(slot, key_b):
                return self._read_data_fields(slot)
            # TOMBSTONE: keep probing
        return None

    def find_slot(self, key: str) -> int:
        """Return the slot index for an occupied key, or -1 if not found."""
        key_b = self._key_bytes(key)
        start = hash_key(key) % self._max_entries
        for i in range(self._max_entries):
            slot = (start + i) % self._max_entries
            state = self._read_state(slot)
            if state == ENTRY_EMPTY:
                return -1
            if state == ENTRY_OCCUPIED and self._key_matches(slot, key_b):
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
            state = self._read_state(slot)

            if state == ENTRY_EMPTY:
                target = first_tombstone if first_tombstone >= 0 else slot
                self._write_new_entry(target, key_b, data_offset, data_size, dtype_str, ndim, shape)
                return target

            if state == ENTRY_TOMBSTONE:
                if first_tombstone < 0:
                    first_tombstone = slot
                continue

            if self._key_matches(slot, key_b):
                self._write_new_entry(slot, key_b, data_offset, data_size, dtype_str, ndim, shape)
                return slot

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

    def remove(self, key: str) -> FoundEntry | None:
        """Remove a key by marking its slot as TOMBSTONE.

        Returns the old FoundEntry (with data_offset/data_size for deallocation),
        or None if the key was not found.
        """
        key_b = self._key_bytes(key)
        start = hash_key(key) % self._max_entries
        for i in range(self._max_entries):
            slot = (start + i) % self._max_entries
            state = self._read_state(slot)
            if state == ENTRY_EMPTY:
                return None
            if state == ENTRY_OCCUPIED and self._key_matches(slot, key_b):
                old = self._read_data_fields(slot)
                # Write tombstone: just set the state byte
                self._buf[self._slot_offset(slot) + _F_STATE] = ENTRY_TOMBSTONE
                return old
        return None

    def iterate(self) -> Iterator[FoundEntry]:
        """Yield all occupied entries."""
        for slot in range(self._max_entries):
            if self._read_state(slot) == ENTRY_OCCUPIED:
                yield self._read_data_fields(slot)

    def keys(self) -> list[str]:
        """Return all occupied keys as strings."""
        result = []
        for slot in range(self._max_entries):
            if self._read_state(slot) == ENTRY_OCCUPIED:
                raw = bytes(self._read_key_bytes(slot))
                null_idx = raw.find(b"\x00")
                if null_idx >= 0:
                    raw = raw[:null_idx]
                result.append(raw.decode("utf-8"))
        return result

    def count(self) -> int:
        """Count occupied entries."""
        n = 0
        for slot in range(self._max_entries):
            if self._read_state(slot) == ENTRY_OCCUPIED:
                n += 1
        return n
