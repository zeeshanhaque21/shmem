"""SharedStore — main API class wrapping shared memory with a KV interface.

Provides put/get/delete for numpy arrays and raw bytes with zero-copy reads.
Uses a readers-writers lock for safe multi-process access.
"""

import ctypes
import multiprocessing
import struct
from multiprocessing import shared_memory
from dataclasses import dataclass

import numpy as np

from .layout import (
    MAGIC,
    VERSION,
    HEADER_SIZE,
    INDEX_ENTRY_SIZE,
    BLOCK_HEADER_SIZE,
    StoreHeader,
    compute_offsets,
)
from .allocator import BlockAllocator
from .index import HashIndex
from .errors import StoreCorruptedError


LOCK_TIMEOUT = 5.0  # seconds

# Field offsets within StoreHeader (packed, _pack_=1):
#   magic:          offset  0, uint32  (4)
#   version:        offset  4, uint32  (4)
#   max_entries:    offset  8, uint32  (4)
#   entry_count:    offset 12, uint32  (4)
#   index_offset:   offset 16, uint64  (8)
#   data_offset:    offset 24, uint64  (8)
#   data_size:      offset 32, uint64  (8)
#   free_list_head: offset 40, int64   (8)
#   readers_count:  offset 48, int32   (4)
#   _pad:           offset 52, 12 bytes
_OFF_MAGIC = 0
_OFF_VERSION = 4
_OFF_MAX_ENTRIES = 8
_OFF_ENTRY_COUNT = 12
_OFF_INDEX_OFFSET = 16
_OFF_DATA_OFFSET = 24
_OFF_DATA_SIZE = 32
_OFF_FREE_LIST_HEAD = 40
_OFF_READERS_COUNT = 48


@dataclass
class StoreLocks:
    """Container for the two multiprocessing locks used by the store."""

    write_lock: multiprocessing.Lock
    read_lock: multiprocessing.Lock


class _HeaderFields:
    """Field-level access to header fields in shared memory.

    Uses struct.pack_into / unpack_from to read/write individual fields,
    avoiding read-modify-write races on the full 64-byte header.
    """

    def __init__(self, buf):
        self._buf = buf

    # -- individual field accessors -------------------------------------------

    def get_magic(self) -> int:
        return struct.unpack_from("<I", self._buf, _OFF_MAGIC)[0]

    def get_version(self) -> int:
        return struct.unpack_from("<I", self._buf, _OFF_VERSION)[0]

    def get_max_entries(self) -> int:
        return struct.unpack_from("<I", self._buf, _OFF_MAX_ENTRIES)[0]

    def get_entry_count(self) -> int:
        return struct.unpack_from("<I", self._buf, _OFF_ENTRY_COUNT)[0]

    def set_entry_count(self, value: int) -> None:
        struct.pack_into("<I", self._buf, _OFF_ENTRY_COUNT, value)

    def get_data_size(self) -> int:
        return struct.unpack_from("<Q", self._buf, _OFF_DATA_SIZE)[0]

    def get_free_list_head(self) -> int:
        return struct.unpack_from("<q", self._buf, _OFF_FREE_LIST_HEAD)[0]

    def set_free_list_head(self, value: int) -> None:
        struct.pack_into("<q", self._buf, _OFF_FREE_LIST_HEAD, value)

    def get_readers_count(self) -> int:
        return struct.unpack_from("<i", self._buf, _OFF_READERS_COUNT)[0]

    def set_readers_count(self, value: int) -> None:
        struct.pack_into("<i", self._buf, _OFF_READERS_COUNT, value)

    # -- callable interface for the allocator ---------------------------------

    def __call__(self, value=None):
        """Allocator callback: call(None) to get free_list_head, call(int) to set."""
        if value is None:
            return self.get_free_list_head()
        self.set_free_list_head(value)

    def get(self) -> int:
        return self.get_free_list_head()

    def set(self, value: int) -> None:
        self.set_free_list_head(value)

    # -- bulk write (only used during init) -----------------------------------

    def init_all(
        self,
        magic: int,
        version: int,
        max_entries: int,
        index_offset: int,
        data_offset: int,
        data_size: int,
    ) -> None:
        """Write all header fields at once (used only at creation time)."""
        struct.pack_into("<I", self._buf, _OFF_MAGIC, magic)
        struct.pack_into("<I", self._buf, _OFF_VERSION, version)
        struct.pack_into("<I", self._buf, _OFF_MAX_ENTRIES, max_entries)
        struct.pack_into("<I", self._buf, _OFF_ENTRY_COUNT, 0)
        struct.pack_into("<Q", self._buf, _OFF_INDEX_OFFSET, index_offset)
        struct.pack_into("<Q", self._buf, _OFF_DATA_OFFSET, data_offset)
        struct.pack_into("<Q", self._buf, _OFF_DATA_SIZE, data_size)
        struct.pack_into("<q", self._buf, _OFF_FREE_LIST_HEAD, -1)
        struct.pack_into("<i", self._buf, _OFF_READERS_COUNT, 0)


class SharedStore:
    """Zero-copy shared memory key-value store for numpy arrays and raw bytes.

    Use ``create()`` in the parent process and ``connect()`` in child processes.
    """

    def __init__(
        self,
        shm: shared_memory.SharedMemory,
        max_entries: int,
        locks: StoreLocks,
        *,
        is_creator: bool = False,
    ):
        self._shm = shm
        self._max_entries = max_entries
        self._locks = locks
        self._is_creator = is_creator

        self._buf: memoryview = shm.buf

        # Map sub-regions
        index_offset, data_offset = compute_offsets(max_entries)
        data_size = len(self._buf) - data_offset

        self._index_buf = self._buf[index_offset : index_offset + max_entries * INDEX_ENTRY_SIZE]
        self._data_buf = self._buf[data_offset : data_offset + data_size]

        self._hdr = _HeaderFields(self._buf)
        self._index = HashIndex(self._index_buf, max_entries)
        self._allocator = BlockAllocator(self._data_buf)

        self._data_offset = data_offset

    @classmethod
    def create(
        cls,
        name: str,
        size_mb: int = 256,
        max_entries: int = 1024,
    ) -> "SharedStore":
        """Create a new shared memory store (call from the parent process)."""
        index_offset, data_offset = compute_offsets(max_entries)
        total_size = data_offset + size_mb * 1024 * 1024

        shm = shared_memory.SharedMemory(name=name, create=True, size=total_size)

        locks = StoreLocks(
            write_lock=multiprocessing.Lock(),
            read_lock=multiprocessing.Lock(),
        )

        store = cls(shm, max_entries, locks, is_creator=True)
        store._hdr.init_all(
            magic=MAGIC,
            version=VERSION,
            max_entries=max_entries,
            index_offset=index_offset,
            data_offset=data_offset,
            data_size=total_size - data_offset,
        )
        store._allocator.init(store._hdr)

        return store

    @classmethod
    def connect(
        cls,
        name: str,
        locks: StoreLocks,
        max_entries: int = 1024,
    ) -> "SharedStore":
        """Attach to an existing shared memory store (call from child processes)."""
        shm = shared_memory.SharedMemory(name=name, create=False)

        store = cls(shm, max_entries, locks, is_creator=False)
        store._verify_header()

        return store

    def locks(self) -> StoreLocks:
        """Return the StoreLocks for passing to child processes."""
        return self._locks

    # -- Header helpers -------------------------------------------------------

    def _verify_header(self) -> None:
        magic = self._hdr.get_magic()
        if magic != MAGIC:
            raise StoreCorruptedError(
                f"Bad magic: 0x{magic:08X} (expected 0x{MAGIC:08X})"
            )
        version = self._hdr.get_version()
        if version != VERSION:
            raise StoreCorruptedError(
                f"Version mismatch: {version} (expected {VERSION})"
            )

    def _update_entry_count(self, delta: int) -> None:
        cur = self._hdr.get_entry_count()
        self._hdr.set_entry_count(max(0, cur + delta))

    # -- Readers-writers lock -------------------------------------------------

    def _acquire_read(self) -> None:
        if not self._locks.read_lock.acquire(timeout=LOCK_TIMEOUT):
            raise TimeoutError("Timed out acquiring read lock")
        try:
            rc = self._hdr.get_readers_count()
            rc += 1
            if rc == 1:
                if not self._locks.write_lock.acquire(timeout=LOCK_TIMEOUT):
                    raise TimeoutError("Timed out acquiring write lock for first reader")
            self._hdr.set_readers_count(rc)
        finally:
            self._locks.read_lock.release()

    def _release_read(self) -> None:
        if not self._locks.read_lock.acquire(timeout=LOCK_TIMEOUT):
            raise TimeoutError("Timed out acquiring read lock for release")
        try:
            rc = self._hdr.get_readers_count()
            rc -= 1
            self._hdr.set_readers_count(rc)
            if rc == 0:
                self._locks.write_lock.release()
        finally:
            self._locks.read_lock.release()

    def _acquire_write(self) -> None:
        if not self._locks.write_lock.acquire(timeout=LOCK_TIMEOUT):
            raise TimeoutError("Timed out acquiring write lock")

    def _release_write(self) -> None:
        self._locks.write_lock.release()

    # -- Public API -----------------------------------------------------------

    def put(self, key: str, array: np.ndarray) -> None:
        """Copy a numpy array into shared memory under *key*.

        If *key* already exists, the old data block is freed first.
        """
        self._acquire_write()
        try:
            self._put_impl(key, array)
        finally:
            self._release_write()

    def _put_impl(self, key: str, array: np.ndarray) -> None:
        data = np.ascontiguousarray(array)
        nbytes = data.nbytes
        dtype_str = data.dtype.str
        shape = data.shape

        # If key exists, free old block first
        old_entry = self._index.find(key)
        if old_entry is not None:
            old_block_off = old_entry.data_offset - BLOCK_HEADER_SIZE
            self._allocator.deallocate(
                old_block_off,
                self._hdr.get(),
                self._hdr,
            )
            self._update_entry_count(-1)

        # Allocate new block
        block_off = self._allocator.allocate(
            nbytes,
            self._hdr.get(),
            self._hdr,
        )

        # Copy data into the block (after the block header)
        payload_off = block_off + BLOCK_HEADER_SIZE
        dest = np.ndarray(data.shape, dtype=data.dtype, buffer=self._data_buf, offset=payload_off)
        np.copyto(dest, data)

        # Update index
        self._index.insert(
            key,
            data_offset=payload_off,
            data_size=nbytes,
            dtype_str=dtype_str,
            ndim=len(shape),
            shape=shape,
        )
        self._update_entry_count(1)

    def get(self, key: str) -> np.ndarray | None:
        """Return a **read-only** zero-copy numpy array view, or None if key not found."""
        self._acquire_read()
        try:
            return self._get_impl(key, writable=False)
        finally:
            self._release_read()

    def get_mut(self, key: str) -> np.ndarray | None:
        """Return a **writable** zero-copy numpy array view, or None if key not found.

        The caller is responsible for coordinating access when mutating.
        """
        self._acquire_read()
        try:
            return self._get_impl(key, writable=True)
        finally:
            self._release_read()

    def _get_impl(self, key: str, *, writable: bool) -> np.ndarray | None:
        entry = self._index.find(key)
        if entry is None:
            return None

        dtype = np.dtype(entry.dtype_str.rstrip(b"\x00").decode("utf-8"))
        shape = tuple(entry.shape[i] for i in range(entry.ndim))

        arr = np.ndarray(
            shape,
            dtype=dtype,
            buffer=self._data_buf,
            offset=entry.data_offset,
        )

        if not writable:
            arr.flags.writeable = False

        return arr

    def put_bytes(self, key: str, data: bytes | bytearray | memoryview) -> None:
        """Store raw bytes under *key*."""
        self._acquire_write()
        try:
            self._put_bytes_impl(key, data)
        finally:
            self._release_write()

    def _put_bytes_impl(self, key: str, data: bytes | bytearray | memoryview) -> None:
        nbytes = len(data)

        # Free old block if key exists
        old_entry = self._index.find(key)
        if old_entry is not None:
            old_block_off = old_entry.data_offset - BLOCK_HEADER_SIZE
            self._allocator.deallocate(
                old_block_off,
                self._hdr.get(),
                self._hdr,
            )
            self._update_entry_count(-1)

        # Allocate
        block_off = self._allocator.allocate(
            nbytes,
            self._hdr.get(),
            self._hdr,
        )

        payload_off = block_off + BLOCK_HEADER_SIZE

        # Copy bytes in
        self._data_buf[payload_off : payload_off + nbytes] = data

        # Index with empty dtype (signals raw bytes)
        self._index.insert(
            key,
            data_offset=payload_off,
            data_size=nbytes,
            dtype_str="",
            ndim=0,
            shape=(),
        )
        self._update_entry_count(1)

    def get_bytes(self, key: str) -> memoryview | None:
        """Return a zero-copy memoryview of raw bytes, or None if key not found."""
        self._acquire_read()
        try:
            entry = self._index.find(key)
            if entry is None:
                return None
            return self._data_buf[entry.data_offset : entry.data_offset + entry.data_size]
        finally:
            self._release_read()

    def delete(self, key: str) -> bool:
        """Delete a key and free its data block. Returns True if the key existed."""
        self._acquire_write()
        try:
            return self._delete_impl(key)
        finally:
            self._release_write()

    def _delete_impl(self, key: str) -> bool:
        entry = self._index.remove(key)
        if entry is None:
            return False

        block_off = entry.data_offset - BLOCK_HEADER_SIZE
        self._allocator.deallocate(
            block_off,
            self._hdr.get(),
            self._hdr,
        )
        self._update_entry_count(-1)
        return True

    def keys(self) -> list[str]:
        """Return all keys currently stored."""
        self._acquire_read()
        try:
            return self._index.keys()
        finally:
            self._release_read()

    def info(self) -> dict:
        """Return usage statistics."""
        self._acquire_read()
        try:
            alloc_stats = self._allocator.stats(self._hdr.get())
            return {
                "name": self._shm.name,
                "total_shm_bytes": len(self._buf),
                "max_entries": self._hdr.get_max_entries(),
                "entry_count": self._hdr.get_entry_count(),
                **alloc_stats,
            }
        finally:
            self._release_read()

    def _release_buffers(self) -> None:
        """Release all memoryview references so the underlying mmap can be closed."""
        self._index_buf = None
        self._data_buf = None
        self._buf = None
        self._index = None
        self._allocator = None
        self._hdr = None

    def close(self) -> None:
        """Detach from shared memory (does not destroy it)."""
        self._release_buffers()
        self._shm.close()

    def destroy(self) -> None:
        """Unlink (delete) the shared memory segment. Only the creator should call this."""
        self._release_buffers()
        self._shm.close()
        self._shm.unlink()
