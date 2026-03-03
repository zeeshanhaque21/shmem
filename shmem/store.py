"""SharedStore — main API class wrapping shared memory with a KV interface.

Provides put/get/delete for numpy arrays and raw bytes with zero-copy reads.
Uses a readers-writers lock for safe multi-process access.

The store is split into a control block (header + index) and separate data
chunks that are allocated on demand, making it elastic.
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
    CHUNK_HEADER_SIZE,
    INDEX_ENTRY_SIZE,
    BLOCK_HEADER_SIZE,
    compute_control_size,
)
from .allocator import BlockAllocator
from .index import HashIndex
from .errors import OutOfMemoryError, StoreCorruptedError


LOCK_TIMEOUT = 5.0  # seconds

# Field offsets within StoreHeader (packed, _pack_=1):
#   magic:           offset  0, uint32  (4)
#   version:         offset  4, uint32  (4)
#   max_entries:     offset  8, uint32  (4)
#   entry_count:     offset 12, uint32  (4)
#   index_offset:    offset 16, uint64  (8)
#   chunk_data_size: offset 24, uint64  (8)
#   chunk_count:     offset 32, uint32  (4)
#   _reserved:       offset 36, uint32  (4)
#   _unused:         offset 40, int64   (8)
#   readers_count:   offset 48, int32   (4)
#   _pad:            offset 52, 12 bytes
_OFF_MAGIC = 0
_OFF_VERSION = 4
_OFF_MAX_ENTRIES = 8
_OFF_ENTRY_COUNT = 12
_OFF_INDEX_OFFSET = 16
_OFF_CHUNK_DATA_SIZE = 24
_OFF_CHUNK_COUNT = 32
_OFF_READERS_COUNT = 48

# ChunkHeader field offsets (within each data chunk SharedMemory):
#   free_list_head: offset 0, int64  (8)
#   data_size:      offset 8, uint64 (8)
#   _pad:           offset 16, 16 bytes
_COFF_FREE_LIST_HEAD = 0
_COFF_DATA_SIZE = 8


@dataclass
class StoreLocks:
    """Container for the two multiprocessing locks used by the store."""

    write_lock: multiprocessing.Lock
    read_lock: multiprocessing.Lock


class _HeaderFields:
    """Field-level access to header fields in the control block shared memory.

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

    def get_chunk_data_size(self) -> int:
        return struct.unpack_from("<Q", self._buf, _OFF_CHUNK_DATA_SIZE)[0]

    def get_chunk_count(self) -> int:
        return struct.unpack_from("<I", self._buf, _OFF_CHUNK_COUNT)[0]

    def set_chunk_count(self, value: int) -> None:
        struct.pack_into("<I", self._buf, _OFF_CHUNK_COUNT, value)

    def get_readers_count(self) -> int:
        return struct.unpack_from("<i", self._buf, _OFF_READERS_COUNT)[0]

    def set_readers_count(self, value: int) -> None:
        struct.pack_into("<i", self._buf, _OFF_READERS_COUNT, value)

    # -- bulk write (only used during init) -----------------------------------

    def init_all(
        self,
        magic: int,
        version: int,
        max_entries: int,
        index_offset: int,
        chunk_data_size: int,
    ) -> None:
        """Write all header fields at once (used only at creation time)."""
        struct.pack_into("<I", self._buf, _OFF_MAGIC, magic)
        struct.pack_into("<I", self._buf, _OFF_VERSION, version)
        struct.pack_into("<I", self._buf, _OFF_MAX_ENTRIES, max_entries)
        struct.pack_into("<I", self._buf, _OFF_ENTRY_COUNT, 0)
        struct.pack_into("<Q", self._buf, _OFF_INDEX_OFFSET, index_offset)
        struct.pack_into("<Q", self._buf, _OFF_CHUNK_DATA_SIZE, chunk_data_size)
        struct.pack_into("<I", self._buf, _OFF_CHUNK_COUNT, 0)
        struct.pack_into("<i", self._buf, _OFF_READERS_COUNT, 0)


class _ChunkHeaderFields:
    """Field-level access to a ChunkHeader in a data chunk's shared memory.

    Also implements the callable interface expected by the BlockAllocator
    for free_list_head get/set.
    """

    def __init__(self, buf):
        self._buf = buf

    def get_free_list_head(self) -> int:
        return struct.unpack_from("<q", self._buf, _COFF_FREE_LIST_HEAD)[0]

    def set_free_list_head(self, value: int) -> None:
        struct.pack_into("<q", self._buf, _COFF_FREE_LIST_HEAD, value)

    def init(self, data_size: int) -> None:
        """Initialize the chunk header fields."""
        struct.pack_into("<q", self._buf, _COFF_FREE_LIST_HEAD, -1)
        struct.pack_into("<Q", self._buf, _COFF_DATA_SIZE, data_size)

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


@dataclass
class _ChunkInfo:
    """Per-chunk state."""

    shm: shared_memory.SharedMemory
    buf: memoryview
    data_buf: memoryview
    allocator: BlockAllocator
    chunk_hdr: _ChunkHeaderFields


class SharedStore:
    """Zero-copy shared memory key-value store for numpy arrays and raw bytes.

    Use ``create()`` in the parent process and ``connect()`` in child processes.
    The store automatically grows by allocating new data chunks on demand.
    """

    def __init__(
        self,
        ctrl_shm: shared_memory.SharedMemory,
        max_entries: int,
        locks: StoreLocks,
        *,
        is_creator: bool = False,
        name: str,
        chunk_data_size: int,
    ):
        self._ctrl_shm = ctrl_shm
        self._max_entries = max_entries
        self._locks = locks
        self._is_creator = is_creator
        self._name = name
        self._chunk_data_size = chunk_data_size

        self._ctrl_buf: memoryview = ctrl_shm.buf

        # Map sub-regions of the control block
        index_offset = HEADER_SIZE
        self._index_buf = self._ctrl_buf[index_offset : index_offset + max_entries * INDEX_ENTRY_SIZE]

        self._hdr = _HeaderFields(self._ctrl_buf)
        self._index = HashIndex(self._index_buf, max_entries)

        self._chunks: list[_ChunkInfo | None] = []

    @classmethod
    def create(
        cls,
        name: str,
        chunk_size_mb: int = 64,
        max_entries: int = 1024,
    ) -> "SharedStore":
        """Create a new shared memory store (call from the parent process)."""
        control_size = compute_control_size(max_entries)
        chunk_data_size = chunk_size_mb * 1024 * 1024

        ctrl_shm = shared_memory.SharedMemory(name=name, create=True, size=control_size)

        locks = StoreLocks(
            write_lock=multiprocessing.Lock(),
            read_lock=multiprocessing.Lock(),
        )

        store = cls(
            ctrl_shm, max_entries, locks,
            is_creator=True,
            name=name,
            chunk_data_size=chunk_data_size,
        )
        store._hdr.init_all(
            magic=MAGIC,
            version=VERSION,
            max_entries=max_entries,
            index_offset=HEADER_SIZE,
            chunk_data_size=chunk_data_size,
        )

        # Create the first data chunk
        store._create_chunk()

        return store

    @classmethod
    def connect(
        cls,
        name: str,
        locks: StoreLocks,
    ) -> "SharedStore":
        """Attach to an existing shared memory store (call from child processes).

        Reads max_entries and chunk_data_size from the header, so the caller
        does not need to specify them.
        """
        ctrl_shm = shared_memory.SharedMemory(name=name, create=False)
        buf = ctrl_shm.buf

        # Read header fields to discover configuration
        max_entries = struct.unpack_from("<I", buf, _OFF_MAX_ENTRIES)[0]
        chunk_data_size = struct.unpack_from("<Q", buf, _OFF_CHUNK_DATA_SIZE)[0]
        chunk_count = struct.unpack_from("<I", buf, _OFF_CHUNK_COUNT)[0]

        store = cls(
            ctrl_shm, max_entries, locks,
            is_creator=False,
            name=name,
            chunk_data_size=chunk_data_size,
        )
        store._verify_header()

        # Open all existing chunks
        for i in range(chunk_count):
            store._open_chunk(i)

        return store

    def locks(self) -> StoreLocks:
        """Return the StoreLocks for passing to child processes."""
        return self._locks

    # -- Chunk management -----------------------------------------------------

    def _open_chunk(self, i: int) -> _ChunkInfo:
        """Open an existing data chunk by index."""
        chunk_name = f"{self._name}_{i}"
        shm = shared_memory.SharedMemory(name=chunk_name, create=False)
        buf = shm.buf
        data_buf = buf[CHUNK_HEADER_SIZE:]
        chunk_hdr = _ChunkHeaderFields(buf)
        allocator = BlockAllocator(data_buf)

        info = _ChunkInfo(
            shm=shm,
            buf=buf,
            data_buf=data_buf,
            allocator=allocator,
            chunk_hdr=chunk_hdr,
        )

        # Extend list if needed and set at index i
        while len(self._chunks) <= i:
            self._chunks.append(None)
        self._chunks[i] = info
        return info

    def _create_chunk(self) -> _ChunkInfo:
        """Create a new data chunk and initialize its allocator."""
        i = self._hdr.get_chunk_count()
        chunk_name = f"{self._name}_{i}"
        chunk_total_size = CHUNK_HEADER_SIZE + self._chunk_data_size

        shm = shared_memory.SharedMemory(name=chunk_name, create=True, size=chunk_total_size)
        buf = shm.buf
        data_buf = buf[CHUNK_HEADER_SIZE:]
        chunk_hdr = _ChunkHeaderFields(buf)
        chunk_hdr.init(self._chunk_data_size)
        allocator = BlockAllocator(data_buf)
        allocator.init(chunk_hdr)

        info = _ChunkInfo(
            shm=shm,
            buf=buf,
            data_buf=data_buf,
            allocator=allocator,
            chunk_hdr=chunk_hdr,
        )

        while len(self._chunks) <= i:
            self._chunks.append(None)
        self._chunks[i] = info

        self._hdr.set_chunk_count(i + 1)
        return info

    def _ensure_chunk(self, i: int) -> _ChunkInfo:
        """Lazily open chunk i if not yet in the local list."""
        if i < len(self._chunks) and self._chunks[i] is not None:
            return self._chunks[i]
        return self._open_chunk(i)

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

    # -- Virtual offset encoding/decoding ------------------------------------

    def _encode_offset(self, chunk_index: int, local_offset: int) -> int:
        """Encode chunk index and local offset into a virtual offset."""
        return chunk_index * self._chunk_data_size + local_offset

    def _decode_offset(self, virtual_offset: int) -> tuple[int, int]:
        """Decode a virtual offset into (chunk_index, local_offset)."""
        return divmod(virtual_offset, self._chunk_data_size)

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

        if nbytes > self._chunk_data_size:
            raise OutOfMemoryError(
                f"Value too large ({nbytes} bytes) for chunk_data_size "
                f"({self._chunk_data_size} bytes). Use a larger chunk_size_mb."
            )

        # If key exists, free old block first
        old_entry = self._index.find(key)
        if old_entry is not None:
            chunk_idx, local_payload_off = self._decode_offset(old_entry.data_offset)
            chunk = self._ensure_chunk(chunk_idx)
            old_block_off = local_payload_off - BLOCK_HEADER_SIZE
            chunk.allocator.deallocate(
                old_block_off,
                chunk.chunk_hdr.get(),
                chunk.chunk_hdr,
            )
            self._update_entry_count(-1)

        # Try allocating from each existing chunk
        block_off = None
        alloc_chunk_idx = None
        for ci in range(self._hdr.get_chunk_count()):
            chunk = self._ensure_chunk(ci)
            try:
                block_off = chunk.allocator.allocate(
                    nbytes,
                    chunk.chunk_hdr.get(),
                    chunk.chunk_hdr,
                )
                alloc_chunk_idx = ci
                break
            except OutOfMemoryError:
                continue

        # If all chunks are full, create a new one and retry
        if block_off is None:
            new_chunk = self._create_chunk()
            alloc_chunk_idx = self._hdr.get_chunk_count() - 1
            block_off = new_chunk.allocator.allocate(
                nbytes,
                new_chunk.chunk_hdr.get(),
                new_chunk.chunk_hdr,
            )

        chunk = self._chunks[alloc_chunk_idx]

        # Copy data into the block (after the block header)
        local_payload_off = block_off + BLOCK_HEADER_SIZE
        dest = np.ndarray(data.shape, dtype=data.dtype, buffer=chunk.data_buf, offset=local_payload_off)
        np.copyto(dest, data)

        # Encode virtual offset
        virtual_offset = self._encode_offset(alloc_chunk_idx, local_payload_off)

        # Update index
        self._index.insert(
            key,
            data_offset=virtual_offset,
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

        chunk_idx, local_offset = self._decode_offset(entry.data_offset)
        chunk = self._ensure_chunk(chunk_idx)

        dtype = np.dtype(entry.dtype_str.rstrip(b"\x00"))
        arr = np.ndarray(
            entry.shape,
            dtype=dtype,
            buffer=chunk.data_buf,
            offset=local_offset,
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

        if nbytes > self._chunk_data_size:
            raise OutOfMemoryError(
                f"Value too large ({nbytes} bytes) for chunk_data_size "
                f"({self._chunk_data_size} bytes). Use a larger chunk_size_mb."
            )

        # Free old block if key exists
        old_entry = self._index.find(key)
        if old_entry is not None:
            chunk_idx, local_payload_off = self._decode_offset(old_entry.data_offset)
            chunk = self._ensure_chunk(chunk_idx)
            old_block_off = local_payload_off - BLOCK_HEADER_SIZE
            chunk.allocator.deallocate(
                old_block_off,
                chunk.chunk_hdr.get(),
                chunk.chunk_hdr,
            )
            self._update_entry_count(-1)

        # Try allocating from each existing chunk
        block_off = None
        alloc_chunk_idx = None
        for ci in range(self._hdr.get_chunk_count()):
            chunk = self._ensure_chunk(ci)
            try:
                block_off = chunk.allocator.allocate(
                    nbytes,
                    chunk.chunk_hdr.get(),
                    chunk.chunk_hdr,
                )
                alloc_chunk_idx = ci
                break
            except OutOfMemoryError:
                continue

        if block_off is None:
            new_chunk = self._create_chunk()
            alloc_chunk_idx = self._hdr.get_chunk_count() - 1
            block_off = new_chunk.allocator.allocate(
                nbytes,
                new_chunk.chunk_hdr.get(),
                new_chunk.chunk_hdr,
            )

        chunk = self._chunks[alloc_chunk_idx]
        local_payload_off = block_off + BLOCK_HEADER_SIZE

        # Copy bytes in
        chunk.data_buf[local_payload_off : local_payload_off + nbytes] = data

        virtual_offset = self._encode_offset(alloc_chunk_idx, local_payload_off)

        # Index with empty dtype (signals raw bytes)
        self._index.insert(
            key,
            data_offset=virtual_offset,
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
            chunk_idx, local_offset = self._decode_offset(entry.data_offset)
            chunk = self._ensure_chunk(chunk_idx)
            return chunk.data_buf[local_offset : local_offset + entry.data_size]
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

        chunk_idx, local_payload_off = self._decode_offset(entry.data_offset)
        chunk = self._ensure_chunk(chunk_idx)
        block_off = local_payload_off - BLOCK_HEADER_SIZE
        chunk.allocator.deallocate(
            block_off,
            chunk.chunk_hdr.get(),
            chunk.chunk_hdr,
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
        """Return usage statistics aggregated across all chunks."""
        self._acquire_read()
        try:
            total_bytes = 0
            used_bytes = 0
            free_bytes = 0
            used_blocks = 0
            free_blocks = 0
            largest_free = 0

            chunk_count = self._hdr.get_chunk_count()
            for ci in range(chunk_count):
                chunk = self._ensure_chunk(ci)
                stats = chunk.allocator.stats(chunk.chunk_hdr.get())
                total_bytes += stats["total_bytes"]
                used_bytes += stats["used_bytes"]
                free_bytes += stats["free_bytes"]
                used_blocks += stats["used_blocks"]
                free_blocks += stats["free_blocks"]
                if stats["largest_free_payload"] > largest_free:
                    largest_free = stats["largest_free_payload"]

            return {
                "name": self._name,
                "max_entries": self._hdr.get_max_entries(),
                "entry_count": self._hdr.get_entry_count(),
                "chunk_count": chunk_count,
                "chunk_data_size": self._chunk_data_size,
                "total_bytes": total_bytes,
                "used_bytes": used_bytes,
                "free_bytes": free_bytes,
                "used_blocks": used_blocks,
                "free_blocks": free_blocks,
                "largest_free_payload": largest_free,
            }
        finally:
            self._release_read()

    def _release_buffers(self) -> None:
        """Release all memoryview references so the underlying mmap can be closed."""
        for chunk in self._chunks:
            if chunk is not None:
                chunk.data_buf = None
                chunk.buf = None
                chunk.allocator = None
                chunk.chunk_hdr = None
        self._chunks = []
        self._index_buf = None
        self._ctrl_buf = None
        self._index = None
        self._hdr = None

    def close(self) -> None:
        """Detach from shared memory (does not destroy it)."""
        chunks = list(self._chunks)
        self._release_buffers()
        for chunk in chunks:
            if chunk is not None:
                chunk.shm.close()
        self._ctrl_shm.close()

    def destroy(self) -> None:
        """Unlink (delete) the shared memory segments. Only the creator should call this."""
        chunks = list(self._chunks)
        self._release_buffers()
        for chunk in chunks:
            if chunk is not None:
                chunk.shm.close()
                chunk.shm.unlink()
        self._ctrl_shm.close()
        self._ctrl_shm.unlink()
