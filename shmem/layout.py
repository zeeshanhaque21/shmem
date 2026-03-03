"""ctypes.Structure definitions mapped directly onto the shared memory buffer.

Memory layout (multi-chunk):
  Control Block: "{name}"
    [StoreHeader 64B] [IndexEntry × N, each 256B]

  Data Chunk i: "{name}_{i}"
    [ChunkHeader 32B] [Data region: BlockHeader+payload ...]
"""

import ctypes

# -- Constants ----------------------------------------------------------------

MAGIC = 0x534D4B56  # "SMKV"
VERSION = 2

HEADER_SIZE = 64
CHUNK_HEADER_SIZE = 32
INDEX_ENTRY_SIZE = 256
BLOCK_HEADER_SIZE = 16
FREE_BLOCK_LINKS_SIZE = 16

MAX_KEY_LEN = 128
MAX_DTYPE_LEN = 8
MAX_SHAPE_DIMS = 8  # up to 8-dimensional arrays

# Index entry state flags
ENTRY_EMPTY = 0
ENTRY_OCCUPIED = 1
ENTRY_TOMBSTONE = 2


# -- Structures ---------------------------------------------------------------


class StoreHeader(ctypes.Structure):
    """64-byte header at offset 0 of the control block."""

    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32),            # 4   magic number
        ("version", ctypes.c_uint32),           # 4   format version
        ("max_entries", ctypes.c_uint32),        # 4   hash table capacity
        ("entry_count", ctypes.c_uint32),        # 4   number of occupied entries
        ("index_offset", ctypes.c_uint64),       # 8   byte offset to index region
        ("chunk_data_size", ctypes.c_uint64),    # 8   usable bytes per data chunk (excl ChunkHeader)
        ("chunk_count", ctypes.c_uint32),        # 4   number of allocated data chunks
        ("_reserved", ctypes.c_uint32),          # 4   padding
        ("_unused", ctypes.c_int64),             # 8   reserved (was free_list_head)
        ("readers_count", ctypes.c_int32),       # 4   current number of readers
        ("_pad", ctypes.c_char * 12),            # 12  padding to 64 bytes
    ]


class ChunkHeader(ctypes.Structure):
    """32-byte header at offset 0 of each data chunk SharedMemory."""

    _pack_ = 1
    _fields_ = [
        ("free_list_head", ctypes.c_int64),    # 8   first free block in this chunk (-1 = none)
        ("data_size", ctypes.c_uint64),        # 8   usable bytes after ChunkHeader
        ("_pad", ctypes.c_char * 16),          # 16  reserved
    ]


class IndexEntry(ctypes.Structure):
    """256-byte hash table entry.

    Fields:
      key:        UTF-8 key (null-terminated, max 128 bytes including null)
      data_offset: byte offset into the data region where payload starts
      data_size:  size of the user payload in bytes (excludes block header)
      dtype_str:  numpy dtype string, e.g. "uint8", "float32" (empty for raw bytes)
      ndim:       number of dimensions (0 for raw bytes)
      shape:      array shape dimensions
      state:      ENTRY_EMPTY / ENTRY_OCCUPIED / ENTRY_TOMBSTONE
    """

    _pack_ = 1
    _fields_ = [
        ("key", ctypes.c_char * MAX_KEY_LEN),          # 128
        ("data_offset", ctypes.c_uint64),               # 8
        ("data_size", ctypes.c_uint64),                  # 8
        ("dtype_str", ctypes.c_char * MAX_DTYPE_LEN),   # 8
        ("ndim", ctypes.c_uint32),                       # 4
        ("shape", ctypes.c_uint64 * MAX_SHAPE_DIMS),    # 64
        ("state", ctypes.c_uint8),                       # 1
        ("_pad", ctypes.c_char * 35),                    # 35  → total 256
    ]


class BlockHeader(ctypes.Structure):
    """16-byte header prepended to every block in the data region.

    Fields:
      size:      total block size INCLUDING this header
      prev_size: total size of the previous block (0 if first block)
      is_free:   1 if free, 0 if allocated
    """

    _pack_ = 1
    _fields_ = [
        ("size", ctypes.c_uint64),       # 8
        ("prev_size", ctypes.c_uint32),  # 4
        ("is_free", ctypes.c_uint8),     # 1
        ("_pad", ctypes.c_char * 3),     # 3  → total 16
    ]


class FreeBlockLinks(ctypes.Structure):
    """16-byte free-list pointers stored immediately after BlockHeader in free blocks.

    Only valid when BlockHeader.is_free == 1.  Offsets are relative to data region start.
    -1 means null (no next/prev).
    """

    _pack_ = 1
    _fields_ = [
        ("next_free", ctypes.c_int64),  # 8  offset of next free block
        ("prev_free", ctypes.c_int64),  # 8  offset of prev free block
    ]


# -- Size assertions ----------------------------------------------------------

assert ctypes.sizeof(StoreHeader) == HEADER_SIZE, (
    f"StoreHeader is {ctypes.sizeof(StoreHeader)}B, expected {HEADER_SIZE}B"
)
assert ctypes.sizeof(ChunkHeader) == CHUNK_HEADER_SIZE, (
    f"ChunkHeader is {ctypes.sizeof(ChunkHeader)}B, expected {CHUNK_HEADER_SIZE}B"
)
assert ctypes.sizeof(IndexEntry) == INDEX_ENTRY_SIZE, (
    f"IndexEntry is {ctypes.sizeof(IndexEntry)}B, expected {INDEX_ENTRY_SIZE}B"
)
assert ctypes.sizeof(BlockHeader) == BLOCK_HEADER_SIZE, (
    f"BlockHeader is {ctypes.sizeof(BlockHeader)}B, expected {BLOCK_HEADER_SIZE}B"
)
assert ctypes.sizeof(FreeBlockLinks) == FREE_BLOCK_LINKS_SIZE, (
    f"FreeBlockLinks is {ctypes.sizeof(FreeBlockLinks)}B, expected {FREE_BLOCK_LINKS_SIZE}B"
)


def compute_control_size(max_entries: int) -> int:
    """Return the total size of the control block (header + index, no data)."""
    return HEADER_SIZE + max_entries * INDEX_ENTRY_SIZE
