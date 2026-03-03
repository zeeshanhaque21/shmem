"""Tests for ctypes struct sizes and layout constants."""

import ctypes
from shmem.layout import (
    StoreHeader,
    ChunkHeader,
    IndexEntry,
    BlockHeader,
    FreeBlockLinks,
    HEADER_SIZE,
    CHUNK_HEADER_SIZE,
    INDEX_ENTRY_SIZE,
    BLOCK_HEADER_SIZE,
    FREE_BLOCK_LINKS_SIZE,
    VERSION,
    compute_control_size,
)


def test_store_header_size():
    assert ctypes.sizeof(StoreHeader) == HEADER_SIZE == 64


def test_chunk_header_size():
    assert ctypes.sizeof(ChunkHeader) == CHUNK_HEADER_SIZE == 32


def test_index_entry_size():
    assert ctypes.sizeof(IndexEntry) == INDEX_ENTRY_SIZE == 256


def test_block_header_size():
    assert ctypes.sizeof(BlockHeader) == BLOCK_HEADER_SIZE == 16


def test_free_block_links_size():
    assert ctypes.sizeof(FreeBlockLinks) == FREE_BLOCK_LINKS_SIZE == 16


def test_version():
    assert VERSION == 2


def test_compute_control_size():
    size = compute_control_size(1024)
    assert size == 64 + 1024 * 256


def test_store_header_fields():
    hdr = StoreHeader()
    hdr.magic = 0xDEADBEEF
    hdr.version = 2
    hdr.max_entries = 512
    hdr.chunk_data_size = 1024 * 1024
    hdr.chunk_count = 3
    assert hdr.magic == 0xDEADBEEF
    assert hdr.chunk_data_size == 1024 * 1024
    assert hdr.chunk_count == 3


def test_chunk_header_fields():
    hdr = ChunkHeader()
    hdr.free_list_head = -1
    hdr.data_size = 65536
    assert hdr.free_list_head == -1
    assert hdr.data_size == 65536


def test_index_entry_key_capacity():
    entry = IndexEntry()
    # 127 bytes of key data + null terminator
    entry.key = b"a" * 127
    assert entry.key == b"a" * 127


def test_block_header_fields():
    hdr = BlockHeader()
    hdr.size = 4096
    hdr.prev_size = 1024
    hdr.is_free = 1
    assert hdr.size == 4096
    assert hdr.prev_size == 1024
    assert hdr.is_free == 1
