"""Tests for ctypes struct sizes and layout constants."""

import ctypes
from shmem.layout import (
    StoreHeader,
    IndexEntry,
    BlockHeader,
    FreeBlockLinks,
    HEADER_SIZE,
    INDEX_ENTRY_SIZE,
    BLOCK_HEADER_SIZE,
    FREE_BLOCK_LINKS_SIZE,
    compute_offsets,
)


def test_store_header_size():
    assert ctypes.sizeof(StoreHeader) == HEADER_SIZE == 64


def test_index_entry_size():
    assert ctypes.sizeof(IndexEntry) == INDEX_ENTRY_SIZE == 256


def test_block_header_size():
    assert ctypes.sizeof(BlockHeader) == BLOCK_HEADER_SIZE == 16


def test_free_block_links_size():
    assert ctypes.sizeof(FreeBlockLinks) == FREE_BLOCK_LINKS_SIZE == 16


def test_compute_offsets():
    index_off, data_off = compute_offsets(1024)
    assert index_off == 64
    assert data_off == 64 + 1024 * 256


def test_store_header_fields():
    hdr = StoreHeader()
    hdr.magic = 0xDEADBEEF
    hdr.version = 1
    hdr.max_entries = 512
    hdr.free_list_head = -1
    assert hdr.magic == 0xDEADBEEF
    assert hdr.free_list_head == -1


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
