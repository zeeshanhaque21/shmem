"""Tests for the block allocator (uses plain bytearray, no shared memory)."""

import pytest
from shmem.allocator import BlockAllocator, FreeListHeadAccessor, MIN_BLOCK_SIZE
from shmem.layout import BLOCK_HEADER_SIZE, BlockHeader
from shmem.errors import OutOfMemoryError

REGION_SIZE = 4096  # 4 KB test region


def make_allocator(size=REGION_SIZE):
    buf = bytearray(size)
    alloc = BlockAllocator(buf)
    accessor = FreeListHeadAccessor(-1)
    alloc.init(accessor)
    return alloc, accessor


def test_init_single_free_block():
    alloc, acc = make_allocator()
    stats = alloc.stats(acc.get())
    assert stats["free_blocks"] == 1
    assert stats["used_blocks"] == 0
    assert stats["total_bytes"] == REGION_SIZE


def test_allocate_single():
    alloc, acc = make_allocator()
    off = alloc.allocate(64, acc.get(), acc)
    assert off == 0
    stats = alloc.stats(acc.get())
    assert stats["used_blocks"] == 1
    assert stats["free_blocks"] == 1  # remainder is free


def test_allocate_fills_region():
    alloc, acc = make_allocator(size=MIN_BLOCK_SIZE)
    off = alloc.allocate(1, acc.get(), acc)
    assert off == 0
    stats = alloc.stats(acc.get())
    assert stats["used_blocks"] == 1
    assert stats["free_blocks"] == 0


def test_allocate_oom():
    alloc, acc = make_allocator(size=MIN_BLOCK_SIZE)
    alloc.allocate(1, acc.get(), acc)
    with pytest.raises(OutOfMemoryError):
        alloc.allocate(1, acc.get(), acc)


def test_free_and_reuse():
    alloc, acc = make_allocator()
    off1 = alloc.allocate(64, acc.get(), acc)
    alloc.deallocate(off1, acc.get(), acc)
    stats = alloc.stats(acc.get())
    assert stats["used_blocks"] == 0
    assert stats["free_blocks"] == 1  # coalesced back to one

    off2 = alloc.allocate(64, acc.get(), acc)
    assert off2 == 0  # reuses the same space


def test_coalesce_forward():
    alloc, acc = make_allocator(size=1024)
    off1 = alloc.allocate(64, acc.get(), acc)
    off2 = alloc.allocate(64, acc.get(), acc)
    # Free second then first — first should coalesce forward into second
    alloc.deallocate(off2, acc.get(), acc)
    alloc.deallocate(off1, acc.get(), acc)
    stats = alloc.stats(acc.get())
    assert stats["free_blocks"] == 1
    assert stats["used_blocks"] == 0


def test_coalesce_backward():
    alloc, acc = make_allocator(size=1024)
    off1 = alloc.allocate(64, acc.get(), acc)
    off2 = alloc.allocate(64, acc.get(), acc)
    # Free first then second — second should coalesce backward into first
    alloc.deallocate(off1, acc.get(), acc)
    alloc.deallocate(off2, acc.get(), acc)
    stats = alloc.stats(acc.get())
    assert stats["free_blocks"] == 1
    assert stats["used_blocks"] == 0


def test_coalesce_both_directions():
    alloc, acc = make_allocator(size=2048)
    off1 = alloc.allocate(64, acc.get(), acc)
    off2 = alloc.allocate(64, acc.get(), acc)
    off3 = alloc.allocate(64, acc.get(), acc)

    alloc.deallocate(off1, acc.get(), acc)
    alloc.deallocate(off3, acc.get(), acc)
    # Free middle — should coalesce with both neighbors
    alloc.deallocate(off2, acc.get(), acc)
    stats = alloc.stats(acc.get())
    assert stats["free_blocks"] == 1
    assert stats["used_blocks"] == 0


def test_multiple_alloc_free():
    alloc, acc = make_allocator(size=4096)
    offsets = []
    for _ in range(10):
        off = alloc.allocate(32, acc.get(), acc)
        offsets.append(off)

    stats = alloc.stats(acc.get())
    assert stats["used_blocks"] == 10

    for off in offsets:
        alloc.deallocate(off, acc.get(), acc)

    stats = alloc.stats(acc.get())
    assert stats["used_blocks"] == 0
    assert stats["free_blocks"] == 1  # all coalesced


def test_fragmentation_recovery():
    """Allocate A B C, free B, free A, allocate D that needs A+B space."""
    alloc, acc = make_allocator(size=2048)
    a = alloc.allocate(100, acc.get(), acc)
    b = alloc.allocate(100, acc.get(), acc)
    _c = alloc.allocate(100, acc.get(), acc)

    alloc.deallocate(b, acc.get(), acc)
    alloc.deallocate(a, acc.get(), acc)
    # A and B are coalesced, so we should be able to allocate their combined space
    d = alloc.allocate(200, acc.get(), acc)
    assert d == a  # reuses the coalesced A+B region
