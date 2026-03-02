"""Block allocator with first-fit free-list and forward/backward coalescing.

Manages the data region of the shared memory buffer.  Each block has:
  [BlockHeader 16B] [FreeBlockLinks 16B (only if free)] [user payload ...]

The allocator works on a raw buffer (memoryview or bytearray) so it can be
tested in isolation without shared memory.
"""

import ctypes

from .layout import (
    BLOCK_HEADER_SIZE,
    FREE_BLOCK_LINKS_SIZE,
    BlockHeader,
    FreeBlockLinks,
)
from .errors import OutOfMemoryError

# Minimum block size: header + free-list links (so a free block can hold its pointers).
MIN_BLOCK_SIZE = BLOCK_HEADER_SIZE + FREE_BLOCK_LINKS_SIZE  # 32 bytes


class BlockAllocator:
    """First-fit free-list allocator over a contiguous byte buffer.

    Parameters
    ----------
    buf : writable buffer (memoryview, bytearray, or mmap)
        The data region to manage.  Must be at least MIN_BLOCK_SIZE bytes.
    """

    def __init__(self, buf):
        self._buf = buf
        self._size = len(buf)

    # -- helpers to read/write structs at an offset ---------------------------

    def _read_block_header(self, offset: int) -> BlockHeader:
        hdr = BlockHeader()
        ctypes.memmove(ctypes.addressof(hdr), self._buf_ptr(offset), BLOCK_HEADER_SIZE)
        return hdr

    def _write_block_header(self, offset: int, hdr: BlockHeader) -> None:
        ctypes.memmove(self._buf_ptr(offset), ctypes.addressof(hdr), BLOCK_HEADER_SIZE)

    def _read_free_links(self, offset: int) -> FreeBlockLinks:
        links = FreeBlockLinks()
        link_off = offset + BLOCK_HEADER_SIZE
        ctypes.memmove(ctypes.addressof(links), self._buf_ptr(link_off), FREE_BLOCK_LINKS_SIZE)
        return links

    def _write_free_links(self, offset: int, links: FreeBlockLinks) -> None:
        link_off = offset + BLOCK_HEADER_SIZE
        ctypes.memmove(self._buf_ptr(link_off), ctypes.addressof(links), FREE_BLOCK_LINKS_SIZE)

    def _buf_ptr(self, offset: int) -> int:
        """Return a ctypes-compatible pointer (integer address) into the buffer."""
        if isinstance(self._buf, memoryview):
            return ctypes.addressof(ctypes.c_char.from_buffer(self._buf, offset))
        elif isinstance(self._buf, bytearray):
            return ctypes.addressof((ctypes.c_char * len(self._buf)).from_buffer(self._buf)) + offset
        else:
            return ctypes.addressof(ctypes.c_char.from_buffer(self._buf, offset))

    # -- public API -----------------------------------------------------------

    def init(self, free_list_head_setter) -> None:
        """Initialize the data region as a single large free block.

        Parameters
        ----------
        free_list_head_setter : callable(int)
            Called with the offset of the free-list head (always 0 after init).
        """
        hdr = BlockHeader()
        hdr.size = self._size
        hdr.prev_size = 0
        hdr.is_free = 1
        self._write_block_header(0, hdr)

        links = FreeBlockLinks()
        links.next_free = -1
        links.prev_free = -1
        self._write_free_links(0, links)

        free_list_head_setter(0)

    def allocate(self, payload_size: int, free_list_head: int, free_list_head_setter) -> int:
        """Allocate a block for *payload_size* bytes.

        Returns the offset of the **payload** (i.e. offset + BLOCK_HEADER_SIZE
        is where user data starts — but we return the block offset; the caller
        adds BLOCK_HEADER_SIZE).

        Actually, returns the offset of the BlockHeader.  The caller should
        write user data at offset + BLOCK_HEADER_SIZE.

        Raises OutOfMemoryError if no suitable block is found.
        """
        needed = payload_size + BLOCK_HEADER_SIZE
        if needed < MIN_BLOCK_SIZE:
            needed = MIN_BLOCK_SIZE

        # Align to 8 bytes
        needed = (needed + 7) & ~7

        # First-fit search through the free list
        cur = free_list_head
        while cur >= 0:
            hdr = self._read_block_header(cur)
            if hdr.size >= needed:
                # Found a fit — remove from free list and possibly split
                self._remove_from_free_list(cur, free_list_head, free_list_head_setter)

                remainder = hdr.size - needed
                if remainder >= MIN_BLOCK_SIZE:
                    # Split: shrink this block, create a new free block for the remainder
                    hdr.size = needed
                    hdr.is_free = 0
                    self._write_block_header(cur, hdr)

                    # Create remainder block
                    rem_off = cur + needed
                    rem_hdr = BlockHeader()
                    rem_hdr.size = remainder
                    rem_hdr.prev_size = needed
                    rem_hdr.is_free = 1
                    self._write_block_header(rem_off, rem_hdr)

                    # Update the block AFTER the remainder so its prev_size is correct
                    next_off = rem_off + remainder
                    if next_off < self._size:
                        next_hdr = self._read_block_header(next_off)
                        next_hdr.prev_size = remainder
                        self._write_block_header(next_off, next_hdr)

                    # Add remainder to free list
                    self._add_to_free_list(rem_off, free_list_head_setter)
                else:
                    # Use the entire block (no split)
                    hdr.is_free = 0
                    self._write_block_header(cur, hdr)

                return cur

            # Move to next free block
            links = self._read_free_links(cur)
            cur = links.next_free

        raise OutOfMemoryError(
            f"Cannot allocate {payload_size} bytes (needed {needed} with header). "
            f"Data region is {self._size} bytes."
        )

    def deallocate(self, block_offset: int, free_list_head: int, free_list_head_setter) -> None:
        """Free the block at *block_offset* and coalesce with adjacent free blocks."""
        hdr = self._read_block_header(block_offset)
        hdr.is_free = 1
        self._write_block_header(block_offset, hdr)

        # Coalesce with NEXT block
        next_off = block_offset + hdr.size
        if next_off < self._size:
            next_hdr = self._read_block_header(next_off)
            if next_hdr.is_free:
                self._remove_from_free_list(next_off, free_list_head, free_list_head_setter)
                hdr.size += next_hdr.size
                self._write_block_header(block_offset, hdr)
                # Update the block after the merged block
                after = block_offset + hdr.size
                if after < self._size:
                    after_hdr = self._read_block_header(after)
                    after_hdr.prev_size = hdr.size
                    self._write_block_header(after, after_hdr)

        # Coalesce with PREV block
        if hdr.prev_size > 0:
            prev_off = block_offset - hdr.prev_size
            prev_hdr = self._read_block_header(prev_off)
            if prev_hdr.is_free:
                self._remove_from_free_list(prev_off, free_list_head, free_list_head_setter)
                prev_hdr.size += hdr.size
                self._write_block_header(prev_off, prev_hdr)
                # Update the block after the merged block
                after = prev_off + prev_hdr.size
                if after < self._size:
                    after_hdr = self._read_block_header(after)
                    after_hdr.prev_size = prev_hdr.size
                    self._write_block_header(after, after_hdr)
                # The merged block starts at prev_off now
                block_offset = prev_off
                hdr = prev_hdr

        # Add the (possibly merged) block to the free list
        self._add_to_free_list(block_offset, free_list_head_setter)

    def stats(self, free_list_head: int) -> dict:
        """Walk all blocks and return usage statistics."""
        total = 0
        used = 0
        free = 0
        used_blocks = 0
        free_blocks = 0
        largest_free = 0

        offset = 0
        while offset < self._size:
            hdr = self._read_block_header(offset)
            if hdr.size == 0:
                break  # safety: corrupt or uninitialized
            total += hdr.size
            if hdr.is_free:
                free += hdr.size
                free_blocks += 1
                payload = hdr.size - BLOCK_HEADER_SIZE
                if payload > largest_free:
                    largest_free = payload
            else:
                used += hdr.size
                used_blocks += 1
            offset += hdr.size

        return {
            "total_bytes": self._size,
            "used_bytes": used,
            "free_bytes": free,
            "used_blocks": used_blocks,
            "free_blocks": free_blocks,
            "largest_free_payload": largest_free,
        }

    # -- free-list management (internal) --------------------------------------

    def _get_free_list_head(self, free_list_head: int) -> int:
        return free_list_head

    def _add_to_free_list(self, offset: int, free_list_head_setter) -> None:
        """Prepend a block to the head of the free list."""
        # Read the current head from the setter's closure (we pass it via callback)
        # We need to get the current head — the setter stores it externally.
        # Convention: free_list_head_setter also has a .get() or we pass head separately.
        # Simpler: we use a two-way callback. Let's just use a mutable wrapper.
        # Actually, let's read the current head from the header each time.
        # The caller manages the head externally. We'll use a simpler approach:
        # the setter is called, and we also need a getter. Let's refactor.

        # We'll make add/remove take current head as arg and return new head via setter.
        # But to keep the interface, we use the pattern where free_list_head_setter
        # is an object with .get() and .set() — or we just use a closure pattern.

        # For simplicity: the setter is a callable that also acts as getter when called
        # with no args. Let's use a different approach: we read links from existing head.

        # Get current head by reading from the setter (it stores the value)
        current_head = free_list_head_setter(None)  # None = get

        links = FreeBlockLinks()
        links.next_free = current_head
        links.prev_free = -1
        self._write_free_links(offset, links)

        if current_head >= 0:
            head_links = self._read_free_links(current_head)
            head_links.prev_free = offset
            self._write_free_links(current_head, head_links)

        free_list_head_setter(offset)

    def _remove_from_free_list(
        self, offset: int, free_list_head: int, free_list_head_setter
    ) -> None:
        """Remove a block from the free list."""
        links = self._read_free_links(offset)

        if links.prev_free >= 0:
            prev_links = self._read_free_links(links.prev_free)
            prev_links.next_free = links.next_free
            self._write_free_links(links.prev_free, prev_links)
        else:
            # This was the head
            free_list_head_setter(links.next_free)

        if links.next_free >= 0:
            next_links = self._read_free_links(links.next_free)
            next_links.prev_free = links.prev_free
            self._write_free_links(links.next_free, next_links)


class FreeListHeadAccessor:
    """Mutable wrapper so the allocator can get/set the free-list head.

    Usage:
        accessor = FreeListHeadAccessor(initial_value)
        allocator.allocate(size, accessor.get(), accessor)
        # accessor is callable: accessor(value) sets, accessor(None) gets.
    """

    def __init__(self, initial: int = -1):
        self._value = initial

    def __call__(self, value=None):
        if value is None:
            return self._value
        self._value = value

    def get(self) -> int:
        return self._value

    def set(self, value: int) -> None:
        self._value = value
