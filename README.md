# shmem

Zero-copy shared memory key-value store for sharing numpy arrays and raw bytes between Python processes.

`put()` copies data into shared memory (one `memcpy`). `get()` returns a read-only `np.ndarray` view directly on that memory — no copy, no deserialization, ~4 microseconds regardless of data size. The store grows automatically by allocating new data chunks on demand.

## Installation

```
uv sync
```

## Quick start

```python
from shmem import SharedStore

# Parent process — create the store with 64 MB chunks
store = SharedStore.create("pipeline", chunk_size_mb=64, max_entries=1024)

# Write a numpy array (one memcpy)
import numpy as np
frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
store.put("frame_0", frame)

# Read — zero-copy view (~4 us)
arr = store.get("frame_0")            # read-only ndarray view
arr = store.get_mut("frame_0")        # writable view (caller coordinates access)

# Raw bytes
store.put_bytes("metadata", b'{"ts": 1234}')
mv = store.get_bytes("metadata")      # zero-copy memoryview

# Lifecycle
store.delete("frame_0")
store.keys()                           # list all keys
store.info()                           # usage stats (aggregated across chunks)
store.destroy()                        # unlink all shared memory segments
```

### Child processes

Child processes get full read/write access. `connect()` reads configuration (max_entries, chunk size) from the header automatically:

```python
import multiprocessing
from shmem import SharedStore

def worker(locks):
    store = SharedStore.connect("pipeline", locks)
    arr = store.get("frame_0")         # zero-copy read
    store.put("result", processed)     # write new keys
    store.delete("frame_0")            # delete keys
    store.close()

p = multiprocessing.Process(target=worker, args=(store.locks(),))
p.start()
```

## Memory layout

The store is split into a **control block** (header + index) and separate **data chunks** allocated on demand:

```
Control Block: "{name}"
┌──────────────┬────────────────────┐
│ STORE HEADER │   INDEX (hash tbl) │
│  (64 bytes)  │  (N × 256 bytes)   │
└──────────────┴────────────────────┘

Data Chunk 0: "{name}_0"
┌──────────────┬──────────────────────────────┐
│ CHUNK HEADER │         DATA REGION          │
│  (32 bytes)  │   (free-list managed blocks) │
└──────────────┴──────────────────────────────┘

Data Chunk 1: "{name}_1"       ← created on demand when chunk 0 is full
┌──────────────┬──────────────────────────────┐
│ CHUNK HEADER │         DATA REGION          │
│  (32 bytes)  │   (free-list managed blocks) │
└──────────────┴──────────────────────────────┘
```

- **Header** (64B): magic, version, capacity, chunk_data_size, chunk_count, entry count, readers count
- **Index**: open-addressing hash table with linear probing. Each 256B entry holds key (128B), virtual offset, size, dtype/shape metadata, state flag
- **Chunk Header** (32B): free_list_head, data_size
- **Data region**: variable-sized blocks with a free-list allocator per chunk. Each block has a 16B header (size, prev_size, is_free) followed by user data. Adjacent free blocks are coalesced on `delete()`
- **Virtual offsets**: `chunk_index * chunk_data_size + local_offset`. The index stores these opaquely; decoding happens in the store layer

### Auto-growth

When `put()` cannot allocate from any existing chunk, it creates a new shared memory segment (`{name}_{i}`) and retries. Connected processes lazily open chunks as needed. A single value must fit within one chunk — storing a value larger than `chunk_data_size` raises `OutOfMemoryError`.

## Locking

Readers-writers lock using two `multiprocessing.Lock` objects:

- Multiple concurrent readers allowed (`get`, `get_mut`, `get_bytes`, `keys`, `info`)
- Writers get exclusive access (`put`, `put_bytes`, `delete`)
- 5-second timeout on acquire to handle crashed processes

## Tests

```
uv run pytest                     # all 64 tests
uv run pytest tests/test_store.py # integration tests only
```

## Benchmarks

```
uv run python bench_store.py      # SharedStore put/get latency, throughput & auto-growth
uv run python benchmark.py        # raw shm latency comparison
```
