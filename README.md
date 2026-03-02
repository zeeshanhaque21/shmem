# shmem

Zero-copy shared memory key-value store for sharing numpy arrays and raw bytes between Python processes.

`put()` copies data into a single large `SharedMemory` block (one `memcpy`). `get()` returns a read-only `np.ndarray` view directly on that memory — no copy, no deserialization, ~0.5 microseconds regardless of data size.

## Installation

```
uv sync
```

## Quick start

```python
from shmem import SharedStore

# Parent process — create the store
store = SharedStore.create("pipeline", size_mb=256, max_entries=1024)

# Write a numpy array (one memcpy)
import numpy as np
frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
store.put("frame_0", frame)

# Read — zero-copy view (~0.5 us)
arr = store.get("frame_0")            # read-only ndarray view
arr = store.get_mut("frame_0")        # writable view (caller coordinates access)

# Raw bytes
store.put_bytes("metadata", b'{"ts": 1234}')
mv = store.get_bytes("metadata")      # zero-copy memoryview

# Lifecycle
store.delete("frame_0")
store.keys()                           # list all keys
store.info()                           # usage stats
store.destroy()                        # unlink shared memory
```

### Child processes

Child processes get full read/write access:

```python
import multiprocessing
from shmem import SharedStore

def worker(locks):
    store = SharedStore.connect("pipeline", locks, max_entries=1024)
    arr = store.get("frame_0")         # zero-copy read
    store.put("result", processed)     # write new keys
    store.delete("frame_0")            # delete keys
    store.close()

p = multiprocessing.Process(target=worker, args=(store.locks(),))
p.start()
```

## Memory layout

```
┌──────────────┬────────────────────┬──────────────────────────────┐
│ STORE HEADER │   INDEX (hash tbl) │         DATA REGION          │
│  (64 bytes)  │  (N x 256 bytes)   │   (free-list managed blocks) │
└──────────────┴────────────────────┴──────────────────────────────┘
```

- **Header** (64B): magic, version, capacity, offsets, entry count, free-list head, readers count
- **Index**: open-addressing hash table with linear probing. Each 256B entry holds key (128B), offset, size, dtype/shape metadata, state flag
- **Data region**: variable-sized blocks with a free-list allocator. Each block has a 16B header (size, prev_size, is_free) followed by user data. Adjacent free blocks are coalesced on `delete()`

## Locking

Readers-writers lock using two `multiprocessing.Lock` objects:

- Multiple concurrent readers allowed (`get`, `get_mut`, `get_bytes`, `keys`, `info`)
- Writers get exclusive access (`put`, `put_bytes`, `delete`)
- 5-second timeout on acquire to handle crashed processes

## Tests

```
uv run pytest                     # all 55 tests
uv run pytest tests/test_store.py # integration tests only
```

## Benchmarks

```
uv run python benchmark.py        # raw shm latency comparison
uv run python bench_store.py      # SharedStore put/get latency & throughput
```
