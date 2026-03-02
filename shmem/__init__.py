"""shmem — Zero-copy shared memory key-value store for numpy arrays and raw bytes."""

from .store import SharedStore, StoreLocks
from .errors import StoreFullError, OutOfMemoryError, StoreCorruptedError

__all__ = [
    "SharedStore",
    "StoreLocks",
    "StoreFullError",
    "OutOfMemoryError",
    "StoreCorruptedError",
]
