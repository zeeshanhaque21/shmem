"""Custom exceptions for the shared memory store."""


class StoreFullError(Exception):
    """Raised when the hash index has no free slots."""


class OutOfMemoryError(Exception):
    """Raised when the data region cannot satisfy an allocation."""


class StoreCorruptedError(Exception):
    """Raised when header magic/version check fails."""
