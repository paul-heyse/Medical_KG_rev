"""Abstract storage interfaces.

This module defines abstract base classes and interfaces for storage
backends, providing a common contract for object storage, ledger tracking,
and caching operations.

The module provides:
- ObjectStore interface for blob storage
- LedgerStore interface for state tracking
- CacheBackend interface for caching operations
- Common metadata and error types

Thread Safety:
    Thread-safe: Abstract interfaces with no shared state.

Performance:
    Interface definitions only - no performance characteristics.

Example:
    >>> class MyObjectStore(ObjectStore):
    ...     async def put(self, key: str, data: bytes) -> None:
    ...         # Implementation
    ...         pass
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class ObjectMetadata:
    """Metadata returned by object store operations.

    Attributes:
        content_type: MIME type of the stored object.
        size: Size of the object in bytes.
    """

    content_type: str | None
    size: int


# ==============================================================================
# INTERFACES
# ==============================================================================

class StorageError(RuntimeError):
    """Base exception for storage backends."""


class ObjectStore(ABC):
    """Interface for object storage backends.

    Abstract base class defining the contract for object storage operations
    including put, get, and delete operations with metadata support.
    """

    @abstractmethod
    async def put(self, key: str, data: bytes, *, metadata: dict[str, str] | None = None) -> None:
        """Store data with the given key.

        Args:
            key: Unique identifier for the data.
            data: Binary data to store.
            metadata: Optional metadata for the object.

        Raises:
            StorageError: If the operation fails.
        """
        raise NotImplementedError

    @abstractmethod
    async def get(self, key: str) -> bytes:
        """Retrieve data by key.

        Args:
            key: Unique identifier for the data.

        Returns:
            Binary data stored under the key.

        Raises:
            StorageError: If the operation fails.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete data by key.

        Args:
            key: Unique identifier for the data.

        Raises:
            StorageError: If the operation fails.
        """
        raise NotImplementedError


class LedgerStore(ABC):
    """Interface used to track ingestion state.

    Abstract base class for storing and retrieving job state information
    during ingestion and processing workflows.
    """

    @abstractmethod
    async def record_state(self, job_id: str, state: dict[str, Any]) -> None:
        """Record state for a job.

        Args:
            job_id: Unique identifier for the job.
            state: State data to store.

        Raises:
            StorageError: If the operation fails.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_state(self, job_id: str) -> dict[str, Any] | None:
        """Retrieve state for a job.

        Args:
            job_id: Unique identifier for the job.

        Returns:
            State data if found, None otherwise.

        Raises:
            StorageError: If the operation fails.
        """
        raise NotImplementedError


class CacheBackend(ABC):
    """Simple cache interface used by services.

    Abstract base class for caching operations with TTL support.
    """

    @abstractmethod
    async def get(self, key: str) -> bytes | None:
        """Retrieve cached value.

        Args:
            key: Cache key.

        Returns:
            Cached value if found, None otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        """Store value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time to live in seconds.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete cached value.

        Args:
            key: Cache key.
        """
        raise NotImplementedError


# ==============================================================================
# IMPLEMENTATIONS
# ==============================================================================


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = ["CacheBackend", "LedgerStore", "ObjectMetadata", "ObjectStore", "StorageError"]
