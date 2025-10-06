"""Abstract storage interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


class StorageError(RuntimeError):
    """Base exception for storage backends."""


@dataclass
class ObjectMetadata:
    """Metadata returned by object store operations."""

    content_type: Optional[str]
    size: int


class ObjectStore(ABC):
    """Interface for object storage backends."""

    @abstractmethod
    async def put(self, key: str, data: bytes, *, metadata: Optional[Dict[str, str]] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get(self, key: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: str) -> None:
        raise NotImplementedError


class LedgerStore(ABC):
    """Interface used to track ingestion state."""

    @abstractmethod
    async def record_state(self, job_id: str, state: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_state(self, job_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class CacheBackend(ABC):
    """Simple cache interface used by services."""

    @abstractmethod
    async def get(self, key: str) -> Optional[bytes]:
        raise NotImplementedError

    @abstractmethod
    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: str) -> None:
        raise NotImplementedError
