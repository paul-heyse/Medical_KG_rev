"""Abstract storage interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class StorageError(RuntimeError):
    """Base exception for storage backends."""


@dataclass
class ObjectMetadata:
    """Metadata returned by object store operations."""

    content_type: str | None
    size: int


class ObjectStore(ABC):
    """Interface for object storage backends."""

    @abstractmethod
    async def put(self, key: str, data: bytes, *, metadata: dict[str, str] | None = None) -> None:
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
    async def record_state(self, job_id: str, state: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_state(self, job_id: str) -> dict[str, Any] | None:
        raise NotImplementedError


class CacheBackend(ABC):
    """Simple cache interface used by services."""

    @abstractmethod
    async def get(self, key: str) -> bytes | None:
        raise NotImplementedError

    @abstractmethod
    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: str) -> None:
        raise NotImplementedError
