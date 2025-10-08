"""Caching utilities for pipeline state serialisation."""

from __future__ import annotations

import time
from typing import Any

from attrs import define, field


@define(slots=True)
class _CacheEntry:
    payload: bytes
    created_at: float = field(factory=time.time)
    hits: int = 0

    def mark_hit(self) -> None:
        self.hits += 1


class PipelineStateCache:
    """Time-based cache used to store serialised pipeline state blobs."""

    def __init__(self, ttl_seconds: float = 30.0) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._ttl = ttl_seconds
        self._entries: dict[str, _CacheEntry] = {}

    def set(self, key: str, payload: bytes) -> None:
        self._entries[key] = _CacheEntry(payload)

    def get(self, key: str) -> bytes | None:
        entry = self._entries.get(key)
        if not entry:
            return None
        now = time.time()
        if now - entry.created_at > self._ttl:
            self._entries.pop(key, None)
            return None
        entry.mark_hit()
        return entry.payload

    def purge(self) -> None:
        now = time.time()
        expired = [key for key, entry in self._entries.items() if now - entry.created_at > self._ttl]
        for key in expired:
            self._entries.pop(key, None)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._entries)

    def snapshot(self) -> dict[str, Any]:
        """Return diagnostic information about the cache contents."""

        return {
            key: {"age": time.time() - entry.created_at, "hits": entry.hits}
            for key, entry in self._entries.items()
        }
