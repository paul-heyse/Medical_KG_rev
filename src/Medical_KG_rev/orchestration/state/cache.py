"""Caching utilities for pipeline state serialisation."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import MutableMapping
from typing import TYPE_CHECKING

from attrs import define, field

from Medical_KG_rev.observability.metrics import (
    PIPELINE_STATE_CACHE_HITS,
    PIPELINE_STATE_CACHE_MISSES,
    PIPELINE_STATE_CACHE_SIZE,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from Medical_KG_rev.orchestration.stages.contracts import PipelineStateSnapshot


@define(slots=True)
class PipelineStateCache:
    """Maintain an LRU cache of recent pipeline state snapshots."""

    max_entries: int = 64
    ttl_seconds: float | None = None
    _entries: MutableMapping[str, PipelineStateSnapshot] = field(
        factory=OrderedDict, init=False
    )
    _timestamps: MutableMapping[str, float] = field(factory=dict, init=False)

    def store(self, key: str, snapshot: PipelineStateSnapshot) -> None:
        """Store a snapshot for the provided key."""
        self._entries[key] = snapshot
        self._timestamps[key] = time.time()
        self._entries.move_to_end(key)
        self._prune()
        PIPELINE_STATE_CACHE_SIZE.set(len(self._entries))

    def get(self, key: str) -> PipelineStateSnapshot | None:
        """Return a cached snapshot if available and not expired."""
        snapshot = self._entries.get(key)
        if snapshot is None:
            PIPELINE_STATE_CACHE_MISSES.inc()
            return None

        if self.ttl_seconds is not None:
            timestamp = self._timestamps.get(key, 0.0)
            if time.time() - timestamp > self.ttl_seconds:
                self._entries.pop(key, None)
                self._timestamps.pop(key, None)
                PIPELINE_STATE_CACHE_MISSES.inc()
                PIPELINE_STATE_CACHE_SIZE.set(len(self._entries))
                return None

        self._entries.move_to_end(key)
        PIPELINE_STATE_CACHE_HITS.inc()
        return snapshot

    def clear(self) -> None:
        """Evict all cached entries."""
        self._entries.clear()
        self._timestamps.clear()
        PIPELINE_STATE_CACHE_SIZE.set(0)

    def _prune(self) -> None:
        while len(self._entries) > self.max_entries:
            key, _ = self._entries.popitem(last=False)
            self._timestamps.pop(key, None)


__all__ = ["PipelineStateCache"]

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
