"""Caching utilities for pipeline state serialisation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import MutableMapping
from typing import TYPE_CHECKING
import time

from attrs import define, field

from Medical_KG_rev.observability.metrics import HttpClient
from Medical_KG_rev.orchestration.stages.contracts import PipelineStateSnapshot



@define(slots=True)
class PipelineStateCache:
    """Maintain an LRU cache of recent pipeline state snapshots."""

    max_entries: int = 64
    ttl_seconds: float | None = None
    _entries: MutableMapping[str, PipelineStateSnapshot] = field(factory=OrderedDict, init=False)
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
