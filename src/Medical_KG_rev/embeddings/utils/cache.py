"""Simple in-memory cache for expensive embedding operations."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Generic, Hashable, TypeVar

import structlog

logger = structlog.get_logger(__name__)

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass(slots=True)
class CacheEntry(Generic[V]):
    value: V
    expires_at: datetime


class EmbeddingCache(Generic[K, V]):
    """LRU cache with TTL semantics used to cache expensive embeddings."""

    def __init__(self, maxsize: int = 512, ttl_seconds: int = 600) -> None:
        self._store: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = timedelta(seconds=ttl_seconds)

    def get(self, key: K) -> V | None:
        entry = self._store.get(key)
        if not entry:
            return None
        if entry.expires_at < datetime.utcnow():
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return entry.value

    def set(self, key: K, value: V) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = CacheEntry(value=value, expires_at=datetime.utcnow() + self._ttl)
        if len(self._store) > self._maxsize:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug("embedding.cache.evicted", key=evicted_key)

    def clear(self) -> None:
        self._store.clear()
