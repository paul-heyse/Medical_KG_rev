"""TTL cache for reranking scores."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from ..models import CacheMetrics, RerankResult


@dataclass(slots=True)
class CacheEntry:
    value: RerankResult
    expires_at: float


@dataclass(slots=True)
class RerankCacheManager:
    ttl_seconds: int = 3600
    _store: MutableMapping[str, CacheEntry] = field(default_factory=dict)
    _hits: int = 0
    _misses: int = 0

    def _key(self, reranker_id: str, tenant_id: str, doc_id: str, version: str) -> str:
        return f"{tenant_id}:{reranker_id}:{version}:{doc_id}"

    def lookup(
        self,
        reranker_id: str,
        tenant_id: str,
        doc_id: str,
        version: str,
    ) -> RerankResult | None:
        key = self._key(reranker_id, tenant_id, doc_id, version)
        entry = self._store.get(key)
        if entry and entry.expires_at > monotonic():
            self._hits += 1
            return entry.value
        if entry:
            self._store.pop(key, None)
        self._misses += 1
        return None

    def store(
        self,
        reranker_id: str,
        tenant_id: str,
        version: str,
        results: Iterable[RerankResult],
    ) -> None:
        expires_at = monotonic() + float(self.ttl_seconds)
        for result in results:
            key = self._key(reranker_id, tenant_id, result.doc_id, version)
            self._store[key] = CacheEntry(value=result, expires_at=expires_at)

    def invalidate(self, tenant_id: str, doc_ids: Iterable[str]) -> None:
        for key in list(self._store):
            if any(key.endswith(f":{doc_id}") and key.startswith(f"{tenant_id}:") for doc_id in doc_ids):
                self._store.pop(key, None)

    def metrics(self) -> CacheMetrics:
        total = self._hits + self._misses
        hit_rate = float(self._hits) / total if total else 0.0
        return CacheMetrics(hits=self._hits, misses=self._misses, hit_rate=hit_rate)

    def reset_metrics(self) -> None:
        self._hits = 0
        self._misses = 0
