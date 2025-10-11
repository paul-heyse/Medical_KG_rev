"""TTL cache for reranking scores."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass, field
from json import dumps, loads
from time import monotonic
from typing import Any, Protocol

from ..models import CacheMetrics, RerankResult



@dataclass(slots=True)
class CacheEntry:
    value: RerankResult
    expires_at: float


class CacheBackend(Protocol):
    def get(self, key: str) -> RerankResult | None:
        ...

    def set(self, key: str, value: RerankResult, ttl: int) -> None:
        ...

    def invalidate(self, pattern: str) -> None:
        ...


class RedisCacheBackend:
    """Redis-based cache backend using simple JSON serialisation."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def get(self, key: str) -> RerankResult | None:
        raw = self.client.get(key)
        if raw is None:
            return None
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            payload = loads(raw)
        except Exception:
            return None
        return RerankResult(
            doc_id=payload["doc_id"],
            score=float(payload["score"]),
            rank=int(payload["rank"]),
            metadata=payload.get("metadata", {}),
        )

    def set(self, key: str, value: RerankResult, ttl: int) -> None:
        payload = dumps(
            {
                "doc_id": value.doc_id,
                "score": value.score,
                "rank": value.rank,
                "metadata": dict(value.metadata),
            }
        )
        self.client.setex(key, ttl, payload)

    def invalidate(self, pattern: str) -> None:
        matches = list(self.client.scan_iter(match=pattern))
        if matches:
            self.client.delete(*matches)


@dataclass(slots=True)
class RerankCacheManager:
    ttl_seconds: int = 3600
    _store: MutableMapping[str, CacheEntry] = field(default_factory=dict)
    _hits: int = 0
    _misses: int = 0
    backend: CacheBackend | None = None

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
        if self.backend is not None:
            cached = self.backend.get(key)
            if cached is not None:
                self._hits += 1
                return cached
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
            if self.backend is not None:
                self.backend.set(key, result, self.ttl_seconds)

    def invalidate(self, tenant_id: str, doc_ids: Iterable[str]) -> None:
        for key in list(self._store):
            if any(
                key.endswith(f":{doc_id}") and key.startswith(f"{tenant_id}:") for doc_id in doc_ids
            ):
                self._store.pop(key, None)
        if self.backend is not None:
            for doc_id in doc_ids:
                pattern = f"{tenant_id}:*:*:{doc_id}"
                self.backend.invalidate(pattern)

    def metrics(self) -> CacheMetrics:
        total = self._hits + self._misses
        hit_rate = float(self._hits) / total if total else 0.0
        return CacheMetrics(hits=self._hits, misses=self._misses, hit_rate=hit_rate)

    def reset_metrics(self) -> None:
        self._hits = 0
        self._misses = 0

    def warm(
        self,
        reranker_id: str,
        tenant_id: str,
        version: str,
        documents: Iterable[RerankResult],
    ) -> None:
        self.store(reranker_id, tenant_id, version, documents)
