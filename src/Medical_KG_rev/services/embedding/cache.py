"""Embedding cache utilities backed by Redis or in-memory storage."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Protocol

from Medical_KG_rev.embeddings.ports import EmbeddingRecord

try:  # pragma: no cover - optional dependency during tests
    import redis
except Exception:  # pragma: no cover - handled by NullEmbeddingCache fallback
    redis = None  # type: ignore[assignment]


def _serialize(record: EmbeddingRecord) -> str:
    payload = asdict(record)
    payload["created_at"] = record.created_at.isoformat()
    return json.dumps(payload)


def _deserialize(payload: str) -> EmbeddingRecord:
    data = json.loads(payload)
    created_at = (
        datetime.fromisoformat(data.get("created_at"))
        if data.get("created_at")
        else datetime.utcnow()
    )
    return EmbeddingRecord(
        id=data["id"],
        tenant_id=data["tenant_id"],
        namespace=data["namespace"],
        model_id=data["model_id"],
        model_version=data["model_version"],
        kind=data["kind"],
        dim=data.get("dim"),
        vectors=data.get("vectors"),
        terms=data.get("terms"),
        neural_fields=data.get("neural_fields"),
        normalized=data.get("normalized", False),
        metadata=data.get("metadata", {}),
        created_at=created_at,
        correlation_id=data.get("correlation_id"),
    )


class EmbeddingCache(Protocol):
    """Simple protocol describing cache operations."""

    def get(self, namespace: str, embedding_id: str) -> EmbeddingRecord | None: ...

    def set(self, record: EmbeddingRecord, *, ttl: int | None = None) -> None: ...

    def invalidate_namespace(self, namespace: str) -> None: ...


class NullEmbeddingCache:
    """No-op cache implementation used when Redis is unavailable."""

    def get(self, namespace: str, embedding_id: str) -> EmbeddingRecord | None:
        return None

    def set(self, record: EmbeddingRecord, *, ttl: int | None = None) -> None:
        return None

    def invalidate_namespace(self, namespace: str) -> None:
        return None


class InMemoryEmbeddingCache:
    """In-process cache useful for tests and local development."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], EmbeddingRecord] = {}

    def get(self, namespace: str, embedding_id: str) -> EmbeddingRecord | None:
        return self._records.get((namespace, embedding_id))

    def set(self, record: EmbeddingRecord, *, ttl: int | None = None) -> None:
        self._records[(record.namespace, record.id)] = record

    def invalidate_namespace(self, namespace: str) -> None:
        keys_to_delete = [key for key in self._records if key[0] == namespace]
        for key in keys_to_delete:
            self._records.pop(key, None)


class RedisEmbeddingCache:
    """Redis-backed cache compatible with the embedding worker."""

    def __init__(self, *, url: str = "redis://localhost:6379/0", prefix: str = "embedding") -> None:
        if redis is None:  # pragma: no cover - exercised only when redis missing
            raise RuntimeError("redis dependency is not installed")
        self._client = redis.Redis.from_url(url)  # type: ignore[union-attr]
        self._prefix = prefix

    def _key(self, namespace: str, embedding_id: str) -> str:
        return f"{self._prefix}:{namespace}:{embedding_id}"

    def get(self, namespace: str, embedding_id: str) -> EmbeddingRecord | None:
        payload = self._client.get(self._key(namespace, embedding_id))
        if not payload:
            return None
        return _deserialize(payload.decode("utf-8"))

    def set(self, record: EmbeddingRecord, *, ttl: int | None = None) -> None:
        payload = _serialize(record)
        key = self._key(record.namespace, record.id)
        if ttl:
            self._client.setex(key, ttl, payload)
        else:
            self._client.set(key, payload)

    def invalidate_namespace(self, namespace: str) -> None:
        pattern = f"{self._prefix}:{namespace}:*"
        keys = list(self._client.scan_iter(pattern))
        if keys:
            self._client.delete(*keys)


__all__ = [
    "EmbeddingCache",
    "InMemoryEmbeddingCache",
    "NullEmbeddingCache",
    "RedisEmbeddingCache",
]
