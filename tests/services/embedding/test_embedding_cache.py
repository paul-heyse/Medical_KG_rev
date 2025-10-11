import pytest

import Medical_KG_rev.services.embedding.cache as cache_module
from Medical_KG_rev.embeddings.ports import EmbeddingRecord
from Medical_KG_rev.services.embedding.cache import (
    InMemoryEmbeddingCache,
    NullEmbeddingCache,
    RedisEmbeddingCache,
)


def _record(chunk_id: str = "chunk-1") -> EmbeddingRecord:
    return EmbeddingRecord(
        id=chunk_id,
        tenant_id="tenant",
        namespace="single_vector.qwen3.4096.v1",
        model_id="Qwen/Qwen2.5-Embedding-8B-Instruct",
        model_version="v1",
        kind="single_vector",
        dim=2,
        vectors=[[0.1, 0.2]],
        metadata={"provider": "vllm"},
    )


def test_in_memory_cache_roundtrip() -> None:
    cache = InMemoryEmbeddingCache()
    record = _record()
    cache.set(record)
    cached = cache.get(record.namespace, record.id)
    assert cached is not None
    assert cached.vectors == record.vectors
    cache.invalidate_namespace(record.namespace)
    assert cache.get(record.namespace, record.id) is None


def test_null_cache_is_noop() -> None:
    cache = NullEmbeddingCache()
    record = _record()
    cache.set(record)
    assert cache.get(record.namespace, record.id) is None
    cache.invalidate_namespace(record.namespace)


@pytest.mark.skipif(cache_module.redis is None, reason="redis dependency not available")
def test_redis_cache_uses_client(monkeypatch: pytest.MonkeyPatch) -> None:
    storage: dict[str, str] = {}

    class StubRedisClient:
        def get(self, key: str):
            value = storage.get(key)
            return value.encode("utf-8") if value is not None else None

        def set(self, key: str, value: str) -> None:
            storage[key] = value

        def setex(self, key: str, ttl: int, value: str) -> None:
            storage[key] = value

        def scan_iter(self, pattern: str):
            prefix = pattern.rstrip("*")
            for key in list(storage):
                if key.startswith(prefix):
                    yield key

        def delete(self, *keys: str) -> None:
            for key in keys:
                storage.pop(key, None)

    class StubRedisModule:
        @staticmethod
        def from_url(url: str) -> StubRedisClient:
            return StubRedisClient()

    monkeypatch.setattr(cache_module.redis, "Redis", StubRedisModule)

    cache = RedisEmbeddingCache(prefix="unit-test")
    record = _record()
    cache.set(record, ttl=5)
    cached = cache.get(record.namespace, record.id)
    assert cached is not None
    assert cached.model_id == record.model_id
    cache.invalidate_namespace(record.namespace)
    assert cache.get(record.namespace, record.id) is None
