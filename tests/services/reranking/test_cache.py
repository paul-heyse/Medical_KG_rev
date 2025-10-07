from Medical_KG_rev.services.reranking.pipeline.cache import (
    RedisCacheBackend,
    RerankCacheManager,
)
from Medical_KG_rev.services.reranking import RerankResult


class FakeRedis:
    def __init__(self) -> None:
        self.store = {}

    def get(self, key: str):  # noqa: D401 - simple fake
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:  # noqa: ARG002
        self.store[key] = value

    def scan_iter(self, match: str):
        return [key for key in self.store if key.startswith(match.split('*')[0])]

    def delete(self, *keys: str) -> None:
        for key in keys:
            self.store.pop(key, None)


def test_redis_backend_serialisation():
    backend = RedisCacheBackend(FakeRedis())
    manager = RerankCacheManager(ttl_seconds=10, backend=backend)
    manager.store(
        "ce",
        "tenant",
        "v1",
        [RerankResult(doc_id="doc", score=0.5, rank=1)],
    )
    cached = manager.lookup("ce", "tenant", "doc", "v1")
    assert cached is not None and cached.score == 0.5
