from Medical_KG_rev.services.reranking.factory import RerankerFactory
from Medical_KG_rev.services.reranking.models import RerankerConfig


def test_factory_caches_instances() -> None:
    factory = RerankerFactory()
    first = factory.resolve("cross_encoder:bge")
    second = factory.resolve("cross_encoder:bge")
    assert first is second


def test_factory_cache_handles_configurations() -> None:
    factory = RerankerFactory()
    config = RerankerConfig(method="cross_encoder", model="custom", batch_size=16)
    first = factory.resolve("cross_encoder:minilm", config)
    second = factory.resolve("cross_encoder:minilm", config)
    assert first is second

    factory.clear_cache()
    third = factory.resolve("cross_encoder:minilm", config)
    assert third is not first
