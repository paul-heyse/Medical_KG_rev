from Medical_KG_rev.embeddings.multi_vector.colbert import (
    ColBERTRagatouilleEmbedder,
    ColbertShardManager,
    maxsim_score,
)
from Medical_KG_rev.embeddings.ports import EmbedderConfig, EmbeddingRequest


def test_colbert_shard_manager_assigns_shards() -> None:
    manager = ColbertShardManager()
    manager.register("a", dimension=64, capacity=2)
    manager.register("b", dimension=64, capacity=2)
    shard = manager.store("doc-1", [[0.1] * 64])
    assert shard in {"a", "b"}


def test_maxsim_score_monotonic() -> None:
    query = [[1.0, 0.0], [0.0, 1.0]]
    doc = [[1.0, 0.0], [0.5, 0.5]]
    score = maxsim_score(query, doc)
    assert score >= 1.0


def test_colbert_embedder_records_shard_metadata() -> None:
    config = EmbedderConfig(
        name="colbert",
        provider="colbert",
        kind="multi_vector",
        namespace="multi.colbert.128.v1",
        model_id="colbert/colbertv2",
        dim=128,
        parameters={"shards": 2},
    )
    embedder = ColBERTRagatouilleEmbedder(config)
    request = EmbeddingRequest(
        tenant_id="tenant",
        namespace=config.namespace,
        texts=["sample text for colbert"],
    )
    records = embedder.embed_documents(request)
    assert records[0].metadata["shard"].startswith("shard-")
