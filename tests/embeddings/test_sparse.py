from Medical_KG_rev.embeddings.ports import EmbedderConfig, EmbeddingRequest
from Medical_KG_rev.embeddings.sparse.splade import (
    PyseriniSparseEmbedder,
    SPLADEDocEmbedder,
    build_rank_features_mapping,
)


def _request(namespace: str) -> EmbeddingRequest:
    return EmbeddingRequest(tenant_id="tenant", namespace=namespace, texts=["Token alpha beta"])


def test_splade_vocab_tracking() -> None:
    config = EmbedderConfig(
        name="splade",
        provider="splade-doc",
        kind="sparse",
        namespace="sparse.splade.400.v1",
        model_id="splade",
        parameters={"normalization": "l1"},
    )
    embedder = SPLADEDocEmbedder(config)
    request = _request(config.namespace)
    embedder.embed_documents(request)
    vocab = embedder.vocabulary_snapshot(2)
    assert vocab


def test_pyserini_normalization() -> None:
    config = EmbedderConfig(
        name="pyserini",
        provider="pyserini",
        kind="sparse",
        namespace="sparse.pyserini.0.v1",
        model_id="pyserini",
        parameters={"normalization": "max"},
    )
    embedder = PyseriniSparseEmbedder(config)
    request = _request(config.namespace)
    records = embedder.embed_documents(request)
    weights = records[0].terms
    assert weights
    assert max(weights.values()) == 1.0


def test_build_rank_features_mapping() -> None:
    mapping = build_rank_features_mapping("sparse.splade.v1")
    assert mapping["properties"]["sparse_splade_v1"]["type"] == "rank_features"
