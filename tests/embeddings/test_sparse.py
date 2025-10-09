from Medical_KG_rev.embeddings.ports import EmbedderConfig, EmbeddingRequest
import sys
from types import ModuleType

import pytest

from Medical_KG_rev.embeddings.ports import EmbedderConfig, EmbeddingRequest
from Medical_KG_rev.embeddings.sparse.splade import (
    PyseriniSparseEmbedder,
    build_rank_features_mapping,
)


@pytest.fixture(autouse=True)
def fake_pyserini(monkeypatch: pytest.MonkeyPatch):
    module = ModuleType("pyserini")
    encode = ModuleType("pyserini.encode")

    class DocumentEncoder:
        called = 0

        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def encode(self, text: str, top_k: int = 400):  # noqa: D401 - mirrors Pyserini
            DocumentEncoder.called += 1
            tokens = [token for token in text.lower().split() if token]
            return {token: float(index + 1) for index, token in enumerate(tokens[:top_k])}

    class QueryEncoder(DocumentEncoder):
        called = 0

        def encode(self, text: str, top_k: int = 400):
            QueryEncoder.called += 1
            return super().encode(text, top_k=top_k)

    encode.SpladeDocumentEncoder = DocumentEncoder
    encode.SpladeQueryEncoder = QueryEncoder
    module.encode = encode
    monkeypatch.setitem(sys.modules, "pyserini", module)
    monkeypatch.setitem(sys.modules, "pyserini.encode", encode)
    yield
    DocumentEncoder.called = 0
    QueryEncoder.called = 0


def _request(namespace: str, text: str = "Token alpha beta") -> EmbeddingRequest:
    return EmbeddingRequest(tenant_id="tenant", namespace=namespace, texts=[text])


def test_pyserini_document_expansion_respects_top_k() -> None:
    config = EmbedderConfig(
        name="splade",
        provider="pyserini",
        kind="sparse",
        namespace="sparse.splade_v3.400.v1",
        model_id="naver/splade-v3",
        parameters={"top_k": 2},
    )
    embedder = PyseriniSparseEmbedder(config)
    records = embedder.embed_documents(_request(config.namespace))
    weights = records[0].terms
    assert weights is not None
    assert len(weights) == 2
    assert all(value > 0 for value in weights.values())


def test_pyserini_query_mode_uses_query_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    config = EmbedderConfig(
        name="splade-query",
        provider="pyserini",
        kind="sparse",
        namespace="sparse.splade_query.400.v1",
        model_id="naver/splade-v3",
        parameters={"mode": "query", "top_k": 4},
    )
    embedder = PyseriniSparseEmbedder(config)
    records = embedder.embed_queries(_request(config.namespace, text="diabetes treatment"))
    weights = records[0].terms
    assert weights and "diabetes" in weights


def test_pyserini_handles_empty_text() -> None:
    config = EmbedderConfig(
        name="splade",
        provider="pyserini",
        kind="sparse",
        namespace="sparse.splade_v3.400.v1",
        model_id="naver/splade-v3",
    )
    embedder = PyseriniSparseEmbedder(config)
    records = embedder.embed_documents(_request(config.namespace, text=""))
    assert records[0].terms == {}


def test_build_rank_features_mapping() -> None:
    mapping = build_rank_features_mapping("sparse.splade.v1")
    assert mapping["properties"]["sparse_splade_v1"]["type"] == "rank_features"
