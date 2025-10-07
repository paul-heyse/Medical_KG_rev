"""Tests ensuring embedding worker outputs metadata-rich vectors."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from Medical_KG_rev.embeddings.namespace import NamespaceManager, NamespaceConfig as EmbeddingNamespaceConfig
from Medical_KG_rev.services.embedding.service import (
    EmbeddingRequest,
    EmbeddingWorker,
)
from Medical_KG_rev.services.vector_store.models import NamespaceConfig, IndexParams
from Medical_KG_rev.services.vector_store.registry import NamespaceRegistry


class DummyEmbedder:
    def embed_documents(self, request):  # type: ignore[override]
        class Record:
            def __init__(self, namespace: str, metadata: Sequence[dict[str, object]]) -> None:
                self.id = "chunk-1"
                self.model_id = "test"
                self.namespace = namespace
                self.kind = "single_vector"
                self.vectors = [
                    [float(i) / 100 for i in range(128)]
                ]
                self.terms = None
                self.metadata = {"foo": "bar", **(metadata[0] if metadata else {})}
                self.dim = 128

        return [Record(request.namespace, request.metadata)]


class StubEmbedderFactory:
    def __init__(self, worker: EmbeddingWorker) -> None:
        self.worker = worker

    def get(self, config):  # type: ignore[override]
        return DummyEmbedder()


@pytest.fixture()
def embedding_worker(monkeypatch: pytest.MonkeyPatch):
    class FakeConfig:
        def __init__(self) -> None:
            self.name = "fake"
            self.namespace = "default"
            self.requires_gpu = False
            self.kind = "single_vector"
            self.model_id = "fake-model"
            self.model_version = "v1"
            self.dim = 128

    namespace_manager = NamespaceManager()
    namespace_manager._namespaces[  # type: ignore[attr-defined]
        "single_vector.bge_small_en.384.v1"
    ] = EmbeddingNamespaceConfig(
        name="single_vector.bge_small_en.384.v1",
        kind="single_vector",
        expected_dim=128,
        model_id="test",
        model_version="v1",
        embedder_name="test",
    )
    namespace_manager._namespaces[  # type: ignore[attr-defined]
        "default"
    ] = EmbeddingNamespaceConfig(
        name="default",
        kind="single_vector",
        expected_dim=128,
        model_id="fake-model",
        model_version="v1",
        embedder_name="fake",
    )
    registry = NamespaceRegistry()
    worker = EmbeddingWorker(namespace_manager=namespace_manager)
    fake_config = FakeConfig()
    monkeypatch.setattr(worker, "_resolve_configs", lambda request: [fake_config])
    monkeypatch.setattr(worker, "factory", StubEmbedderFactory(worker))
    registry.register(
        tenant_id="tenant",
        config=NamespaceConfig(
            name="default",
            params=IndexParams(dimension=128),
        ),
    )
    return worker


def test_worker_generates_metadata_rich_vectors(embedding_worker) -> None:
    worker = embedding_worker
    request = EmbeddingRequest(
        tenant_id="tenant",
        chunk_ids=["chunk-1"],
        texts=["sample"],
        metadatas=[{"document_id": "doc-1", "text": "sample"}],
    )
    response = worker.run(request)
    assert response.vectors[0].metadata["storage_target"] == "qdrant"
    assert response.vectors[0].metadata["document_id"] == "doc-1"
    buffered = worker.storage_router.buffered("qdrant")
    assert buffered and buffered[0].metadata["document_id"] == "doc-1"
