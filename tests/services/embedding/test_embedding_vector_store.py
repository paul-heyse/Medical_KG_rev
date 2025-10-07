"""Tests that embedding worker pushes vectors into the vector store."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from Medical_KG_rev.embeddings.namespace import NamespaceManager, NamespaceConfig as EmbeddingNamespaceConfig
from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.services.embedding.service import (
    EmbeddingRequest,
    EmbeddingWorker,
)
from Medical_KG_rev.services.vector_store.models import NamespaceConfig, IndexParams
from Medical_KG_rev.services.vector_store.registry import NamespaceRegistry
from Medical_KG_rev.services.vector_store.service import VectorStoreService


class DummyEmbedder:
    def embed_documents(self, request):  # type: ignore[override]
        class Record:
            def __init__(self) -> None:
                self.id = "chunk-1"
                self.model_id = "test"
                self.namespace = request.namespace
                self.kind = "single_vector"
                self.vectors = [
                    [float(i) / 100 for i in range(128)]
                ]
                self.terms = None
                self.metadata = {"foo": "bar"}
                self.dim = 128

        return [Record()]


class StubEmbedderFactory:
    def __init__(self, worker: EmbeddingWorker) -> None:
        self.worker = worker

    def get(self, config):  # type: ignore[override]
        return DummyEmbedder()


class StubVectorStore:
    def __init__(self) -> None:
        self.records: list[tuple[str, Sequence[float]]] = []

    def create_or_update_collection(self, **kwargs):  # type: ignore[override]
        pass

    def list_collections(self, **kwargs):  # type: ignore[override]
        return []

    def upsert(self, *, tenant_id: str, namespace: str, records):  # type: ignore[override]
        for record in records:
            self.records.append((record.vector_id, record.values))

    def query(self, **kwargs):  # type: ignore[override]
        return []

    def delete(self, **kwargs):  # type: ignore[override]
        return 0


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
    vector_store_adapter = StubVectorStore()
    service = VectorStoreService(vector_store_adapter, registry)
    worker = EmbeddingWorker(namespace_manager=namespace_manager, vector_store=service)
    fake_config = FakeConfig()
    monkeypatch.setattr(worker, "_resolve_configs", lambda request: [fake_config])
    monkeypatch.setattr(worker, "factory", StubEmbedderFactory(worker))
    service.ensure_namespace(
        context=SecurityContext(subject="tester", tenant_id="tenant", scopes={"index:write"}),
        config=NamespaceConfig(
            name="default",
            params=IndexParams(dimension=128),
        ),
    )
    assert registry.get(tenant_id="tenant", namespace="default")
    return worker, vector_store_adapter


def test_worker_upserts_vectors(embedding_worker) -> None:
    worker, vector_store = embedding_worker
    request = EmbeddingRequest(
        tenant_id="tenant",
        chunk_ids=["chunk-1"],
        texts=["sample"],
    )
    response = worker.run(request)
    assert response.vectors[0].metadata["storage_target"]
    assert vector_store.records[0][0] == "chunk-1"
