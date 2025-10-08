"""Tests that embedding worker pushes vectors into the vector store."""

from __future__ import annotations

import pytest

from Medical_KG_rev.embeddings.namespace import NamespaceManager
from Medical_KG_rev.embeddings.ports import EmbedderConfig, EmbeddingRecord
from Medical_KG_rev.embeddings.utils.tokenization import TokenizerCache
from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.services.embedding.service import (
    EmbeddingModelRegistry,
    EmbeddingRequest,
    EmbeddingWorker,
)
from Medical_KG_rev.services.vector_store.models import NamespaceConfig, IndexParams
from Medical_KG_rev.services.vector_store.registry import NamespaceRegistry
from Medical_KG_rev.services.vector_store.service import VectorStoreService


class DummyEmbedder:
    def embed_documents(self, request):  # type: ignore[override]
        return [
            EmbeddingRecord(
                id=request.ids[0] if request.ids else "chunk-1",
                tenant_id=request.tenant_id,
                namespace=request.namespace,
                model_id="Qwen/Qwen2.5-Embedding-8B-Instruct",
                model_version="v1",
                kind="single_vector",
                dim=4096,
                vectors=[[float(i) / 100 for i in range(4096)]],
                metadata={"foo": "bar", "provider": "vllm"},
            )
        ]


class StubVectorStore:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def create_or_update_collection(self, **kwargs):  # type: ignore[override]
        pass

    def list_collections(self, **kwargs):  # type: ignore[override]
        return []

    def upsert(self, *, tenant_id: str, namespace: str, records):  # type: ignore[override]
        for record in records:
            self.records.append(
                {
                    "namespace": namespace,
                    "vector_id": record.vector_id,
                    "values": record.values,
                }
            )

    def query(self, **kwargs):  # type: ignore[override]
        return []

    def delete(self, **kwargs):  # type: ignore[override]
        return 0


@pytest.fixture()
def embedding_worker(monkeypatch: pytest.MonkeyPatch):
    namespace_manager = NamespaceManager()
    fake_config = EmbedderConfig(
        name="qwen3",
        provider="vllm",
        kind="single_vector",
        namespace="single_vector.qwen3.4096.v1",
        model_id="Qwen/Qwen2.5-Embedding-8B-Instruct",
        model_version="v1",
        dim=4096,
        parameters={"max_tokens": 8192, "endpoint": "http://localhost:8001/v1"},
        requires_gpu=False,
    )
    namespace_manager.register(fake_config)
    registry = NamespaceRegistry()
    vector_store_adapter = StubVectorStore()
    service = VectorStoreService(vector_store_adapter, registry)
    worker = EmbeddingWorker(namespace_manager=namespace_manager, vector_store=service)
    worker.namespace_manager.register(fake_config)

    monkeypatch.setattr(EmbeddingWorker, "_resolve_configs", lambda self, request: [fake_config])
    monkeypatch.setattr(EmbeddingModelRegistry, "get", lambda self, config: DummyEmbedder())
    monkeypatch.setattr(TokenizerCache, "ensure_within_limit", lambda *args, **kwargs: None)
    service.ensure_namespace(
        context=SecurityContext(subject="tester", tenant_id="tenant", scopes={"index:write"}),
        config=NamespaceConfig(
            name=fake_config.namespace,
            params=IndexParams(dimension=4096),
        ),
    )
    assert registry.get(tenant_id="tenant", namespace=fake_config.namespace)
    return worker, vector_store_adapter, fake_config


def test_worker_upserts_vectors(embedding_worker) -> None:
    worker, vector_store, config = embedding_worker
    request = EmbeddingRequest(
        tenant_id="tenant",
        chunk_ids=["chunk-1"],
        texts=["sample"],
    )
    response = worker.run(request)
    assert response.vectors[0].metadata["storage_target"] == "faiss"
    assert vector_store.records[0]["vector_id"] == "chunk-1"
    assert vector_store.records[0]["namespace"] == config.namespace


def test_worker_resolves_configs_via_registry(monkeypatch) -> None:
    namespace_manager = NamespaceManager()
    fake_config = EmbedderConfig(
        name="qwen3",
        provider="vllm",
        kind="single_vector",
        namespace="single_vector.qwen3.4096.v1",
        model_id="Qwen/Qwen2.5-Embedding-8B-Instruct",
        model_version="v1",
        dim=4096,
        parameters={"max_tokens": 8192, "endpoint": "http://localhost:8001/v1"},
        requires_gpu=False,
    )

    worker = EmbeddingWorker(namespace_manager=namespace_manager, vector_store=None)
    worker.namespace_manager.register(fake_config)
    monkeypatch.setattr(TokenizerCache, "ensure_within_limit", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        EmbeddingModelRegistry,
        "get",
        lambda self, config: DummyEmbedder(),
    )

    captured: dict[str, object | None] = {}

    def tracking_resolve(self, *, models=None, namespaces=None):  # type: ignore[override]
        captured["models"] = models
        captured["namespaces"] = namespaces
        return [fake_config]

    monkeypatch.setattr(EmbeddingModelRegistry, "resolve", tracking_resolve)

    request = EmbeddingRequest(
        tenant_id="tenant",
        chunk_ids=["chunk-1"],
        texts=["sample"],
        namespaces=[fake_config.namespace],
    )
    worker.run(request)

    assert captured["namespaces"] == [fake_config.namespace]
    assert captured["models"] is None
