from __future__ import annotations

from types import SimpleNamespace
import uuid

import pytest

from Medical_KG_rev.embeddings.ports import EmbeddingRecord
from Medical_KG_rev.gateway.models import EmbedRequest, EmbeddingOptions
from Medical_KG_rev.gateway.services import GatewayService
from Medical_KG_rev.gateway.sse.manager import EventStreamManager
from Medical_KG_rev.observability.metrics import CROSS_TENANT_ACCESS_ATTEMPTS
from Medical_KG_rev.services.embedding.namespace.schema import EmbeddingKind, NamespaceConfig


class _StubLedger:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.processing: list[tuple[str, str]] = []
        self.metadata: list[tuple[str, dict[str, object]]] = []
        self.completed: list[tuple[str, dict[str, object]]] = []
        self.failed: list[tuple[str, str, str]] = []

    def create(self, **payload: object) -> None:
        self.created.append(payload)

    def mark_processing(self, job_id: str, *, stage: str) -> None:
        self.processing.append((job_id, stage))

    def mark_completed(self, job_id: str, *, metadata: dict[str, object] | None = None) -> None:
        self.completed.append((job_id, metadata or {}))

    def mark_failed(self, job_id: str, *, stage: str, reason: str) -> None:
        self.failed.append((job_id, stage, reason))

    def update_metadata(self, job_id: str, metadata: dict[str, object]) -> None:
        self.metadata.append((job_id, metadata))


class _StubEmbedder:
    kind = "single_vector"

    def __init__(self) -> None:
        self.requests: list[object] = []

    def embed_documents(self, request) -> list[EmbeddingRecord]:  # noqa: ANN001 - protocol compliance
        self.requests.append(request)
        vectors = [[1.0, 2.0]]
        return [
            EmbeddingRecord(
                id=request.ids[0],
                tenant_id=request.tenant_id,
                namespace=request.namespace,
                model_id="stub-model",
                model_version="v1",
                kind="single_vector",
                dim=len(vectors[0]),
                vectors=vectors,
                normalized=False,
                metadata={"provider": "vllm"},
            )
        ]


@pytest.fixture()
def gateway_service(monkeypatch: pytest.MonkeyPatch) -> GatewayService:
    service = GatewayService(
        events=EventStreamManager(),
        orchestrator=SimpleNamespace(),
        ledger=_StubLedger(),
    )
    embedder = _StubEmbedder()
    monkeypatch.setattr(service.embedding_registry, "get", lambda namespace: embedder)
    service._embedder_stub = embedder  # type: ignore[attr-defined]
    service._ledger_stub = service.ledger  # type: ignore[attr-defined]
    return service


def test_embed_normalizes_vectors(gateway_service: GatewayService) -> None:
    request = EmbedRequest(
        tenant_id="tenant-a",
        texts=["hello world"],
        namespace="single_vector.qwen3.4096.v1",
        options=EmbeddingOptions(model="stub-model", normalize=True),
    )
    response = gateway_service.embed(request)
    assert len(response.embeddings) == 1
    vector = response.embeddings[0]
    assert vector.kind == "single_vector"
    assert pytest.approx(1.0, rel=1e-6) == sum(value * value for value in vector.vector or ()) ** 0.5
    ledger: _StubLedger = gateway_service._ledger_stub  # type: ignore[attr-defined]
    assert ledger.metadata, "expected metadata update for embeddings"


def test_embed_persists_per_tenant(gateway_service: GatewayService) -> None:
    router = gateway_service.embedding_registry.storage_router
    gateway_service.embed(
        EmbedRequest(
            tenant_id="tenant-a",
            texts=["first"],
            namespace="single_vector.qwen3.4096.v1",
        )
    )
    gateway_service.embed(
        EmbedRequest(
            tenant_id="tenant-b",
            texts=["second"],
            namespace="single_vector.qwen3.4096.v1",
        )
    )
    tenant_a_records = router.buffered("faiss", tenant_id="tenant-a")
    tenant_b_records = router.buffered("faiss", tenant_id="tenant-b")
    assert tenant_a_records and tenant_b_records
    assert all(record.tenant_id == "tenant-a" for record in tenant_a_records)
    assert all(record.tenant_id == "tenant-b" for record in tenant_b_records)
    assert tenant_a_records[0].metadata["storage"]["faiss_index"].endswith("tenant-a/single_vector-qwen3-4096-v1.index")


def test_embed_rejects_empty_text(gateway_service: GatewayService) -> None:
    request = EmbedRequest(tenant_id="tenant-a", texts=[" "], namespace="single_vector.qwen3.4096.v1")
    with pytest.raises(RuntimeError):
        gateway_service.embed(request)


def test_embed_denies_cross_tenant_namespace(monkeypatch: pytest.MonkeyPatch, gateway_service: GatewayService) -> None:
    service = gateway_service
    service.namespace_registry.register(
        "single_vector.private.8.v1",
        NamespaceConfig(
            name="private",
            kind=EmbeddingKind.SINGLE_VECTOR,
            model_id="demo",
            provider="vllm",
            dim=8,
            allowed_tenants=["tenant-allowed"],
            allowed_scopes=["embed:write"],
        ),
    )
    calls: list[dict[str, str]] = []

    def _labels(**labels):  # type: ignore[no-redef]
        return SimpleNamespace(inc=lambda: calls.append(labels))

    monkeypatch.setattr(CROSS_TENANT_ACCESS_ATTEMPTS, "labels", _labels)
    request = EmbedRequest(
        tenant_id="tenant-denied",
        texts=["forbidden"],
        namespace="single_vector.private.8.v1",
    )
    with pytest.raises(RuntimeError):
        service.embed(request)
    assert calls and calls[0]["source_tenant"] == "tenant-denied"
