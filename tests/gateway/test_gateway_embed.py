from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
import uuid

import pytest

from Medical_KG_rev.embeddings.ports import EmbeddingRecord
from Medical_KG_rev.gateway.models import EmbedRequest, EmbeddingOptions, JobEvent
from Medical_KG_rev.gateway.services import GatewayService
from Medical_KG_rev.gateway.sse.manager import EventStreamManager


@dataclass
class StubLedger:
    created: list[dict[str, object]] = field(default_factory=list)
    processing: list[dict[str, object]] = field(default_factory=list)
    completed: list[dict[str, object]] = field(default_factory=list)
    metadata_updates: list[dict[str, object]] = field(default_factory=list)
    failed: list[dict[str, object]] = field(default_factory=list)

    def create(self, **kwargs) -> None:  # type: ignore[override]
        self.created.append(kwargs)

    def mark_processing(self, job_id: str, stage: str) -> None:
        self.processing.append({"job_id": job_id, "stage": stage})

    def mark_completed(self, job_id: str, metadata: dict[str, object]) -> None:
        self.completed.append({"job_id": job_id, "metadata": metadata})

    def mark_failed(self, job_id: str, stage: str, reason: str) -> None:
        self.failed.append({"job_id": job_id, "stage": stage, "reason": reason})

    def update_metadata(self, job_id: str, metadata: dict[str, object]) -> None:
        self.metadata_updates.append({"job_id": job_id, "metadata": metadata})


@dataclass
class StubEvents:
    events: list[JobEvent] = field(default_factory=list)

    def publish(self, event: JobEvent) -> None:
        self.events.append(event)


class StubEmbedder:
    def __init__(self, kind: str = "single_vector") -> None:
        self.kind = kind
        self.requests: list[object] = []

    def embed_documents(self, request) -> list[EmbeddingRecord]:  # noqa: ANN001 - protocol compliance
        self.requests.append(request)
        if self.kind == "sparse":
            return [
                EmbeddingRecord(
                    id=request.ids[0],
                    tenant_id=request.tenant_id,
                    namespace=request.namespace,
                    model_id="splade",
                    model_version="v3",
                    kind="sparse",
                    dim=0,
                    vectors=None,
                    terms={"hello": 1.5, "world": 0.8},
                    normalized=False,
                    metadata={"provider": "pyserini"},
                )
            ]
        vectors = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
        return [
            EmbeddingRecord(
                id=request.ids[index],
                tenant_id=request.tenant_id,
                namespace=request.namespace,
                model_id="qwen3",
                model_version="v1",
                kind="single_vector",
                dim=len(vectors[index]),
                vectors=[vectors[index]],
                normalized=False,
                metadata={"provider": "vllm", "input_index": index},
            )
            for index in range(len(request.texts))
        ]


@pytest.fixture()
def gateway_service(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[GatewayService, StubLedger, StubEvents, StubEmbedder]:
    ledger = StubLedger()
    events = StubEvents()
    service = GatewayService(
        events=events,
        orchestrator=SimpleNamespace(),
        ledger=ledger,
    )
    embedder = StubEmbedder()
    monkeypatch.setattr(service.embedding_registry, "get", lambda namespace: embedder)
    return service, ledger, events, embedder


def _build_request(
    namespace: str = "single_vector.qwen3.4096.v1", *, normalize: bool = True
) -> EmbedRequest:
    return EmbedRequest(
        tenant_id="tenant-123",
        texts=["alpha", "beta"],
        namespace=namespace,
        options=EmbeddingOptions(normalize=normalize, model="demo-model"),
    )


def test_embed_returns_vectors_with_metadata(
    gateway_service, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, ledger, events, embedder = gateway_service
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=1))
    response = service.embed(_build_request())
    assert len(response.embeddings) == 2
    first = response.embeddings[0]
    assert first.metadata["tenant_id"] == "tenant-123"
    assert first.metadata["storage"]["faiss_index"].endswith(
        "tenant-123/single_vector-qwen3-4096-v1.index"
    )
    assert all(event.type in {"jobs.started", "jobs.completed"} for event in events.events)
    assert embedder.requests, "Expected embedder to be invoked"


def test_embed_updates_ledger_metadata(gateway_service, monkeypatch: pytest.MonkeyPatch) -> None:
    service, ledger, events, _ = gateway_service
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=2))
    service.embed(_build_request())
    metadata = ledger.metadata_updates[0]["metadata"]
    assert metadata["embeddings"] == 2
    assert metadata["tenant_id"] == "tenant-123"


def test_embed_handles_sparse_namespace(gateway_service, monkeypatch: pytest.MonkeyPatch) -> None:
    service, ledger, events, embedder = gateway_service
    sparse_embedder = StubEmbedder(kind="sparse")
    monkeypatch.setattr(service.embedding_registry, "get", lambda namespace: sparse_embedder)
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=3))
    response = service.embed(_build_request(namespace="sparse.splade_v3.400.v1"))
    assert response.embeddings[0].terms == {"hello": 1.5, "world": 0.8}
    assert response.embeddings[0].vector is None
    assert response.embeddings[0].metadata["storage"]["opensearch_index"].endswith("tenant-123")


def test_embed_handles_empty_inputs(gateway_service, monkeypatch: pytest.MonkeyPatch) -> None:
    service, ledger, events, _ = gateway_service
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=4))
    request = _build_request()
    request = request.model_copy(update={"texts": []})
    response = service.embed(request)
    assert response.embeddings == []
    assert ledger.metadata_updates[0]["metadata"]["embeddings"] == 0


def test_list_namespaces_returns_namespace_info(gateway_service) -> None:
    service, ledger, events, _ = gateway_service
    namespaces = service.list_namespaces(tenant_id="tenant-123")
    ids = {entry.id for entry in namespaces}
    assert "single_vector.qwen3.4096.v1" in ids


def test_validate_namespace_tokens(monkeypatch: pytest.MonkeyPatch, gateway_service) -> None:
    service, ledger, events, _ = gateway_service
    fake_tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=False: list(text.split())
    )
    monkeypatch.setattr(service.namespace_registry, "get_tokenizer", lambda ns: fake_tokenizer)
    monkeypatch.setattr(service.namespace_registry, "get_max_tokens", lambda ns: 5)
    result = service.validate_namespace_texts(
        tenant_id="tenant-123",
        namespace="single_vector.qwen3.4096.v1",
        texts=["hello world"],
    )
    assert result.results[0].token_count == 2
