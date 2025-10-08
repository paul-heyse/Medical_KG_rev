import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("pydantic")

from Medical_KG_rev.gateway.models import EmbedRequest
from Medical_KG_rev.gateway.services import GatewayService, JobEvent


@dataclass
class StubLedger:
    created: list[dict[str, Any]] = field(default_factory=list)
    processing: list[dict[str, Any]] = field(default_factory=list)
    completed: list[dict[str, Any]] = field(default_factory=list)
    metadata_updates: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs):  # type: ignore[override]
        self.created.append(kwargs)

    def mark_processing(self, job_id: str, stage: str) -> None:
        self.processing.append({"job_id": job_id, "stage": stage})

    def mark_completed(self, job_id: str, metadata: dict[str, Any]) -> None:
        self.completed.append({"job_id": job_id, "metadata": metadata})

    def mark_failed(self, job_id: str, stage: str, reason: str) -> None:  # pragma: no cover - not used
        pass

    def update_metadata(self, job_id: str, metadata: dict[str, Any]) -> None:
        self.metadata_updates.append({"job_id": job_id, "metadata": metadata})


@dataclass
class StubEvents:
    events: list[JobEvent] = field(default_factory=list)

    def publish(self, event: JobEvent) -> None:
        self.events.append(event)


@pytest.fixture()
def gateway_service(monkeypatch: pytest.MonkeyPatch) -> tuple[GatewayService, StubLedger, StubEvents]:
    monkeypatch.setattr(GatewayService, "_ensure_pipeline_components", lambda self: None)
    ledger = StubLedger()
    events = StubEvents()
    service = GatewayService(events=events, orchestrator=None, ledger=ledger)  # type: ignore[arg-type]
    return service, ledger, events


def build_request(namespace: str = "single_vector.test.v1", *, normalize: bool = True) -> EmbedRequest:
    return EmbedRequest(
        tenant_id="tenant",
        inputs=["alpha", "beta"],
        model="demo-model",
        namespace=namespace,
        normalize=normalize,
    )


def test_embed_returns_vectors_with_metadata(gateway_service, monkeypatch: pytest.MonkeyPatch) -> None:
    service, ledger, events = gateway_service
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=1))
    vectors = service.embed(build_request())
    assert len(vectors) == 2
    assert vectors[0].metadata["normalized"] is True


def test_embed_updates_ledger_metadata(gateway_service, monkeypatch: pytest.MonkeyPatch) -> None:
    service, ledger, events = gateway_service
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=2))
    service.embed(build_request())
    assert ledger.metadata_updates[0]["metadata"]["embeddings"] == 2


def test_embed_emits_job_events(gateway_service, monkeypatch: pytest.MonkeyPatch) -> None:
    service, ledger, events = gateway_service
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=3))
    service.embed(build_request())
    assert {event.type for event in events.events} == {"jobs.started", "jobs.completed"}


def test_embed_completion_payload_includes_namespace(gateway_service, monkeypatch: pytest.MonkeyPatch) -> None:
    service, ledger, events = gateway_service
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=4))
    service.embed(build_request(namespace="sparse.namespace"))
    payload = ledger.completed[0]["metadata"]
    assert payload["namespace"] == "sparse.namespace"


def test_embed_respects_normalize_flag(gateway_service, monkeypatch: pytest.MonkeyPatch) -> None:
    service, ledger, events = gateway_service
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=5))
    vectors = service.embed(build_request(normalize=False))
    assert vectors[0].metadata["normalized"] is False


def test_embed_handles_empty_inputs(gateway_service, monkeypatch: pytest.MonkeyPatch) -> None:
    service, ledger, events = gateway_service
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=6))
    request = build_request()
    request = request.model_copy(update={"inputs": []})
    vectors = service.embed(request)
    assert vectors == []
    assert ledger.metadata_updates[0]["metadata"]["embeddings"] == 0


def test_embed_records_job_creation(gateway_service, monkeypatch: pytest.MonkeyPatch) -> None:
    service, ledger, events = gateway_service
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=7))
    service.embed(build_request())
    assert ledger.created[0]["tenant_id"] == "tenant"
    assert ledger.processing[0]["stage"] == "embed"


def test_embed_completion_metadata(gateway_service, monkeypatch: pytest.MonkeyPatch) -> None:
    service, ledger, events = gateway_service
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=8))
    service.embed(build_request())
    completion = ledger.completed[0]["metadata"]
    assert completion["model"] == "demo-model"
    assert completion["embeddings"] == 2
