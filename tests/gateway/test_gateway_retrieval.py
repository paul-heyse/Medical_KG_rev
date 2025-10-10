from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from Medical_KG_rev.gateway.models import RetrieveRequest
from Medical_KG_rev.gateway.services import GatewayService
from Medical_KG_rev.gateway.sse.manager import EventStreamManager
from Medical_KG_rev.services.retrieval.retrieval_service import RetrievalResult


class _DummyEvents(EventStreamManager):
    def __init__(self) -> None:
        super().__init__()
        self.published: list[Any] = []

    def publish(self, event: Any) -> None:  # type: ignore[override]
        self.published.append(event)


@dataclass
class _LedgerRecord:
    job_id: str
    metadata: Mapping[str, Any]


class _DummyLedger:
    def __init__(self) -> None:
        self.created: list[_LedgerRecord] = []
        self.processing: list[str] = []
        self.completed: list[_LedgerRecord] = []
        self.failed: list[_LedgerRecord] = []
        self.updated: list[_LedgerRecord] = []

    def create(
        self,
        *,
        job_id: str,
        doc_key: str,
        tenant_id: str,
        pipeline: str,
        metadata: Mapping[str, Any],
    ) -> None:
        self.created.append(_LedgerRecord(job_id=job_id, metadata=dict(metadata)))

    def mark_processing(self, job_id: str, stage: str) -> None:
        self.processing.append(job_id)

    def mark_completed(self, job_id: str, metadata: Mapping[str, Any]) -> None:
        self.completed.append(_LedgerRecord(job_id=job_id, metadata=dict(metadata)))

    def mark_failed(self, job_id: str, stage: str, reason: str) -> None:
        self.failed.append(_LedgerRecord(job_id=job_id, metadata={"reason": reason}))

    def update_metadata(self, job_id: str, metadata: Mapping[str, Any]) -> None:
        self.updated.append(_LedgerRecord(job_id=job_id, metadata=dict(metadata)))


class _DummyOrchestrator:
    pass


class _StubRetrievalService:
    def __init__(self, results: Sequence[RetrievalResult]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    def search(self, **params: Any) -> Sequence[RetrievalResult]:
        self.calls.append(params)
        return list(self._results)


def _gateway_with_results(
    results: Sequence[RetrievalResult],
) -> tuple[GatewayService, _StubRetrievalService, _DummyLedger]:
    events = _DummyEvents()
    ledger = _DummyLedger()
    orchestrator = _DummyOrchestrator()
    stub_service = _StubRetrievalService(results)
    gateway = GatewayService(
        events=events,
        orchestrator=orchestrator,
        ledger=ledger,
        retrieval_service=stub_service,
    )
    return gateway, stub_service, ledger


def test_retrieve_uses_hybrid_service_metadata() -> None:
    results = [
        RetrievalResult(
            id="doc-1",
            text="table",
            retrieval_score=0.82,
            rerank_score=1.15,
            highlights=[],
            metadata={
                "title": "Adverse Event Table",
                "summary": "Structured AE data",
                "source": "bm25",
                "is_table": True,
                "reranking": {"model": {"key": "bge"}, "applied": True},
                "component_scores": {"bm25": 0.82},
                "components": {"errors": ["dense:Timeout"]},
                "pipeline_metrics": {"timing": {"fusion": 0.01}},
            },
        ),
        RetrievalResult(
            id="doc-2",
            text="narrative",
            retrieval_score=0.6,
            rerank_score=None,
            highlights=[],
            metadata={"title": "Narrative", "source": "dense", "component_scores": {"dense": 0.6}},
        ),
    ]

    gateway, stub_service, ledger = _gateway_with_results(results)

    request = RetrieveRequest(
        tenant_id="tenant-a",
        query="adverse events",
        top_k=2,
        filters={},
        rerank=True,
        rerank_model="custom-model",
        table_only=True,
        metadata={"dataset": "pmc"},
    )

    response = gateway.retrieve(request)

    assert response.documents[0].id == "doc-1"
    assert response.documents[0].source == "bm25"
    assert response.intent["detected"] == "tabular"
    assert response.rerank_metrics["requested_model"] == "custom-model"
    assert response.rerank_metrics["component_errors"] == ["dense:Timeout"]
    assert response.partial is True and response.degraded is True
    assert response.errors[0].detail == "dense:Timeout"
    assert "fusion" in response.stage_timings

    call = stub_service.calls[-1]
    assert call["index"] == "pmc-chunks"
    assert call["query"] == "adverse events"

    updated = ledger.updated[-1]
    assert updated.metadata["index"] == "pmc-chunks"
