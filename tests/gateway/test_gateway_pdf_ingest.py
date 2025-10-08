from __future__ import annotations

from types import SimpleNamespace

import pytest

from Medical_KG_rev.gateway.models import IngestionRequest
from Medical_KG_rev.gateway.services import GatewayService
from Medical_KG_rev.gateway.sse.manager import EventStreamManager
from Medical_KG_rev.orchestration.dagster.configuration import PipelineConfigLoader
from Medical_KG_rev.orchestration.dagster.runtime import DagsterRunResult
from Medical_KG_rev.orchestration.stages.contracts import PipelineState
from Medical_KG_rev.orchestration.ledger import JobLedger


class _StubOrchestrator:
    """Minimal orchestrator stub exposing pipeline metadata for tests."""

    def __init__(self, loader: PipelineConfigLoader, pipelines: list[str]) -> None:
        self.pipeline_loader = loader
        self._pipelines = list(pipelines)
        self.submissions: list[dict[str, object]] = []

    def available_pipelines(self) -> list[str]:
        return list(self._pipelines)

    def submit(self, *, pipeline: str, context, adapter_request, payload) -> DagsterRunResult:  # noqa: ANN001 - protocol
        self.submissions.append(
            {
                "pipeline": pipeline,
                "context": context,
                "adapter_request": adapter_request,
                "payload": payload,
            }
        )
        state = PipelineState.initialise(
            context=context,
            adapter_request=adapter_request,
            payload=payload,
        )
        return DagsterRunResult(
            pipeline=pipeline,
            success=True,
            state=state,
            dagster_result=SimpleNamespace(success=True),
        )


@pytest.fixture()
def gateway_service() -> GatewayService:
    loader = PipelineConfigLoader("config/orchestration/pipelines")
    orchestrator = _StubOrchestrator(loader, ["auto", "pdf-two-phase"])
    service = GatewayService(
        events=EventStreamManager(),
        orchestrator=orchestrator,  # type: ignore[arg-type]
        ledger=JobLedger(),
    )
    service._orchestrator_stub = orchestrator  # type: ignore[attr-defined]
    return service


def test_resolve_pipeline_prefers_pdf_for_pmc(gateway_service: GatewayService) -> None:
    service = gateway_service
    pipeline = service._resolve_pipeline("pmc", {"id": "doc-123"})
    assert pipeline == "pdf-two-phase"


def test_resolve_pipeline_falls_back_to_pdf_document_type(gateway_service: GatewayService) -> None:
    service = gateway_service
    pipeline = service._resolve_pipeline(
        "unknown-source",
        {"id": "doc-456", "document_type": "pdf"},
    )
    assert pipeline == "pdf-two-phase"


def test_resolve_pipeline_defaults_to_auto_for_non_pdf(gateway_service: GatewayService) -> None:
    service = gateway_service
    pipeline = service._resolve_pipeline(
        "unknown-source",
        {"id": "doc-789", "document_type": "html"},
    )
    assert pipeline == "auto"


def test_submit_dagster_job_records_pdf_metadata(gateway_service: GatewayService) -> None:
    service = gateway_service
    orchestrator: _StubOrchestrator = service._orchestrator_stub  # type: ignore[attr-defined]

    request = IngestionRequest(
        tenant_id="tenant-a",
        items=[{"id": "pmc-001", "document_type": "pdf"}],
        metadata={"source": "pmc"},
    )

    status = service._submit_dagster_job(
        dataset="pmc",
        request=request,
        item=request.items[0],
        metadata={"source": "pmc"},
    )

    assert status.status == "completed"
    assert status.metadata["pipeline"] == "pdf-two-phase"
    assert status.metadata["duplicate"] is False
    assert status.metadata["state"]["context"]["pipeline_name"] == "pdf-two-phase"
    assert orchestrator.submissions, "expected Dagster submission"
    submission = orchestrator.submissions[0]
    assert submission["pipeline"] == "pdf-two-phase"

    entry = service.ledger.get(status.job_id)
    assert entry is not None
    assert entry.metadata["pipeline_version"] == "2025-01-01"
    assert entry.metadata["payload"]["dataset"] == "pmc"
    assert entry.metadata["adapter_request"]["parameters"]["dataset"] == "pmc"
