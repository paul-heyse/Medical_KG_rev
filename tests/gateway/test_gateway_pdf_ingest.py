from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("MK_FEATURE_FLAGS__FLAGS__EVALUATION", "false")

from typing import TYPE_CHECKING

from Medical_KG_rev.config.settings import get_settings
from Medical_KG_rev.gateway.services import GatewayService
from Medical_KG_rev.gateway.sse.manager import EventStreamManager
from Medical_KG_rev.orchestration.dagster.configuration import PipelineConfigLoader
from Medical_KG_rev.orchestration.dagster.runtime import DagsterRunResult
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.stages.contracts import PipelineState

if TYPE_CHECKING:
    pass


class _StubOrchestrator:
    """Minimal orchestrator stub exposing pipeline metadata for tests."""

    def __init__(self, loader: PipelineConfigLoader, pipelines: list[str]) -> None:
        self.pipeline_loader = loader
        self._pipelines = list(pipelines)
        self.submissions: list[dict[str, object]] = []

    def available_pipelines(self) -> list[str]:
        return list(self._pipelines)

    def submit(
        self, *, pipeline: str, context, adapter_request, payload
    ) -> DagsterRunResult:
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
    orchestrator = _StubOrchestrator(loader, ["auto", "pdf-two-phase", "pmc-fulltext"])
    service = GatewayService(
        events=EventStreamManager(),
        orchestrator=orchestrator,  # type: ignore[arg-type]
        ledger=JobLedger(),
    )
    service._orchestrator_stub = orchestrator  # type: ignore[attr-defined]
    return service


def test_resolve_pipeline_prefers_pmc_fulltext_for_pmc(gateway_service: GatewayService) -> None:
    service = gateway_service
    pipeline = service._resolve_pipeline("pmc", {"id": "doc-123"})
    assert pipeline == "pmc-fulltext"


def test_resolve_pipeline_prefers_pdf_for_openalex(gateway_service: GatewayService) -> None:
    service = gateway_service
    pipeline = service._resolve_pipeline("openalex", {"id": "doc-openalex"})
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
    from Medical_KG_rev.gateway.models import IngestionRequest

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
    assert status.metadata["pipeline"] == "pmc-fulltext"
    assert status.metadata["duplicate"] is False
    assert status.metadata["state"]["context"]["pipeline_name"] == "pmc-fulltext"
    assert orchestrator.submissions, "expected Dagster submission"
    submission = orchestrator.submissions[0]
    assert submission["pipeline"] == "pmc-fulltext"

    entry = service.ledger.get(status.job_id)
    assert entry is not None
    assert entry.metadata["pipeline_version"] == "2025-01-01"
    assert entry.metadata["payload"]["dataset"] == "pmc"
    assert entry.metadata["adapter_request"]["parameters"]["dataset"] == "pmc"


def test_submit_openalex_job_injects_settings(monkeypatch, gateway_service: GatewayService) -> None:
    from Medical_KG_rev.gateway.models import IngestionRequest

    service = gateway_service
    orchestrator: _StubOrchestrator = service._orchestrator_stub  # type: ignore[attr-defined]
    orchestrator.submissions.clear()

    get_settings.cache_clear()
    monkeypatch.setenv("MK_OPENALEX__CONTACT_EMAIL", "openalex@example.com")
    monkeypatch.setenv("MK_OPENALEX__USER_AGENT", "MedicalKG-Test/1.0")
    monkeypatch.setenv("MK_OPENALEX__MAX_RESULTS", "7")
    monkeypatch.setenv("MK_OPENALEX__REQUESTS_PER_SECOND", "4.5")
    monkeypatch.setenv("MK_OPENALEX__TIMEOUT_SECONDS", "45.0")

    request = IngestionRequest(
        tenant_id="tenant-openalex",
        items=[{"id": "oa-001", "query": "lung cancer"}],
        metadata={"source": "openalex"},
    )

    try:
        status = service._submit_dagster_job(
            dataset="openalex",
            request=request,
            item=request.items[0],
            metadata={"source": "openalex"},
        )

        assert status.metadata["pipeline"] == "pdf-two-phase"
        submission = orchestrator.submissions[0]
        adapter_request = submission["adapter_request"]
        params = adapter_request.parameters
        assert params["contact_email"] == "openalex@example.com"
        assert params["user_agent"] == "MedicalKG-Test/1.0"
        assert params["max_results"] == 7
        assert params["requests_per_second"] == 4.5
        assert params["timeout_seconds"] == 45.0
    finally:
        get_settings.cache_clear()
