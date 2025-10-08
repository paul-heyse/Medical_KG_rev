from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition
from Medical_KG_rev.orchestration.dagster.stage_registry import StageRegistration
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.stage_plugins import (
    GateConditionError,
    register_download_stage,
    register_gate_stage,
)
from Medical_KG_rev.orchestration.stages.contracts import StageContext


def _definition(stage_type: str, name: str, config: dict[str, Any]) -> StageDefinition:
    payload = {"name": name, "type": stage_type, "policy": "default", "config": config}
    return StageDefinition.model_validate(payload)


def _ledger_with_entry(job_id: str, doc_key: str, tenant: str, pipeline: str) -> JobLedger:
    ledger = JobLedger()
    ledger.create(job_id=job_id, doc_key=doc_key, tenant_id=tenant, pipeline=pipeline)
    return ledger


def test_download_stage_success(tmp_path) -> None:
    registration: StageRegistration = register_download_stage()
    definition = _definition(
        "download",
        "download",
        {
            "url_extractors": [{"source": "payload", "path": "best_oa_location.pdf_url"}],
            "storage": {"base_path": str(tmp_path)},
            "http": {"timeout_seconds": 1, "max_attempts": 1},
        },
    )
    stage = registration.builder(definition)

    class DummyResponse:
        def __init__(self) -> None:
            self.content = b"%PDF-1.4\n"
            self.headers = {"content-type": "application/pdf"}

        def raise_for_status(self) -> None:  # pragma: no cover - nothing to raise
            return None

    class DummyClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def request(self, method: str, url: str, **_: Any) -> DummyResponse:
            self.calls.append((method, url))
            return DummyResponse()

    stage.http_client = DummyClient()  # type: ignore[assignment]

    ctx = StageContext(
        tenant_id="tenant-a",
        job_id="job-1",
        doc_id="doc-1",
        correlation_id="corr-1",
        metadata={"title": "Sample"},
        pipeline_name="pdf-two-phase",
        pipeline_version="2025-03-15",
    )
    ledger = _ledger_with_entry("job-1", "doc-1", "tenant-a", "pdf-two-phase")
    stage.bind_runtime(job_ledger=ledger)

    upstream = [{"best_oa_location": {"pdf_url": "https://example.org/test.pdf"}}]
    result = stage.execute(ctx, upstream)

    assert result["status"] == "success"
    assert result["files"][0]["url"].endswith("test.pdf")
    saved_files = list(tmp_path.glob("*.pdf"))
    assert saved_files, "expected a downloaded PDF file"

    entry = ledger.get("job-1")
    assert entry is not None
    assert entry.pdf_downloaded is True
    assert entry.metadata.get("pdf_url") == "https://example.org/test.pdf"


def test_download_stage_failure_updates_ledger(tmp_path) -> None:
    registration: StageRegistration = register_download_stage()
    definition = _definition(
        "download",
        "download",
        {
            "url_extractors": [{"source": "payload", "path": "best_oa_location.url"}],
            "storage": {"base_path": str(tmp_path)},
            "http": {"timeout_seconds": 0.1, "max_attempts": 1},
        },
    )
    stage = registration.builder(definition)

    class ErrorClient:
        def request(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("network down")

    stage.http_client = ErrorClient()  # type: ignore[assignment]

    ctx = StageContext(
        tenant_id="tenant-b",
        job_id="job-2",
        doc_id="doc-2",
        correlation_id="corr-2",
        metadata={},
        pipeline_name="pdf-two-phase",
        pipeline_version="2025-03-15",
    )
    ledger = _ledger_with_entry("job-2", "doc-2", "tenant-b", "pdf-two-phase")
    stage.bind_runtime(job_ledger=ledger)

    upstream = [{"best_oa_location": {"url": "https://example.org/fail.pdf"}}]
    result = stage.execute(ctx, upstream)

    assert result["status"] == "failed"
    entry = ledger.get("job-2")
    assert entry is not None
    assert entry.pdf_downloaded is False
    assert entry.metadata.get("pdf_download_error") == "network down"


def test_gate_stage_passes_when_ledger_ready(monkeypatch) -> None:
    registration: StageRegistration = register_gate_stage()
    definition = _definition(
        "gate",
        "gate_pdf_ir_ready",
        {
            "gate": "pdf_ir_ready",
            "field": "pdf_ir_ready",
            "equals": True,
            "timeout_seconds": 2,
            "poll_interval_seconds": 0.01,
        },
    )
    stage = registration.builder(definition)
    ledger = _ledger_with_entry("job-3", "doc-3", "tenant-c", "pdf-two-phase")
    ledger.set_pdf_ir_ready("job-3", True)
    stage.bind_runtime(job_ledger=ledger)

    ctx = StageContext(
        tenant_id="tenant-c",
        job_id="job-3",
        doc_id="doc-3",
        correlation_id="corr-3",
        metadata={},
        pipeline_name="pdf-two-phase",
        pipeline_version="2025-03-15",
    )
    result = stage.execute(ctx, {})
    assert result["gate"] == "pdf_ir_ready"
    assert result["value"] is True


def test_gate_stage_times_out(monkeypatch) -> None:
    registration: StageRegistration = register_gate_stage()
    definition = _definition(
        "gate",
        "gate_pdf_ir_ready",
        {
            "gate": "pdf_ir_ready",
            "field": "pdf_ir_ready",
            "equals": True,
            "timeout_seconds": 0.05,
            "poll_interval_seconds": 0.01,
        },
    )
    stage = registration.builder(definition)
    ledger = _ledger_with_entry("job-4", "doc-4", "tenant-d", "pdf-two-phase")
    stage.bind_runtime(job_ledger=ledger)

    ctx = StageContext(
        tenant_id="tenant-d",
        job_id="job-4",
        doc_id="doc-4",
        correlation_id="corr-4",
        metadata={},
        pipeline_name="pdf-two-phase",
        pipeline_version="2025-03-15",
    )

    with pytest.raises(GateConditionError) as exc:
        stage.execute(ctx, {})
    assert "timed out" in str(exc.value)
