from __future__ import annotations

import pytest

from Medical_KG_rev.orchestration.dagster.configuration import (
    GateCondition,
    GateConditionClause,
    GateConditionOperator,
    GateDefinition,
    GateRetryConfig,
)
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.stage_plugins import GateConditionError, GateStage
from Medical_KG_rev.orchestration.stages.contracts import StageContext


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def _build_context(job_id: str) -> StageContext:
    return StageContext(
        tenant_id="tenant-1",
        job_id=job_id,
        doc_id="doc-1",
        correlation_id="corr-1",
        metadata={},
        pipeline_name="pdf-two-phase",
        pipeline_version="2025-01-01",
    )


def _create_gate_definition(timeout: float | None = 900.0) -> GateDefinition:
    return GateDefinition(
        name="pdf_ir_ready",
        stage="gate_pdf_ir_ready",
        resume_stage="chunk",
        skip_download_on_resume=True,
        retry=GateRetryConfig(max_attempts=1, delay_seconds=0.0),
        condition=GateCondition(
            match="all",
            clauses=[
                GateConditionClause(
                    field="pdf_ir_ready",
                    operator=GateConditionOperator.EQUALS,
                    value=True,
                )
            ],
            timeout_seconds=timeout,
            poll_interval_seconds=0.01,
        ),
    )


def test_gate_stage_succeeds_when_condition_met() -> None:
    ledger = JobLedger()
    entry = ledger.create(job_id="job-1", doc_key="doc-1", tenant_id="tenant-1", pipeline="pdf-two-phase")
    ledger.mark_processing(entry.job_id, stage="download")
    ledger.set_pdf_downloaded(entry.job_id)
    ledger.set_pdf_ir_ready(entry.job_id)

    gate = GateStage(definition=_create_gate_definition())
    context = _build_context(entry.job_id)
    result = gate.execute(context, {"_phase": "post-gate", "pipeline": "pdf-two-phase"}, ledger)

    assert result.resume_stage == "chunk"
    assert result.attempts == 1
    metadata = ledger.get(entry.job_id)
    assert metadata is not None
    gate_state = metadata.metadata.get("gate", {}).get("pdf_ir_ready")  # type: ignore[arg-type]
    assert gate_state["status"] == "passed"
    assert gate_state["resume_stage"] == "chunk"


def test_gate_stage_fails_when_condition_not_met() -> None:
    ledger = JobLedger()
    entry = ledger.create(job_id="job-2", doc_key="doc-2", tenant_id="tenant-1", pipeline="pdf-two-phase")
    ledger.mark_processing(entry.job_id, stage="download")

    gate = GateStage(definition=_create_gate_definition(timeout=None))
    context = _build_context(entry.job_id)

    with pytest.raises(GateConditionError):
        gate.execute(context, {"_phase": "gate", "pipeline": "pdf-two-phase"}, ledger)

    gate_meta = ledger.get(entry.job_id).metadata.get("gate", {})  # type: ignore[union-attr]
    assert gate_meta["pdf_ir_ready"]["status"] == "failed"


def test_gate_stage_times_out_after_deadline() -> None:
    clock = _Clock()
    ledger = JobLedger()
    entry = ledger.create(job_id="job-3", doc_key="doc-3", tenant_id="tenant-1", pipeline="pdf-two-phase")
    ledger.mark_processing(entry.job_id, stage="download")

    definition = _create_gate_definition(timeout=0.05)
    gate = GateStage(definition=definition, sleep=clock.sleep, monotonic=clock.now)
    context = _build_context(entry.job_id)

    with pytest.raises(GateConditionError):
        gate.execute(context, {"_phase": "gate", "pipeline": "pdf-two-phase"}, ledger)

    gate_meta = ledger.get(entry.job_id).metadata.get("gate", {})  # type: ignore[union-attr]
    assert gate_meta["pdf_ir_ready"]["status"] == "timeout"
