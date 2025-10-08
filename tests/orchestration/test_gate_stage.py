import itertools

import pytest
from pydantic import ValidationError

from Medical_KG_rev.orchestration.dagster.configuration import (
    GateCondition,
    GateConditionOperator,
    GateDefinition,
    GatePredicate,
    GateRetryConfig,
)
from Medical_KG_rev.orchestration.dagster.gates import (
    GateConditionError,
    GateConditionEvaluator,
    GateTimeoutError,
)
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.stage_plugins import GateStage
from Medical_KG_rev.orchestration.stages.contracts import StageContext


def _build_gate_definition(**overrides) -> GateDefinition:
    base = {
        "name": "pdf_ir_ready",
        "resume_stage": "chunk",
        "timeout_seconds": 1,
        "poll_interval_seconds": 0.01,
        "resume_phase": "post-gate",
        "conditions": [
            GateCondition(
                all=[
                    GatePredicate(
                        field="pdf_downloaded",
                        operator=GateConditionOperator.EQUALS,
                        value=True,
                    )
                ]
            ),
            GateCondition(
                any=[
                    GatePredicate(
                        field="pdf_ir_ready",
                        operator=GateConditionOperator.EQUALS,
                        value=True,
                    )
                ]
            ),
        ],
    }
    base.update(overrides)
    return GateDefinition.model_validate(base)


def _build_stage(definition: GateDefinition) -> GateStage:
    retry = definition.retry
    return GateStage(
        name="gate_pdf_ir_ready",
        definition=definition,
        evaluator=GateConditionEvaluator(definition),
        timeout_seconds=definition.timeout_seconds,
        poll_interval=definition.poll_interval_seconds,
        max_attempts=retry.max_attempts if retry else None,
        retry_backoff=retry.backoff_seconds if retry else definition.poll_interval_seconds,
    )


def test_gate_condition_evaluator_all_any_passes() -> None:
    gate_def = _build_gate_definition()
    evaluator = GateConditionEvaluator(gate_def)
    entry_state = {
        "pdf_downloaded": True,
        "pdf_ir_ready": True,
        "metadata": {"pdf_ir_ready": True},
    }
    ledger = JobLedger()
    entry = ledger.create(
        job_id="job-1",
        doc_key="doc-1",
        tenant_id="tenant-1",
        pipeline="pdf-two-phase",
        metadata=entry_state["metadata"],
    )
    ledger.set_pdf_downloaded(entry.job_id)
    ledger.set_pdf_ir_ready(entry.job_id)
    gate_state: dict[str, object] = {}
    success, details = evaluator.evaluate(ledger.get(entry.job_id), entry_state, gate_state)
    assert success is True
    assert details["clauses"]  # non-empty diagnostic payload
    assert gate_state["last_values"]["pdf_downloaded"] is True


def test_gate_condition_evaluator_changed_detects_mutations() -> None:
    gate_def = _build_gate_definition(
        conditions=[
            GateCondition(
                all=[
                    GatePredicate(
                        field="metadata.version",
                        operator=GateConditionOperator.CHANGED,
                        value=True,
                    )
                ]
            )
        ]
    )
    evaluator = GateConditionEvaluator(gate_def)
    ledger = JobLedger()
    entry = ledger.create(
        job_id="job-1",
        doc_key="doc-1",
        tenant_id="tenant-1",
        pipeline="pdf-two-phase",
        metadata={"version": "v1"},
    )
    gate_state: dict[str, object] = {}
    success_first, _ = evaluator.evaluate(ledger.get(entry.job_id), {}, gate_state)
    assert success_first is True
    # Second evaluation with unchanged version should fail
    success_second, details = evaluator.evaluate(ledger.get(entry.job_id), {}, gate_state)
    assert success_second is False
    assert not details["clauses"][0]["passed"]


def test_gate_stage_evaluate_success(monkeypatch: pytest.MonkeyPatch) -> None:
    gate_def = _build_gate_definition()
    stage = _build_stage(gate_def)
    ledger = JobLedger()
    entry = ledger.create(
        job_id="job-success",
        doc_key="doc-success",
        tenant_id="tenant-1",
        pipeline="pdf-two-phase",
        metadata={},
    )
    ledger.set_pdf_downloaded(entry.job_id)

    call_count = {"value": 0}
    original_get = ledger.get

    def fake_get(job_id: str):
        call_count["value"] += 1
        if call_count["value"] == 2:
            ledger.set_pdf_ir_ready(job_id)
        return original_get(job_id)

    monkeypatch.setattr(ledger, "get", fake_get)
    monkeypatch.setattr("Medical_KG_rev.orchestration.stage_plugins.time.sleep", lambda _: None)

    state: dict[str, object] = {"context": StageContext(tenant_id="tenant-1", job_id=entry.job_id)}
    result = stage.evaluate(StageContext(tenant_id="tenant-1", job_id=entry.job_id), ledger, state)
    assert result.satisfied is True
    assert result.attempts == 2
    metadata = ledger.get(entry.job_id).metadata
    assert metadata["gate.pdf_ir_ready.status"] == "passed"
    assert state["gates"]["pdf_ir_ready"]["status"] == "passed"


def test_gate_stage_evaluate_failure_raises() -> None:
    gate_def = _build_gate_definition(
        retry=GateRetryConfig(max_attempts=1, backoff_seconds=0.01)
    )
    stage = _build_stage(gate_def)
    ledger = JobLedger()
    entry = ledger.create(
        job_id="job-fail",
        doc_key="doc-fail",
        tenant_id="tenant-1",
        pipeline="pdf-two-phase",
        metadata={},
    )
    ledger.set_pdf_downloaded(entry.job_id)
    state: dict[str, object] = {"context": StageContext(tenant_id="tenant-1", job_id=entry.job_id)}

    with pytest.raises(GateConditionError) as exc:
        stage.evaluate(StageContext(tenant_id="tenant-1", job_id=entry.job_id), ledger, state)

    error = exc.value
    metadata = ledger.get(entry.job_id).metadata
    assert metadata["gate.pdf_ir_ready.status"] == "failed"
    assert state["gates"]["pdf_ir_ready"]["status"] == "failed"
    assert "Gate" in str(error)


def test_gate_stage_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    gate_def = _build_gate_definition(timeout_seconds=1, poll_interval_seconds=0.1)
    stage = _build_stage(gate_def)
    ledger = JobLedger()
    entry = ledger.create(
        job_id="job-timeout",
        doc_key="doc-timeout",
        tenant_id="tenant-1",
        pipeline="pdf-two-phase",
        metadata={},
    )
    ledger.set_pdf_downloaded(entry.job_id)

    counter = itertools.count()

    def fake_perf_counter() -> float:
        return next(counter) * 0.6

    monkeypatch.setattr(
        "Medical_KG_rev.orchestration.stage_plugins.time.perf_counter",
        fake_perf_counter,
    )
    monkeypatch.setattr(
        "Medical_KG_rev.orchestration.dagster.gates.time.perf_counter",
        fake_perf_counter,
    )
    monkeypatch.setattr(
        "Medical_KG_rev.orchestration.stage_plugins.time.sleep", lambda _: None
    )

    state: dict[str, object] = {"context": StageContext(tenant_id="tenant-1", job_id=entry.job_id)}

    with pytest.raises(GateTimeoutError):
        stage.evaluate(StageContext(tenant_id="tenant-1", job_id=entry.job_id), ledger, state)


def test_gate_definition_requires_conditions() -> None:
    with pytest.raises(ValidationError):
        GateDefinition.model_validate(
            {
                "name": "invalid_gate",
                "resume_stage": "chunk",
                "conditions": [],
            }
        )
