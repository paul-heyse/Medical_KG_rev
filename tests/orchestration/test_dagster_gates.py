from __future__ import annotations

import json
from typing import Any

import pytest
from dagster import ResourceDefinition

from Medical_KG_rev.orchestration.dagster.configuration import (
    GateCondition,
    GateConditionClause,
    GateConditionOperator,
    GateDefinition,
    GateRetryPolicy,
    PipelineTopologyConfig,
)
from Medical_KG_rev.orchestration.dagster.runtime import (
    GateConditionError,
    GateConditionEvaluator,
    GateStage,
    StageFactory,
    _build_pipeline_job,
)
from Medical_KG_rev.orchestration.ledger import JobLedger, JobLedgerEntry
from Medical_KG_rev.orchestration.stages.contracts import StageContext


class _StubStage:
    def __init__(self, result: Any = None) -> None:
        self._result = result

    def execute(self, ctx: StageContext, *_: Any, **__: Any) -> Any:
        return self._result


class _StubPolicies:
    def apply(self, _name: str, _stage: str, func, *, hooks=None):  # type: ignore[override]
        return func


class _StubEmitter:
    def emit_retrying(self, *args, **kwargs):  # pragma: no cover - instrumentation stub
        return None

    def emit_started(self, *args, **kwargs):  # pragma: no cover
        return None

    def emit_failed(self, *args, **kwargs):  # pragma: no cover
        return None

    def emit_completed(self, *args, **kwargs):  # pragma: no cover
        return None


@pytest.fixture()
def gate_definition() -> GateDefinition:
    return GateDefinition(
        name="pdf_ir_ready",
        resume_stage="chunk",
        timeout_seconds=30,
        retry=GateRetryPolicy(max_attempts=2, backoff_seconds=0.0),
        condition=GateCondition(
            logic="all",
            poll_interval_seconds=0.5,
            timeout_seconds=5,
            clauses=[
                GateConditionClause(
                    field="pdf_ir_ready",
                    operator=GateConditionOperator.EQUALS,
                    value=True,
                ),
                GateConditionClause(
                    field="metadata.gate.pdf_ir_ready.status",
                    operator=GateConditionOperator.EQUALS,
                    value="passed",
                ),
            ],
        ),
    )


def test_gate_condition_evaluator_supports_changed_operator(gate_definition: GateDefinition) -> None:
    changed_definition = gate_definition.model_copy(
        update={
            "condition": GateCondition(
                logic="all",
                poll_interval_seconds=0.5,
                timeout_seconds=5,
                clauses=[
                    GateConditionClause(
                        field="pdf_ir_ready",
                        operator=GateConditionOperator.EQUALS,
                        value=True,
                    ),
                    GateConditionClause(
                        field="metadata.gate.pdf_ir_ready.status",
                        operator=GateConditionOperator.CHANGED,
                    ),
                ],
            )
        }
    )
    evaluator = GateConditionEvaluator(changed_definition)
    entry = JobLedgerEntry(job_id="job-1", doc_key="doc-1", tenant_id="tenant")
    entry.metadata["gate"] = {"pdf_ir_ready": {"status": "passed"}}
    entry.pdf_ir_ready = True

    satisfied, observed = evaluator.evaluate(
        entry,
        previous_observed={"pdf_ir_ready": False, "metadata.gate.pdf_ir_ready.status": "waiting"},
    )

    assert satisfied is True
    assert observed["pdf_ir_ready"] is True
    assert observed["metadata.gate.pdf_ir_ready.status"] == "passed"


def test_gate_stage_returns_success_when_conditions_met(gate_definition: GateDefinition) -> None:
    ledger = JobLedger()
    entry = ledger.create(job_id="job-1", doc_key="doc-1", tenant_id="tenant")
    ledger.set_pdf_ir_ready(entry.job_id, True)
    ledger.update_metadata(
        entry.job_id,
        {
            "gate": {"pdf_ir_ready": {"status": "passed"}},
            "gate.pdf_ir_ready.status": "passed",
        },
    )

    stage = GateStage(
        gate_definition,
        ledger=ledger,
        evaluator=GateConditionEvaluator(gate_definition),
        sleep=lambda seconds: None,
    )
    context = StageContext(tenant_id="tenant", job_id=entry.job_id)
    state: dict[str, Any] = {"job_id": entry.job_id}

    result = stage.execute(context, state)

    assert result.satisfied is True
    assert result.resume_stage == "chunk"
    assert result.observed["pdf_ir_ready"] is True


def test_gate_stage_times_out(monkeypatch: pytest.MonkeyPatch, gate_definition: GateDefinition) -> None:
    ledger = JobLedger()
    entry = ledger.create(job_id="job-2", doc_key="doc-2", tenant_id="tenant")
    clock = {"now": 0.0}

    def advance(seconds: float) -> None:
        clock["now"] += seconds

    monkeypatch.setattr(
        "Medical_KG_rev.orchestration.dagster.runtime.time.monotonic",
        lambda: clock["now"],
    )
    monkeypatch.setattr(
        "Medical_KG_rev.orchestration.dagster.runtime.time.perf_counter",
        lambda: clock["now"],
    )

    timeout_definition = gate_definition.model_copy(
        update={
            "timeout_seconds": 1,
            "condition": gate_definition.condition.model_copy(
                update={"timeout_seconds": 1, "poll_interval_seconds": 0.5}
            ),
        }
    )
    stage = GateStage(
        timeout_definition,
        ledger=ledger,
        evaluator=GateConditionEvaluator(timeout_definition),
        sleep=advance,
    )
    context = StageContext(tenant_id="tenant", job_id=entry.job_id)

    with pytest.raises(GateConditionError) as excinfo:
        stage.execute(context, {"job_id": entry.job_id})

    error = excinfo.value
    assert isinstance(error, GateConditionError)
    assert error.timeout is True
    assert error.gate == "pdf_ir_ready"


def test_build_pipeline_job_records_phase_map(gate_definition: GateDefinition) -> None:
    topology_payload = {
        "name": "gated-pipeline",
        "version": "2025-01-01",
        "stages": [
            {"name": "ingest", "type": "ingest"},
            {
                "name": "gate_pdf_ir_ready",
                "type": "gate",
                "depends_on": ["ingest"],
                "gate": "pdf_ir_ready",
            },
            {"name": "chunk", "type": "chunk", "depends_on": ["gate_pdf_ir_ready"]},
        ],
        "gates": [json.loads(gate_definition.model_dump_json())],
    }
    topology = PipelineTopologyConfig.model_validate(topology_payload)

    stage_factory = StageFactory(
        {
            "ingest": lambda _: _StubStage([{"payload": "ok"}]),
            "chunk": lambda _: _StubStage([]),
        }
    )
    resource_defs = {
        "stage_factory": ResourceDefinition.hardcoded_resource(stage_factory),
        "resilience_policies": ResourceDefinition.hardcoded_resource(_StubPolicies()),
        "job_ledger": ResourceDefinition.hardcoded_resource(JobLedger()),
        "event_emitter": ResourceDefinition.hardcoded_resource(_StubEmitter()),
    }

    built = _build_pipeline_job(topology, resource_defs=resource_defs)

    assert built.phase_map["ingest"] == 1
    assert built.phase_map["gate_pdf_ir_ready"] == 1
    assert built.phase_map["chunk"] == 2
