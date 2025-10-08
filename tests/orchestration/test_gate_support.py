import time

import pytest

dagster = pytest.importorskip("dagster")
from dagster import ResourceDefinition, build_sensor_context

from Medical_KG_rev.adapters.plugins.models import AdapterDomain
from Medical_KG_rev.orchestration.dagster.configuration import (
    GateCondition,
    GateConditionClause,
    GateDefinition,
    GateOperator,
    PipelineTopologyConfig,
    StageDefinition,
)
from Medical_KG_rev.orchestration.dagster.gates import GateConditionError, GateStage
from Medical_KG_rev.orchestration.dagster.runtime import (
    StageFactory,
    _build_pipeline_job,
    pdf_ir_ready_sensor,
)
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.stages.contracts import StageContext


class DummyResilience:
    def apply(self, name, stage, func, hooks=None):  # pragma: no cover - exercised via tests
        def _wrapped(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive
                if hooks and hooks.on_failure:
                    hooks.on_failure(exc, 1)
                raise
            if hooks and hooks.on_success:
                hooks.on_success(1, 0.0)
            return result

        return _wrapped


class DummyEmitter:
    def emit_started(self, *_, **__):  # pragma: no cover - noop sink
        return None

    emit_retrying = emit_failed = emit_completed = emit_started


class PreStage:
    def execute(self, ctx: StageContext, *_):
        payload = dict(ctx.metadata)
        payload["pre"] = payload.get("pre", 0) + 1
        return payload


class PostStage:
    def __init__(self, marker: str = "post") -> None:
        self.marker = marker

    def execute(self, ctx: StageContext, *_):
        payload = dict(ctx.metadata)
        payload[self.marker] = payload.get(self.marker, 0) + 1
        return payload


@pytest.fixture()
def ledger() -> JobLedger:
    return JobLedger()


def _build_gate_definition(timeout: int = 1) -> GateDefinition:
    return GateDefinition(
        name="pdf_ready",
        resume_stage="post",
        timeout_seconds=timeout,
        poll_interval_seconds=0.5,
        condition=GateCondition(
            mode="all",
            clauses=[
                GateConditionClause(
                    field="pdf_ir_ready",
                    operator=GateOperator.EQUALS,
                    value=True,
                )
            ],
        ),
    )


def test_gate_condition_evaluation_success(ledger: JobLedger) -> None:
    entry = ledger.create(job_id="job-1", doc_key="doc", tenant_id="t", pipeline="p")
    ledger.set_pdf_ir_ready(entry.job_id, True)
    gate = _build_gate_definition()
    stage_def = StageDefinition(name="gate", stage_type="gate", gate=gate.name)
    stage = GateStage(stage_def, gate)
    ctx = StageContext(tenant_id="t", job_id=entry.job_id)

    result = stage.execute(ctx, {}, ledger=ledger)

    assert result.satisfied is True
    assert ledger.get(entry.job_id).phase == "phase-2"


def test_gate_condition_timeout(ledger: JobLedger) -> None:
    entry = ledger.create(job_id="job-2", doc_key="doc", tenant_id="t", pipeline="p")
    gate = _build_gate_definition(timeout=1)
    stage_def = StageDefinition(name="gate", stage_type="gate", gate=gate.name)
    stage = GateStage(stage_def, gate)
    ctx = StageContext(tenant_id="t", job_id=entry.job_id)

    start = time.perf_counter()
    with pytest.raises(GateConditionError) as excinfo:
        stage.execute(ctx, {}, ledger=ledger)
    duration = time.perf_counter() - start

    assert excinfo.value.status == "timeout"
    assert duration >= 1


def test_pipeline_validation_requires_gate_reference() -> None:
    stages = [
        StageDefinition(name="gate", stage_type="gate", gate="g1"),
        StageDefinition(name="post", stage_type="post", depends_on=["gate"]),
    ]
    gates = [
        GateDefinition(
            name="g1",
            resume_stage="post",
            condition=GateCondition(
                clauses=[GateConditionClause(field="pdf_ir_ready", operator=GateOperator.EQUALS, value=True)]
            ),
        )
    ]
    topology = PipelineTopologyConfig(
        name="pipeline",
        version="2025-01-01",
        stages=stages,
        gates=gates,
    )

    assert topology.gates[0].name == "g1"
    assert stages[0].execution_phase == "phase-1"
    assert stages[1].execution_phase == "phase-2"


def test_pipeline_validation_rejects_missing_resume_dependency() -> None:
    stages = [
        StageDefinition(name="gate", stage_type="gate", gate="g1"),
        StageDefinition(name="post", stage_type="post"),
    ]
    gates = [
        GateDefinition(
            name="g1",
            resume_stage="post",
            condition=GateCondition(
                clauses=[GateConditionClause(field="pdf_ir_ready", operator=GateOperator.EQUALS, value=True)]
            ),
        )
    ]
    with pytest.raises(ValueError):
        PipelineTopologyConfig(
            name="pipeline",
            version="2025-01-01",
            stages=stages,
            gates=gates,
        )


def _build_two_phase_job(ledger: JobLedger):
    gate = _build_gate_definition(timeout=1)
    stages = [
        StageDefinition(name="pre", stage_type="pre"),
        StageDefinition(name="ready_gate", stage_type="gate", gate=gate.name, depends_on=["pre"]),
        StageDefinition(name="post", stage_type="post", depends_on=["ready_gate"]),
    ]
    topology = PipelineTopologyConfig(
        name="two-phase",
        version="2025-01-01",
        stages=stages,
        gates=[gate],
    )

    registry = {
        "pre": lambda _top, _def: PreStage(),
        "post": lambda _top, _def: PostStage(),
        "gate": lambda top, definition: GateStage(
            definition,
            next(item for item in top.gates if item.name == definition.gate),
        ),
    }
    stage_factory = StageFactory(registry)
    resources = {
        "stage_factory": stage_factory,
        "resilience_policies": DummyResilience(),
        "job_ledger": ledger,
        "event_emitter": DummyEmitter(),
    }
    resource_defs = {
        key: ResourceDefinition.hardcoded_resource(value) for key, value in resources.items()
    }
    built = _build_pipeline_job(topology, resource_defs=resource_defs)
    return topology, built, resources


def test_two_phase_execution_flow(ledger: JobLedger) -> None:
    topology, job, resources = _build_two_phase_job(ledger)
    ledger.create(job_id="job-two-phase", doc_key="doc", tenant_id="tenant", pipeline=topology.name)

    run_config = {
        "ops": {
            "bootstrap": {
                "config": {
                    "context": {
                        "tenant_id": "tenant",
                        "job_id": "job-two-phase",
                        "doc_id": "doc",
                        "correlation_id": "corr",
                        "metadata": {},
                        "pipeline_name": topology.name,
                        "pipeline_version": topology.version,
                        "phase": "phase-1",
                    },
                    "adapter_request": {
                        "tenant_id": "tenant",
                        "correlation_id": "corr",
                        "domain": AdapterDomain.BIOMEDICAL.value,
                        "parameters": {},
                    },
                    "payload": {},
                }
            }
        }
    }
    result = job.job_definition.execute_in_process(
        run_config=run_config,
        resources=resources,
    )

    state = result.output_for_node(job.final_node)
    assert state["phase_index"] == 1
    assert state["phase_ready"] is False
    assert "post" not in state.get("results", {})

    ledger.set_pdf_ir_ready("job-two-phase", True)

    resume_config = {
        "ops": {
            "bootstrap": {
                "config": {
                    "context": {
                        "tenant_id": "tenant",
                        "job_id": "job-two-phase",
                        "doc_id": "doc",
                        "correlation_id": "corr",
                        "metadata": {},
                        "pipeline_name": topology.name,
                        "pipeline_version": topology.version,
                        "phase": "phase-2",
                        "phase_ready": True,
                    },
                    "adapter_request": {
                        "tenant_id": "tenant",
                        "correlation_id": "corr",
                        "domain": AdapterDomain.BIOMEDICAL.value,
                        "parameters": {},
                    },
                    "payload": {},
                }
            }
        }
    }
    resume_result = job.job_definition.execute_in_process(
        run_config=resume_config,
        resources=resources,
    )
    resume_state = resume_result.output_for_node(job.final_node)

    assert resume_state["phase_index"] == 2
    assert resume_state["phase_ready"] is True
    assert "post" in resume_state.get("results", {})


def test_pdf_sensor_emits_resume_request(ledger: JobLedger) -> None:
    entry = ledger.create(job_id="job-sensor", doc_key="doc", tenant_id="tenant", pipeline="pdf-two-phase")
    ledger.update_metadata(
        entry.job_id,
        {
            "pipeline_version": "2025-01-01",
            "adapter_request": {
                "tenant_id": "tenant",
                "correlation_id": "corr",
                "domain": AdapterDomain.BIOMEDICAL.value,
                "parameters": {},
            },
            "payload": {},
            "resume_stage": "chunk",
            "phase_index": 1,
            "phase_ready": False,
        },
    )
    ledger.set_pdf_ir_ready(entry.job_id, True)
    ledger.set_phase(entry.job_id, "phase-1")
    ledger.mark_processing(entry.job_id, stage="gate_pdf_ir_ready")

    context = build_sensor_context(resources={"job_ledger": ledger})
    requests = list(pdf_ir_ready_sensor(context))

    assert requests
    run_request = requests[0]
    assert run_request.tags["medical_kg.resume_stage"] == "chunk"
    assert run_request.tags["medical_kg.resume_phase"] == "phase-2"
    phase_value = run_request.run_config["ops"]["bootstrap"]["config"]["context"]["phase"]
    assert phase_value == "phase-2"
