"""Dagster runtime orchestration primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4

from dagster import (
    Definitions,
    ExecuteInProcessResult,
    Field,
    In,
    Out,
    ResourceDefinition,
    RunRequest,
    SensorEvaluationContext,
    SkipReason,
    graph,
    op,
    sensor,
)

from Medical_KG_rev.adapters.plugins.bootstrap import get_plugin_manager
from Medical_KG_rev.adapters.plugins.manager import AdapterPluginManager
from Medical_KG_rev.adapters.plugins.models import AdapterRequest
from Medical_KG_rev.orchestration.dagster.configuration import (
    PipelineConfigLoader,
    PipelinePhasePlan,
    PipelineTopologyConfig,
    StageExecutionHooks,
    ResiliencePolicyLoader,
    StageDefinition,
)
from Medical_KG_rev.orchestration.dagster.gates import (
    GateConditionError,
    GateEvaluationResult,
    GateTimeoutError,
)
from Medical_KG_rev.orchestration.dagster.stage_registry import (
    StageMetadata,
    StageRegistry,
    StageRegistryError,
)
from Medical_KG_rev.orchestration.dagster.stages import (
    HaystackPipelineResource,
    build_default_stage_factory,
    create_default_pipeline_resource,
)
from Medical_KG_rev.orchestration.events import StageEventEmitter
from Medical_KG_rev.observability.metrics import (
    record_gate_evaluation,
    record_phase_transition,
)
from Medical_KG_rev.orchestration.kafka import KafkaClient
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.state_manager import LedgerStateManager
from Medical_KG_rev.orchestration.openlineage import OpenLineageEmitter
from Medical_KG_rev.orchestration.stages.contracts import StageContext
from Medical_KG_rev.observability.metrics import (
    observe_gate_duration,
    record_gate_evaluation,
    record_gate_timeout,
    record_phase_transition,
)
from Medical_KG_rev.utils.logging import get_logger

logger = get_logger(__name__)


class StageResolutionError(RuntimeError):
    """Raised when a stage cannot be resolved from the registry."""


@dataclass(slots=True)
class StageFactory:
    """Resolve orchestration stages by topology stage type."""

    registry: StageRegistry = field(default_factory=StageRegistry)

    def resolve(
        self,
        topology: PipelineTopologyConfig | str | None,
        stage: StageDefinition,
    ) -> object:
        try:
            builder = self.registry.get_builder(stage.stage_type)
            metadata = self.registry.get_metadata(stage.stage_type)
        except StageRegistryError as exc:  # pragma: no cover - defensive guard
            raise StageResolutionError(
                f"Pipeline '{getattr(topology, 'name', topology)}' declared unknown stage type '{stage.stage_type}'"
            ) from exc
        instance = builder(topology if isinstance(topology, PipelineTopologyConfig) else None, stage)
        logger.debug(
            "dagster.stage.resolved",
            pipeline=getattr(topology, "name", topology),
            stage=stage.name,
            stage_type=stage.stage_type,
            description=metadata.description,
        )
        return instance

    def get_metadata(self, stage_type: str) -> StageMetadata:
        return self.registry.get_metadata(stage_type)

    def register_stage(
        self,
        *,
        metadata: StageMetadata,
        builder: Callable[[PipelineTopologyConfig | None, StageDefinition], object],
        replace: bool = False,
    ) -> None:
        self.registry.register_stage(metadata=metadata, builder=builder, replace=replace)

    def load_plugins(self) -> list[str]:
        return self.registry.load_plugins()


@op(
    name="bootstrap",
    out=Out(dict),
    config_schema={
        "context": Field(dict),
        "adapter_request": Field(dict),
        "payload": Field(dict),
        "resume": Field(dict, default_value={}),
    },
)
def bootstrap_op(context) -> dict[str, Any]:
    """Initialise the orchestration state for a Dagster run."""

    ctx_payload = context.op_config["context"]
    adapter_payload = context.op_config["adapter_request"]
    payload = context.op_config.get("payload", {})
    resume_payload = context.op_config.get("resume", {})

    stage_ctx = StageContext(
        tenant_id=ctx_payload["tenant_id"],
        job_id=ctx_payload.get("job_id"),
        doc_id=ctx_payload.get("doc_id"),
        correlation_id=ctx_payload.get("correlation_id"),
        metadata=ctx_payload.get("metadata", {}),
        pipeline_name=ctx_payload.get("pipeline_name"),
        pipeline_version=ctx_payload.get("pipeline_version"),
    )
    adapter_request = AdapterRequest.model_validate(adapter_payload)

    state = {
        "context": stage_ctx,
        "adapter_request": adapter_request,
        "payload": payload,
        "results": {},
        "job_id": stage_ctx.job_id,
        "resume": dict(resume_payload or {}),
        "phases": {"order": [], "completed": []},
    }
    logger.debug(
        "dagster.bootstrap.initialised",
        tenant_id=stage_ctx.tenant_id,
        pipeline=stage_ctx.pipeline_name,
    )
    return state


def _apply_stage_output(
    metadata: StageMetadata,
    stage_name: str,
    state: dict[str, Any],
    output: Any,
) -> dict[str, Any]:
    metadata.output_handler(state, stage_name, output)
    snapshot = metadata.result_snapshot(state, output)
    state.setdefault("results", {})[stage_name] = {
        "type": metadata.stage_type,
        "output": snapshot,
    }
    return state


def _infer_output_count(metadata: StageMetadata, output: Any) -> int:
    try:
        count = metadata.output_counter(output)
    except Exception:  # pragma: no cover - defensive guard
        return 0
    if not isinstance(count, int):  # pragma: no cover - defensive guard
        try:
            count = int(count)
        except Exception:
            return 0
    return max(count, 0)


def _resolve_upstream_value(
    state: Mapping[str, Any], metadata: StageMetadata, stage_factory: StageFactory
) -> Any:
    if metadata.dependencies:
        aggregated: dict[str, Any] = {}
        for dependency in metadata.dependencies:
            try:
                dep_metadata = stage_factory.get_metadata(dependency)
            except StageRegistryError:  # pragma: no cover - defensive guard
                continue
            dep_keys = dep_metadata.state_keys
            if not dep_keys:
                continue
            if len(dep_keys) == 1:
                key = dep_keys[0]
                aggregated[key] = state.get(key)
            else:
                aggregated[dependency] = {key: state.get(key) for key in dep_keys}
        if aggregated:
            if len(aggregated) == 1:
                return next(iter(aggregated.values()))
            return aggregated
    keys = metadata.state_keys
    if keys is None or not keys:
        return state.get(metadata.stage_type)
    if len(keys) == 1:
        return state.get(keys[0])
    return {key: state.get(key) for key in keys}


def _make_phase_marker_op(
    topology: PipelineTopologyConfig,
    phase: str,
    phase_plan: PipelinePhasePlan,
    *,
    action: str,
):
    op_name = f"{phase}_{action}"

    @op(
        name=op_name,
        ins={"state": In(dict)},
        out=Out(dict),
        required_resource_keys={"job_ledger"},
    )
    def _marker_op(context, state: dict[str, Any]) -> dict[str, Any]:
        ledger: JobLedger = context.resources.job_ledger
        phases_state = state.setdefault("phases", {})
        phases_state.setdefault("order", list(phase_plan.phases))
        resume_state = state.setdefault("resume", {})
        job_id = state.get("job_id") or getattr(state.get("context"), "job_id", None)
        phase_index = phase_plan.phase_index(phase)
        resume_phase = resume_state.get("phase")

        if action == "start":
            skip_phase = False
            if resume_phase is not None:
                try:
                    resume_index = phase_plan.phase_index(resume_phase)
                except KeyError:
                    resume_index = None
                if resume_index is not None and phase_index < resume_index:
                    skip_phase = True
            phases_state["current"] = phase
            phases_state["skip_current"] = skip_phase
            record_phase_transition(topology.name, phase, "start")
            if job_id:
                status = "skipped" if skip_phase else "running"
                metadata = {f"phase.{phase}.status": status}
                metadata["phase.current"] = None if skip_phase else phase
                ledger.update_metadata(job_id, metadata)
        else:
            skip_phase = phases_state.get("skip_current", False)
            completed = phases_state.setdefault("completed", [])
            if phase not in completed:
                completed.append(phase)
            phases_state["skip_current"] = False
            phases_state["current"] = None
            transition = "skipped" if skip_phase else "completed"
            record_phase_transition(topology.name, phase, transition)
            if job_id:
                metadata = {f"phase.{phase}.status": transition, "phase.current": None}
                ledger.update_metadata(job_id, metadata)
            if resume_state.get("phase") == phase and not resume_state.get("stage"):
                resume_state.pop("phase", None)
        return state

    return _marker_op


def _make_phase_graph(
    topology: PipelineTopologyConfig,
    phase: str,
    stage_names: Iterable[str],
    stage_ops: Mapping[str, Any],
    phase_plan: PipelinePhasePlan,
) -> Any:
    safe_name = _normalise_name(topology.name)
    start_op = _make_phase_marker_op(topology, phase, phase_plan, action="start")
    end_op = _make_phase_marker_op(topology, phase, phase_plan, action="end")
    start_alias = start_op.alias(f"{phase}_start")
    end_alias = end_op.alias(f"{phase}_end")
    stage_aliases = [stage_ops[name].alias(name) for name in stage_names]

    @graph(name=f"{safe_name}_{phase}_phase")
    def _phase_graph(state):
        state = start_alias(state)
        for op_def in stage_aliases:
            state = op_def(state)
        state = end_alias(state)
        return state

    return _phase_graph


def _make_stage_op(
    topology: PipelineTopologyConfig,
    stage_definition: StageDefinition,
    phase_plan: PipelinePhasePlan,
):
    stage_type = stage_definition.stage_type
    stage_name = stage_definition.name
    policy_name = stage_definition.policy or "default"
    stage_phase = stage_definition.phase
    stage_position = phase_plan.stage_positions.get(stage_name, 0)

    @op(
        name=stage_name,
        ins={"state": In(dict)},
        out=Out(dict),
        required_resource_keys={
            "stage_factory",
            "resilience_policies",
            "job_ledger",
            "job_state_manager",
            "event_emitter",
        },
    )
    def _stage_op(context, state: dict[str, Any]) -> dict[str, Any]:
        stage_factory: StageFactory = context.resources.stage_factory
        policy_loader: ResiliencePolicyLoader = context.resources.resilience_policies

        phases_state = state.setdefault("phases", {})
        resume_state = state.setdefault("resume", {})
        skip_phase = phases_state.get("skip_current", False)
        if skip_phase:
            logger.debug(
                "dagster.stage.skipped",
                pipeline=topology.name,
                stage=stage_name,
                reason="phase-skipped",
            )
            return state

        resume_phase = resume_state.get("phase")
        if resume_phase == stage_phase:
            resume_stage = resume_state.get("stage")
            if resume_stage:
                resume_index = phase_plan.stage_positions.get(resume_stage, stage_position)
                if stage_name != resume_stage and stage_position < resume_index:
                    logger.debug(
                        "dagster.stage.skipped",
                        pipeline=topology.name,
                        stage=stage_name,
                        reason="before-resume-stage",
                    )
                    return state

        stage = stage_factory.resolve(topology.name, stage_definition)
        bind_runtime = getattr(stage, "bind_runtime", None)
        if callable(bind_runtime):
            bind_runtime(job_ledger=context.resources.job_ledger)
        metadata = stage_factory.get_metadata(stage_type)
        policy_loader: ResiliencePolicyLoader = context.resources.resilience_policies
        state_manager: LedgerStateManager = context.resources.job_state_manager
        ledger: JobLedger = context.resources.job_ledger
        emitter: StageEventEmitter = context.resources.event_emitter

        current_phase_index = int(state.get("phase_index", 1))
        phase_ready = bool(state.get("phase_ready", True))
        target_phase_index = stage_definition.phase_index
        if stage_type != "gate" and target_phase_index > current_phase_index and not phase_ready:
            logger.debug(
                "dagster.stage.skipped_phase",
                pipeline=topology.name,
                stage=stage_name,
                target_phase=f"phase-{target_phase_index}",
                current_phase=f"phase-{current_phase_index}",
            )
            return state

        stage = stage_factory.resolve(topology, stage_definition)
        metadata = stage_factory.get_metadata(stage_type)

        execute = getattr(stage, "execute")
        execution_state: dict[str, Any] = {
            "attempts": 0,
            "duration": 0.0,
            "failed": False,
            "error": None,
        }

        gate_result: GateEvaluationResult | None = None

        def _on_retry(retry_state: Any) -> None:
            job_identifier = state.get("job_id")
            if job_identifier:
                state_manager.record_retry(job_identifier, stage_name)
            sleep_seconds = getattr(getattr(retry_state, "next_action", None), "sleep", 0.0) or 0.0
            attempt_number = getattr(retry_state, "attempt_number", 0) + 1
            error = getattr(getattr(retry_state, "outcome", None), "exception", lambda: None)()
            reason = str(error) if error else "retry"
            emitter.emit_retrying(
                state["context"],
                stage_name,
                attempt=attempt_number,
                backoff_ms=int(sleep_seconds * 1000),
                reason=reason,
            )

        def _on_success(attempts: int, duration: float) -> None:
            execution_state["attempts"] = attempts
            execution_state["duration"] = duration

        def _on_failure(error: BaseException, attempts: int) -> None:
            execution_state["attempts"] = attempts
            execution_state["failed"] = True
            execution_state["error"] = error

        hooks = StageExecutionHooks(
            on_retry=_on_retry,
            on_success=_on_success,
            on_failure=_on_failure,
        )

        stage_ctx: StageContext = state["context"]
        job_id = stage_ctx.job_id or state.get("job_id")

        attempt = state_manager.stage_started(job_id, stage_name)
        initial_attempt = attempt.attempt
        emitter.emit_started(stage_ctx, stage_name, attempt=initial_attempt)

        start_time = time.perf_counter()

        try:
            if stage_type == "gate":
                gate_stage = stage

                def _default_failure_result(message: str) -> GateEvaluationResult:
                    return GateEvaluationResult(
                        gate=gate_stage.definition.name,
                        satisfied=False,
                        attempts=execution_state.get("attempts", 0),
                        duration_seconds=execution_state.get("duration", 0.0),
                        details={},
                        last_error=message,
                    )

                def _update_gate_metadata(result: GateEvaluationResult, status: str) -> None:
                    record_gate_evaluation(gate_stage.definition.name, success=status == "passed")
                    if result.duration_seconds:
                        observe_gate_duration(gate_stage.definition.name, result.duration_seconds)
                    if status == "timeout":
                        record_gate_timeout(gate_stage.definition.name)
                    state.setdefault("gate_results", {})[
                        gate_stage.definition.name
                    ] = result.details
                    if job_id:
                        metadata_update = {
                            f"gate.{gate_stage.definition.name}.status": status,
                            f"gate.{gate_stage.definition.name}.attempts": result.attempts,
                            f"gate.{gate_stage.definition.name}.duration_ms": int(
                                result.duration_seconds * 1000
                            ),
                            f"gate.{gate_stage.definition.name}.resume_stage": gate_stage.definition.resume_stage,
                            f"gate.{gate_stage.definition.name}.resume_phase": gate_stage.definition.resume_phase,
                        }
                        if result.last_error:
                            metadata_update[
                                f"gate.{gate_stage.definition.name}.error"
                            ] = result.last_error
                        ledger.update_metadata(job_id, metadata_update)

                try:
                    gate_result = gate_stage.evaluate(stage_ctx, ledger, state)
                except GateTimeoutError as exc:
                    gate_result = exc.result or _default_failure_result(str(exc))
                    execution_state["attempts"] = gate_result.attempts or 1
                    execution_state["duration"] = gate_result.duration_seconds
                    _update_gate_metadata(gate_result, "timeout")
                    raise
                except GateConditionError as exc:
                    gate_result = exc.result or _default_failure_result(str(exc))
                    execution_state["attempts"] = gate_result.attempts or 1
                    execution_state["duration"] = gate_result.duration_seconds
                    _update_gate_metadata(gate_result, "failed")
                    raise
                else:
                    execution_state["attempts"] = gate_result.attempts or 1
                    execution_state["duration"] = gate_result.duration_seconds
                    _update_gate_metadata(gate_result, "passed")
                    result = gate_result
            else:
                wrapped = policy_loader.apply(policy_name, stage_name, execute, hooks=hooks)
                if stage_type == "ingest":
                    adapter_request: AdapterRequest = state["adapter_request"]
                    result = wrapped(stage_ctx, adapter_request)
                elif stage_type in {"parse", "ir-validation"}:
                    payloads = state.get("payloads", [])
                    result = wrapped(stage_ctx, payloads)
                elif stage_type == "chunk":
                    document = state.get("document")
                    result = wrapped(stage_ctx, document)
                elif stage_type == "embed":
                    chunks = state.get("chunks", [])
                    result = wrapped(stage_ctx, chunks)
                elif stage_type == "index":
                    batch = state.get("embedding_batch")
                    result = wrapped(stage_ctx, batch)
                elif stage_type == "extract":
                    document = state.get("document")
                    result = wrapped(stage_ctx, document)
                elif stage_type == "knowledge-graph":
                    entities = state.get("entities", [])
                    claims = state.get("claims", [])
                    result = wrapped(stage_ctx, entities, claims)
                else:  # pragma: no cover - guard for future expansion
                    upstream = _resolve_upstream_value(state, metadata, stage_factory)
                    result = wrapped(stage_ctx, upstream)
        except Exception as exc:
            attempts = execution_state.get("attempts") or 1
            if stage_type == "gate" and gate_result is not None:
                attempts = gate_result.attempts or attempts
                execution_state["duration"] = gate_result.duration_seconds
            emitter.emit_failed(stage_ctx, stage_name, attempt=attempts, error=str(exc))
            if job_id:
                if stage_type == "download":
                    try:
                        ledger.record_pdf_error(job_id, error=str(exc))
                    except JobLedgerError:
                        logger.debug("dagster.stage.download.error_record_failed", job_id=job_id)
                ledger.mark_failed(job_id, stage=stage_name, reason=str(exc))
            raise

        updated = dict(state)
        if stage_type == "gate":
            gate_output = {
                "status": result.status,
                "attempts": result.attempts,
                "elapsed_seconds": result.elapsed_seconds,
                "metadata": result.metadata,
            }
            gate_name = getattr(stage_definition, "gate", stage_name)
            updated.setdefault("gates", {})[stage_name] = gate_output
            updated.setdefault("results", {})[stage_name] = {
                "type": stage_type,
                "output": gate_output,
            }
            if result.should_resume:
                updated["phase_index"] = stage_definition.phase_index + 1
                updated["phase_ready"] = True
            else:
                updated["phase_index"] = stage_definition.phase_index
                updated["phase_ready"] = False
            if job_id:
                metadata_update = {
                    "phase_index": updated["phase_index"],
                    "phase_ready": updated["phase_ready"],
                    f"gate.{gate_name}.status": result.status,
                }
                resume_target = getattr(getattr(stage, "gate", None), "resume_stage", None)
                if resume_target:
                    metadata_update["resume_stage"] = resume_target
                    metadata_update[f"gate.{gate_name}.resume_stage"] = resume_target
                ledger.update_metadata(job_id, metadata_update)
            record_gate_evaluation(gate_name, result.status)
            if result.should_resume and updated["phase_index"] != current_phase_index:
                record_phase_transition(
                    topology.name,
                    f"phase-{current_phase_index}",
                    f"phase-{updated['phase_index']}",
                )
            output_reference = gate_output
        else:
            _apply_stage_output(metadata, stage_name, updated, result)
            output_reference = result
        attempts = execution_state.get("attempts") or attempt.attempt
        duration_seconds = execution_state.get("duration") or (time.perf_counter() - start_time)
        duration_ms = int(duration_seconds * 1000)
        output_count = _infer_output_count(metadata, output_reference)

        if job_id and stage_type == "download":
            downloads = updated.get("downloaded_files") or []
            success_record = next(
                (
                    record
                    for record in downloads
                    if getattr(record, "status", "") == "success"
                    and getattr(record, "storage_key", None)
                ),
                None,
            )
            if success_record is not None:
                try:
                    ledger.record_pdf_download(
                        job_id,
                        url=str(getattr(success_record, "url", "")),
                        storage_key=str(getattr(success_record, "storage_key")),
                        size_bytes=getattr(success_record, "size_bytes", None),
                        content_type=getattr(success_record, "content_type", None),
                        checksum=getattr(success_record, "checksum", None),
                    )
                except JobLedgerError:
                    logger.debug("dagster.stage.download.ledger_update_failed", job_id=job_id)
            else:
                failure = next(
                    (
                        getattr(record, "error", None)
                        for record in downloads
                        if getattr(record, "status", "") == "failed"
                    ),
                    None,
                )
                if failure:
                    try:
                        ledger.record_pdf_error(job_id, error=str(failure))
                    except JobLedgerError:
                        logger.debug("dagster.stage.download.error_record_failed", job_id=job_id)

        if job_id and stage_type == "mineru":
            try:
                ledger.set_pdf_ir_ready(job_id, True)
            except JobLedgerError:
                logger.debug("dagster.stage.mineru.ledger_update_failed", job_id=job_id)
            mineru_metadata = updated.get("mineru_metadata")
            mineru_duration = updated.get("mineru_duration")
            metadata_updates: dict[str, object] = {}
            if isinstance(mineru_metadata, Mapping):
                metadata_updates.update(
                    {f"mineru.{key}": value for key, value in mineru_metadata.items()}
                )
            if mineru_duration is not None:
                metadata_updates["mineru.duration_seconds"] = float(mineru_duration)
            if metadata_updates:
                try:
                    ledger.update_metadata(job_id, metadata_updates)
                except JobLedgerError:
                    logger.debug("dagster.stage.mineru.metadata_update_failed", job_id=job_id)

        if job_id:
            ledger.update_metadata(
                job_id,
                {
                    f"stage.{stage_name}.attempts": attempts,
                    f"stage.{stage_name}.output_count": output_count,
                    f"stage.{stage_name}.duration_ms": duration_ms,
                },
            )
        emitter.emit_completed(
            stage_ctx,
            stage_name,
            attempt=attempts,
            duration_ms=duration_ms,
            output_count=output_count,
        )
        logger.debug(
            "dagster.stage.completed",
            pipeline=topology.name,
            stage=stage_name,
            stage_type=stage_type,
            policy=policy_name,
            attempts=attempts,
            duration_ms=duration_ms,
            output_count=output_count,
        )
        if stage_type == "download" and job_id:
            ledger.set_pdf_downloaded(job_id, True)
        if resume_state.get("phase") == stage_phase and resume_state.get("stage") == stage_name:
            resume_state.pop("stage", None)
        return updated

    return _stage_op


@dataclass(slots=True)
class BuiltPipelineJob:
    job_name: str
    job_definition: Any
    final_node: str
    version: str
    total_phases: int


def _normalise_name(name: str) -> str:
    """Return a Dagster-safe identifier derived from the pipeline name."""

    candidate = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    if not candidate:
        return "pipeline"
    if candidate[0].isdigit():
        candidate = f"p_{candidate}"
    return candidate


def _build_pipeline_job(
    topology: PipelineTopologyConfig,
    *,
    resource_defs: Mapping[str, ResourceDefinition],
) -> BuiltPipelineJob:
    phase_plan = topology.build_phase_plan()
    stage_ops = {
        stage.name: _make_stage_op(topology, stage, phase_plan)
        for stage in topology.stages
    }

    phase_graphs = [
        (phase, _make_phase_graph(topology, phase, phase_plan.phase_to_stages.get(phase, ()), stage_ops, phase_plan))
        for phase in phase_plan.phases
    ]

    safe_name = _normalise_name(topology.name)

    @graph(name=f"{safe_name}_graph")
    def _pipeline_graph():
        state = bootstrap_op.alias("bootstrap")()
        for phase, graph_def in phase_graphs:
            state = graph_def.alias(f"{phase}_phase")(state)
        return state

    job = _pipeline_graph.to_job(
        name=f"{safe_name}_job",
        resource_defs={
            **resource_defs,
        },
        tags={
            "medical_kg.pipeline": topology.name,
            "medical_kg.pipeline_version": topology.version,
        },
    )

    final_node = "bootstrap"
    if phase_plan.phases:
        final_node = f"{phase_plan.phases[-1]}_phase"

    return BuiltPipelineJob(
        job_name=job.name,
        job_definition=job,
        final_node=final_node,
        version=topology.version,
        total_phases=total_phases,
    )


@dataclass(slots=True)
class DagsterRunResult:
    """Result returned after executing a Dagster job."""

    pipeline: str
    success: bool
    state: dict[str, Any]
    dagster_result: ExecuteInProcessResult


class DagsterOrchestrator:
    """Submit orchestration jobs to Dagster using declarative topology configs."""

    def __init__(
        self,
        pipeline_loader: PipelineConfigLoader,
        resilience_loader: ResiliencePolicyLoader,
        stage_factory: StageFactory,
        *,
        plugin_manager: AdapterPluginManager | None = None,
        job_ledger: JobLedger | None = None,
        kafka_client: KafkaClient | None = None,
        event_emitter: StageEventEmitter | None = None,
        openlineage_emitter: OpenLineageEmitter | None = None,
        pipeline_resource: HaystackPipelineResource | None = None,
        base_path: str | Path | None = None,
    ) -> None:
        self.pipeline_loader = pipeline_loader
        self.resilience_loader = resilience_loader
        self.stage_factory = stage_factory
        self.plugin_manager = plugin_manager or get_plugin_manager()
        self.base_path = Path(base_path or pipeline_loader.base_path)
        self.job_ledger = job_ledger or JobLedger()
        self.state_manager = LedgerStateManager(self.job_ledger)
        self.kafka_client = kafka_client or KafkaClient()
        self.pipeline_resource = pipeline_resource or create_default_pipeline_resource()
        self.event_emitter = event_emitter or StageEventEmitter(self.kafka_client)
        self.openlineage = openlineage_emitter or OpenLineageEmitter()
        self._resource_defs: dict[str, ResourceDefinition] = {
            "stage_factory": ResourceDefinition.hardcoded_resource(stage_factory),
            "resilience_policies": ResourceDefinition.hardcoded_resource(resilience_loader),
            "job_ledger": ResourceDefinition.hardcoded_resource(self.job_ledger),
            "job_state_manager": ResourceDefinition.hardcoded_resource(self.state_manager),
            "event_emitter": ResourceDefinition.hardcoded_resource(self.event_emitter),
            "haystack_pipeline": ResourceDefinition.hardcoded_resource(self.pipeline_resource),
            "plugin_manager": ResourceDefinition.hardcoded_resource(self.plugin_manager),
            "kafka": ResourceDefinition.hardcoded_resource(self.kafka_client),
            "openlineage": ResourceDefinition.hardcoded_resource(self.openlineage),
        }
        self._jobs: dict[str, BuiltPipelineJob] = {}
        self._definitions: Definitions | None = None
        self._refresh_jobs()

    @property
    def definitions(self) -> Definitions:
        if self._definitions is None:
            jobs = [entry.job_definition for entry in self._jobs.values()]
            self._definitions = Definitions(
                jobs=jobs,
                resources=self._resource_defs,
                sensors=[pdf_ir_ready_sensor],
            )
        return self._definitions

    def available_pipelines(self) -> list[str]:
        return sorted(self._jobs)

    def _refresh_jobs(self) -> None:
        job_entries: dict[str, BuiltPipelineJob] = {}
        for path in sorted(self.base_path.glob("*.yaml")):
            topology = self.pipeline_loader.load(path.stem)
            job_entries[topology.name] = _build_pipeline_job(
                topology,
                resource_defs=self._resource_defs,
            )
        self._jobs = job_entries
        self._definitions = None

    def _record_job_attempt(self, job_id: str | None) -> int:
        return self.state_manager.record_job_attempt(job_id)

    def submit(
        self,
        *,
        pipeline: str,
        context: StageContext,
        adapter_request: AdapterRequest,
        payload: Mapping[str, Any],
    ) -> DagsterRunResult:
        if pipeline not in self._jobs:
            self._refresh_jobs()
        try:
            job = self._jobs[pipeline]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unknown pipeline '{pipeline}'") from exc

        run_config = {
            "ops": {
                "bootstrap": {
                    "config": {
                        "context": {
                            "tenant_id": context.tenant_id,
                            "job_id": context.job_id,
                            "doc_id": context.doc_id,
                            "correlation_id": context.correlation_id,
                            "metadata": dict(context.metadata),
                            "pipeline_name": pipeline,
                            "pipeline_version": job.version,
                        },
                        "adapter_request": adapter_request.model_dump(),
                        "payload": dict(payload),
                        "resume": dict(context.metadata.get("resume", {}))
                        if isinstance(context.metadata, Mapping)
                        else {},
                    }
                }
            }
        }

        run_metadata: dict[str, Any] = {}
        if isinstance(context.metadata, Mapping):
            run_metadata = dict(context.metadata)
        run_metadata.setdefault("pipeline_version", job.version)

        job_attempt = self._record_job_attempt(context.job_id)
        run_identifier = context.job_id or context.correlation_id or uuid4().hex

        context_metadata = dict(context.metadata) if isinstance(context.metadata, Mapping) else {}
        self.state_manager.prepare_run(
            context_metadata=context_metadata,
            job_id=context.job_id,
            pipeline=pipeline,
            pipeline_version=job.version,
            adapter_request=adapter_request,
            payload=payload,
        )

        self.openlineage.emit_run_started(
            pipeline,
            run_id=run_identifier,
            context=context,
            attempt=job_attempt,
            run_metadata=run_metadata,
        )

        start_time = time.perf_counter()
        try:
            result = job.job_definition.execute_in_process(run_config=run_config)
        except Exception as exc:
            ledger_entry = self.state_manager.fetch_entry(context.job_id)
            self.openlineage.emit_run_failed(
                pipeline,
                run_id=run_identifier,
                context=context,
                attempt=job_attempt,
                ledger_entry=ledger_entry,
                run_metadata=run_metadata,
                error=str(exc),
            )
            raise

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        ledger_entry = self.state_manager.complete_run(context.job_id)

        self.openlineage.emit_run_completed(
            pipeline,
            run_id=run_identifier,
            context=context,
            attempt=job_attempt,
            ledger_entry=ledger_entry,
            run_metadata=run_metadata,
            duration_ms=duration_ms,
        )

        return DagsterRunResult(
            pipeline=pipeline,
            success=result.success,
            state=final_state,
            dagster_result=result,
        )


def submit_to_dagster(
    orchestrator: DagsterOrchestrator,
    *,
    pipeline: str,
    context: StageContext,
    adapter_request: AdapterRequest,
    payload: Mapping[str, Any] | None = None,
) -> DagsterRunResult:
    """Convenience helper mirroring the legacy orchestration API."""

    return orchestrator.submit(
        pipeline=pipeline,
        context=context,
        adapter_request=adapter_request,
        payload=payload or {},
    )


@sensor(name="pdf_ir_ready_sensor", minimum_interval_seconds=30, required_resource_keys={"job_ledger"})
def pdf_ir_ready_sensor(context: SensorEvaluationContext, job_ledger: JobLedger):
    ledger = job_ledger
    ready_requests: list[RunRequest] = []
    for entry in ledger.all():
        if entry.pipeline_name != "pdf-two-phase":
            continue
        if not entry.pdf_ir_ready or entry.status != "processing":
            continue
        phase_label = entry.phase or entry.metadata.get("phase") or "phase-1"
        try:
            current_phase_index = int(str(phase_label).split("-", maxsplit=1)[1])
        except Exception:
            current_phase_index = 1
        resume_phase_index = max(current_phase_index + 1, 2)
        resume_phase = f"phase-{resume_phase_index}"
        run_key = f"{entry.job_id}-resume-{resume_phase_index}"
        resume_stage = entry.metadata.get("resume_stage", "chunk")
        context_payload = {
            "tenant_id": entry.tenant_id,
            "job_id": entry.job_id,
            "doc_id": entry.doc_key,
            "correlation_id": entry.metadata.get("correlation_id"),
            "metadata": {**dict(entry.metadata), "resume_stage": resume_stage},
            "pipeline_name": entry.pipeline_name,
            "pipeline_version": entry.metadata.get("pipeline_version", entry.pipeline_name or ""),
            "phase": resume_phase,
            "phase_ready": True,
        }
        adapter_payload = entry.metadata.get("adapter_request", {})
        payload = entry.metadata.get("payload", {})
        gate_prefix = "gate.pdf_ir_ready"
        resume_stage = entry.metadata.get(f"{gate_prefix}.resume_stage", "chunk")
        resume_phase = entry.metadata.get(f"{gate_prefix}.resume_phase")
        gate_status = entry.metadata.get(f"{gate_prefix}.status")
        gate_attempts = entry.metadata.get(f"{gate_prefix}.attempts")
        gate_duration = entry.metadata.get(f"{gate_prefix}.duration_ms")
        gate_error = entry.metadata.get(f"{gate_prefix}.error")
        resume_config = {
            "stage": resume_stage,
        }
        if resume_phase:
            resume_config["phase"] = resume_phase
        resume_config["gate"] = {
            "name": "pdf_ir_ready",
            "status": gate_status,
            "attempts": gate_attempts,
            "duration_ms": gate_duration,
            "error": gate_error,
        }
        run_config = {
            "ops": {
                "bootstrap": {
                    "config": {
                        "context": context_payload,
                        "adapter_request": adapter_payload,
                        "payload": payload,
                        "resume": resume_config,
                    }
                }
            }
        }
        ready_requests.append(
            RunRequest(
                run_key=run_key,
                run_config=run_config,
                tags={
                    "medical_kg.pipeline": entry.pipeline_name or "",
                    "medical_kg.resume_stage": resume_stage,
                    "medical_kg.resume_phase": resume_phase or "",
                },
            )
        )
    if not ready_requests:
        yield SkipReason("No PDF ingestion jobs ready for resumption")
        return
    for request in ready_requests:
        yield request


def build_default_orchestrator() -> DagsterOrchestrator:
    """Construct a Dagster orchestrator with default stage builders."""

    pipeline_loader = PipelineConfigLoader()
    resilience_loader = ResiliencePolicyLoader()
    plugin_manager = get_plugin_manager()
    pipeline_resource = create_default_pipeline_resource()
    stage_builders = build_default_stage_factory(plugin_manager, pipeline_resource)
    stage_factory = StageFactory(stage_builders)
    job_ledger = JobLedger()
    kafka_client = KafkaClient()
    event_emitter = StageEventEmitter(kafka_client)
    openlineage_emitter = OpenLineageEmitter()
    return DagsterOrchestrator(
        pipeline_loader,
        resilience_loader,
        stage_factory,
        plugin_manager=plugin_manager,
        job_ledger=job_ledger,
        kafka_client=kafka_client,
        event_emitter=event_emitter,
        openlineage_emitter=openlineage_emitter,
        pipeline_resource=pipeline_resource,
    )


try:  # pragma: no cover - import side effect for CLI usage
    defs = build_default_orchestrator().definitions
except Exception:  # pragma: no cover - avoid hard failure when optional deps missing
    defs = None


__all__ = [
    "DagsterOrchestrator",
    "DagsterRunResult",
    "StageFactory",
    "StageResolutionError",
    "submit_to_dagster",
    "pdf_ir_ready_sensor",
]
