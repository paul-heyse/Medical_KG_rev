"""Dagster runtime orchestration primitives."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from dagster import (
    Definitions,
    ExecuteInProcessResult,
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
    PipelineTopologyConfig,
    StageExecutionHooks,
    ResiliencePolicyLoader,
    StageDefinition,
)
from Medical_KG_rev.orchestration.dagster.stages import (
    HaystackPipelineResource,
    create_default_pipeline_resource,
    create_stage_plugin_manager,
)
from Medical_KG_rev.orchestration.events import StageEventEmitter
from Medical_KG_rev.orchestration.kafka import KafkaClient
from Medical_KG_rev.orchestration.ledger import JobLedger, JobLedgerError
from Medical_KG_rev.orchestration.openlineage import OpenLineageEmitter
from Medical_KG_rev.orchestration.stages.contracts import PipelineState, StageContext
from Medical_KG_rev.orchestration.state import PipelineStatePersister, StatePersistenceError
from Medical_KG_rev.orchestration.stages.plugins import (
    StagePluginBuildError,
    StagePluginLookupError,
    StagePluginManager,
)
from Medical_KG_rev.utils.logging import get_logger

logger = get_logger(__name__)


class StageResolutionError(RuntimeError):
    """Raised when a stage cannot be resolved from the registry."""


@dataclass(slots=True)
class StageFactory:
    """Resolve orchestration stages through the plugin manager."""

    plugin_manager: StagePluginManager

    def resolve(self, pipeline: str, stage: StageDefinition) -> object:
        try:
            instance = self.plugin_manager.build_stage(stage)
        except StagePluginLookupError as exc:
            raise StageResolutionError(
                f"Pipeline '{pipeline}' declared unknown stage type '{stage.stage_type}'"
            ) from exc
        except StagePluginBuildError as exc:
            raise StageResolutionError(
                f"Stage '{stage.name}' of type '{stage.stage_type}' failed to initialise"
            ) from exc
        logger.debug(
            "dagster.stage.resolved",
            pipeline=pipeline,
            stage=stage.name,
            stage_type=stage.stage_type,
        )
        return instance


@op(
    name="bootstrap",
    out=Out(PipelineState),
    config_schema={
        "context": dict,
        "adapter_request": dict,
        "payload": dict,
    },
)
def bootstrap_op(context) -> PipelineState:
    """Initialise the orchestration state for a Dagster run."""

    ctx_payload = context.op_config["context"]
    adapter_payload = context.op_config["adapter_request"]
    payload = context.op_config.get("payload", {})

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

    state = PipelineState.initialise(
        context=stage_ctx,
        adapter_request=adapter_request,
        payload=payload,
    )
    logger.debug(
        "dagster.bootstrap.initialised",
        tenant_id=stage_ctx.tenant_id,
        pipeline=stage_ctx.pipeline_name,
    )
    return state


def _make_stage_op(
    topology: PipelineTopologyConfig,
    stage_definition: StageDefinition,
):
    stage_type = stage_definition.stage_type
    stage_name = stage_definition.name
    policy_name = stage_definition.policy or "default"

    @op(
        name=stage_name,
        ins={"state": In(PipelineState)},
        out=Out(PipelineState),
        required_resource_keys={
            "stage_factory",
            "resilience_policies",
            "job_ledger",
            "event_emitter",
        },
    )
    def _stage_op(context, state: PipelineState) -> PipelineState:
        stage = context.resources.stage_factory.resolve(topology.name, stage_definition)
        policy_loader: ResiliencePolicyLoader = context.resources.resilience_policies
        ledger: JobLedger = context.resources.job_ledger
        emitter: StageEventEmitter = context.resources.event_emitter
        persister = PipelineStatePersister(metadata_store=ledger)
        dependencies = stage_definition.depends_on

        execute = getattr(stage, "execute")
        execution_state: dict[str, Any] = {
            "attempts": 0,
            "duration": 0.0,
            "failed": False,
            "error": None,
        }

        def _on_retry(retry_state: Any) -> None:
            job_identifier = state.job_id
            if job_identifier:
                ledger.increment_retry(job_identifier, stage_name)
            sleep_seconds = getattr(getattr(retry_state, "next_action", None), "sleep", 0.0) or 0.0
            attempt_number = getattr(retry_state, "attempt_number", 0) + 1
            error = getattr(getattr(retry_state, "outcome", None), "exception", lambda: None)()
            reason = str(error) if error else "retry"
            state.rollback_to(stage_name, restore_stage_results=False)
            emitter.emit_retrying(
                state.context,
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

        wrapped = policy_loader.apply(policy_name, stage_name, execute, hooks=hooks)

        stage_ctx: StageContext = state.context
        job_id = state.job_id or stage_ctx.job_id

        initial_attempt = 1
        if job_id:
            entry = ledger.mark_stage_started(job_id, stage_name)
            initial_attempt = entry.retry_count_per_stage.get(stage_name, 0) + 1
            state.job_id = entry.job_id
            stage_ctx.job_id = entry.job_id
        emitter.emit_started(stage_ctx, stage_name, attempt=initial_attempt)

        start_time = time.perf_counter()

        state.ensure_tenant_scope(stage_ctx.tenant_id)

        checkpoint_label = stage_name

        try:
            if dependencies:
                state.ensure_dependencies(stage_name, dependencies)
            state.validate_transition(stage_type)
            state.create_checkpoint(checkpoint_label)
            result = wrapped(stage_ctx, state)
        except Exception as exc:
            attempts = execution_state.get("attempts") or 1
            state.rollback_to(checkpoint_label, restore_stage_results=False)
            state.mark_stage_failed(
                stage_name,
                error=str(exc),
                stage_type=stage_type,
            )
            state.clear_checkpoint(checkpoint_label)
            snapshot_b64 = state.serialise_base64()
            if job_id:
                try:
                    snapshot_b64 = persister.persist_state(job_id, stage=stage_name, state=state)
                except StatePersistenceError as persist_exc:  # pragma: no cover - defensive
                    logger.warning(
                        "dagster.stage.snapshot_persist_failed",
                        job_id=job_id,
                        stage=stage_name,
                        error=str(persist_exc),
                    )
            emitter.emit_failed(
                stage_ctx,
                stage_name,
                attempt=attempts,
                error=str(exc),
                state_snapshot=snapshot_b64,
            )
            if job_id:
                ledger.mark_failed(job_id, stage=stage_name, reason=str(exc))
                ledger.update_metadata(
                    job_id,
                    {
                        f"stage.{stage_name}.error": str(exc),
                        f"state.{stage_name}.snapshot": snapshot_b64,
                    },
                )
            raise

        state.apply_stage_output(stage_type, stage_name, result)
        attempts = execution_state.get("attempts") or 1
        duration_seconds = execution_state.get("duration") or (time.perf_counter() - start_time)
        duration_ms = int(duration_seconds * 1000)
        output_count = state.infer_output_count(stage_type, result)
        state.record_stage_metrics(
            stage_name,
            stage_type=stage_type,
            attempts=attempts,
            duration_ms=duration_ms,
            output_count=output_count,
        )
        snapshot_b64 = state.serialise_base64()
        if job_id:
            try:
                snapshot_b64 = persister.persist_state(job_id, stage=stage_name, state=state)
            except StatePersistenceError as persist_exc:  # pragma: no cover - defensive
                logger.warning(
                    "dagster.stage.snapshot_persist_failed",
                    job_id=job_id,
                    stage=stage_name,
                    error=str(persist_exc),
                )
        state.clear_checkpoint(checkpoint_label)

        if job_id:
            ledger.update_metadata(
                job_id,
                {
                    f"stage.{stage_name}.attempts": attempts,
                    f"stage.{stage_name}.output_count": output_count,
                    f"stage.{stage_name}.duration_ms": duration_ms,
                    f"state.{stage_name}.snapshot": snapshot_b64,
                },
            )
            if stage_type == "pdf-download":
                ledger.set_pdf_downloaded(job_id)
            elif stage_type == "pdf-ir-gate":
                ledger.set_pdf_ir_ready(job_id)
        emitter.emit_completed(
            stage_ctx,
            stage_name,
            attempt=attempts,
            duration_ms=duration_ms,
            output_count=output_count,
            state_snapshot=snapshot_b64,
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
        return state

    return _stage_op


def _topological_order(stages: list[StageDefinition]) -> list[str]:
    graph: dict[str, set[str]] = {stage.name: set(stage.depends_on) for stage in stages}
    resolved: list[str] = []
    temporary: set[str] = set()
    permanent: set[str] = set()

    def visit(node: str) -> None:
        if node in permanent:
            return
        if node in temporary:
            raise ValueError(f"Cycle detected involving stage '{node}'")
        temporary.add(node)
        for dep in graph.get(node, set()):
            visit(dep)
        temporary.remove(node)
        permanent.add(node)
        resolved.append(node)

    for stage in graph:
        visit(stage)
    return resolved


@dataclass(slots=True)
class BuiltPipelineJob:
    job_name: str
    job_definition: Any
    final_node: str
    version: str


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
    stage_ops = {
        stage.name: _make_stage_op(topology, stage)
        for stage in topology.stages
    }
    order = _topological_order(topology.stages)

    safe_name = _normalise_name(topology.name)

    @graph(name=f"{safe_name}_graph")
    def _pipeline_graph():
        state = bootstrap_op.alias("bootstrap")()
        for stage_name in order:
            op_def = stage_ops[stage_name].alias(stage_name)
            state = op_def(state)
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

    return BuiltPipelineJob(
        job_name=job.name,
        job_definition=job,
        final_node=order[-1] if order else "bootstrap",
        version=topology.version,
    )


@dataclass(slots=True)
class DagsterRunResult:
    """Result returned after executing a Dagster job."""

    pipeline: str
    success: bool
    state: PipelineState
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
        self.kafka_client = kafka_client or KafkaClient()
        self.pipeline_resource = pipeline_resource or create_default_pipeline_resource()
        self.event_emitter = event_emitter or StageEventEmitter(self.kafka_client)
        self.openlineage = openlineage_emitter or OpenLineageEmitter()
        self._resource_defs: dict[str, ResourceDefinition] = {
            "stage_factory": ResourceDefinition.hardcoded_resource(stage_factory),
            "resilience_policies": ResourceDefinition.hardcoded_resource(resilience_loader),
            "job_ledger": ResourceDefinition.hardcoded_resource(self.job_ledger),
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
        if not job_id:
            return 1
        try:
            return self.job_ledger.record_attempt(job_id)
        except JobLedgerError:
            logger.debug("dagster.ledger.missing_job", job_id=job_id)
            return 1

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

        if context.job_id:
            try:
                self.job_ledger.update_metadata(
                    context.job_id,
                    {
                        "pipeline_version": job.version,
                        "correlation_id": context.correlation_id,
                        "adapter_request": adapter_request.model_dump(),
                        "payload": dict(payload),
                    },
                )
            except JobLedgerError:
                logger.debug(
                    "dagster.ledger.metadata_update_failed",
                    job_id=context.job_id,
                    pipeline=pipeline,
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
            ledger_entry = self.job_ledger.get(context.job_id) if context.job_id else None
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

        ledger_entry = None
        if context.job_id:
            try:
                ledger_entry = self.job_ledger.mark_completed(context.job_id)
            except JobLedgerError:
                ledger_entry = self.job_ledger.get(context.job_id)

        self.openlineage.emit_run_completed(
            pipeline,
            run_id=run_identifier,
            context=context,
            attempt=job_attempt,
            ledger_entry=ledger_entry,
            run_metadata=run_metadata,
            duration_ms=duration_ms,
        )

        final_state = result.output_for_node(job.final_node)
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
def pdf_ir_ready_sensor(context: SensorEvaluationContext):
    ledger: JobLedger = context.resources.job_ledger
    ready_requests: list[RunRequest] = []
    for entry in ledger.all():
        if entry.pipeline_name != "pdf-two-phase":
            continue
        if not entry.pdf_ir_ready or entry.status != "processing":
            continue
        run_key = f"{entry.job_id}-resume"
        context_payload = {
            "tenant_id": entry.tenant_id,
            "job_id": entry.job_id,
            "doc_id": entry.doc_key,
            "correlation_id": entry.metadata.get("correlation_id"),
            "metadata": dict(entry.metadata),
            "pipeline_name": entry.pipeline_name,
            "pipeline_version": entry.metadata.get("pipeline_version", entry.pipeline_name or ""),
        }
        adapter_payload = entry.metadata.get("adapter_request", {})
        payload = entry.metadata.get("payload", {})
        run_config = {
            "ops": {
                "bootstrap": {
                    "config": {
                        "context": context_payload,
                        "adapter_request": adapter_payload,
                        "payload": payload,
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
                    "medical_kg.resume_stage": "chunk",
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
    adapter_manager = get_plugin_manager()
    pipeline_resource = create_default_pipeline_resource()
    job_ledger = JobLedger()
    stage_plugin_manager = create_stage_plugin_manager(
        adapter_manager,
        pipeline_resource,
        job_ledger=job_ledger,
    )
    stage_factory = StageFactory(stage_plugin_manager)
    kafka_client = KafkaClient()
    event_emitter = StageEventEmitter(kafka_client)
    openlineage_emitter = OpenLineageEmitter()
    return DagsterOrchestrator(
        pipeline_loader,
        resilience_loader,
        stage_factory,
        plugin_manager=adapter_manager,
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
