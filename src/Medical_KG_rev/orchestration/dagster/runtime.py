"""Dagster runtime orchestration primitives."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from dagster import (
    Definitions,
    ExecuteInProcessResult,
    In,
    Out,
    ResourceDefinition,
    graph,
    op,
)

from Medical_KG_rev.adapters.plugins.bootstrap import get_plugin_manager
from Medical_KG_rev.adapters.plugins.models import AdapterRequest
from Medical_KG_rev.orchestration.dagster.configuration import (
    PipelineConfigLoader,
    PipelineTopologyConfig,
    ResiliencePolicyLoader,
    StageDefinition,
)
from Medical_KG_rev.orchestration.dagster.stages import build_default_stage_factory
from Medical_KG_rev.orchestration.stages.contracts import StageContext
from Medical_KG_rev.utils.logging import get_logger

logger = get_logger(__name__)


class StageResolutionError(RuntimeError):
    """Raised when a stage cannot be resolved from the registry."""


@dataclass(slots=True)
class StageFactory:
    """Resolve orchestration stages by topology stage type."""

    registry: Mapping[str, Callable[[StageDefinition], object]]

    def resolve(self, pipeline: str, stage: StageDefinition) -> object:
        try:
            factory = self.registry[stage.stage_type]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise StageResolutionError(
                f"Pipeline '{pipeline}' declared unknown stage type '{stage.stage_type}'"
            ) from exc
        instance = factory(stage)
        logger.debug(
            "dagster.stage.resolved",
            pipeline=pipeline,
            stage=stage.name,
            stage_type=stage.stage_type,
        )
        return instance


@op(
    name="bootstrap",
    out=Out(dict),
    config_schema={
        "context": dict,
        "adapter_request": dict,
        "payload": dict,
    },
)
def bootstrap_op(context) -> dict[str, Any]:
    """Initialise the orchestration state for a Dagster run."""

    ctx_payload = context.op_config["context"]
    adapter_payload = context.op_config["adapter_request"]
    payload = context.op_config.get("payload", {})

    stage_ctx = StageContext(
        tenant_id=ctx_payload["tenant_id"],
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
    }
    logger.debug(
        "dagster.bootstrap.initialised",
        tenant_id=stage_ctx.tenant_id,
        pipeline=stage_ctx.pipeline_name,
    )
    return state


def _stage_state_key(stage_type: str) -> str:
    return {
        "ingest": "payloads",
        "parse": "document",
        "ir-validation": "document",
        "chunk": "chunks",
        "embed": "embedding_batch",
        "index": "index_receipt",
        "extract": "extraction",
        "knowledge-graph": "graph_receipt",
    }.get(stage_type, stage_type)


def _apply_stage_output(
    stage_type: str,
    stage_name: str,
    state: dict[str, Any],
    output: Any,
) -> dict[str, Any]:
    if stage_type == "ingest":
        state["payloads"] = output
    elif stage_type in {"parse", "ir-validation"}:
        state["document"] = output
    elif stage_type == "chunk":
        state["chunks"] = output
    elif stage_type == "embed":
        state["embedding_batch"] = output
    elif stage_type == "index":
        state["index_receipt"] = output
    elif stage_type == "extract":
        entities, claims = output
        state["entities"] = entities
        state["claims"] = claims
    elif stage_type == "knowledge-graph":
        state["graph_receipt"] = output
    else:  # pragma: no cover - guard for future expansion
        state[_stage_state_key(stage_type)] = output
    state.setdefault("results", {})[stage_name] = {
        "type": stage_type,
        "output": state.get(_stage_state_key(stage_type)),
    }
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
        ins={"state": In(dict)},
        out=Out(dict),
        required_resource_keys={"stage_factory", "resilience_policies"},
    )
    def _stage_op(context, state: dict[str, Any]) -> dict[str, Any]:
        stage = context.resources.stage_factory.resolve(topology.name, stage_definition)
        policy_loader: ResiliencePolicyLoader = context.resources.resilience_policies

        execute = getattr(stage, "execute")
        wrapped = policy_loader.apply(policy_name, stage_name, execute)

        stage_ctx: StageContext = state["context"]

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
            upstream = state.get(_stage_state_key(stage_type))
            result = wrapped(stage_ctx, upstream)

        updated = dict(state)
        _apply_stage_output(stage_type, stage_name, updated, result)
        logger.debug(
            "dagster.stage.completed",
            pipeline=topology.name,
            stage=stage_name,
            stage_type=stage_type,
            policy=policy_name,
        )
        return updated

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
    stage_factory: StageFactory,
    resilience_loader: ResiliencePolicyLoader,
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
            "stage_factory": ResourceDefinition.hardcoded_resource(stage_factory),
            "resilience_policies": ResourceDefinition.hardcoded_resource(resilience_loader),
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
        base_path: str | Path | None = None,
    ) -> None:
        self.pipeline_loader = pipeline_loader
        self.resilience_loader = resilience_loader
        self.stage_factory = stage_factory
        self.base_path = Path(base_path or pipeline_loader.base_path)
        self._jobs: dict[str, BuiltPipelineJob] = {}
        self._definitions: Definitions | None = None
        self._refresh_jobs()

    @property
    def definitions(self) -> Definitions:
        if self._definitions is None:
            jobs = [entry.job_definition for entry in self._jobs.values()]
            self._definitions = Definitions(jobs=jobs)
        return self._definitions

    def available_pipelines(self) -> list[str]:
        return sorted(self._jobs)

    def _refresh_jobs(self) -> None:
        job_entries: dict[str, BuiltPipelineJob] = {}
        for path in sorted(self.base_path.glob("*.yaml")):
            topology = self.pipeline_loader.load(path.stem)
            job_entries[topology.name] = _build_pipeline_job(
                topology,
                stage_factory=self.stage_factory,
                resilience_loader=self.resilience_loader,
            )
        self._jobs = job_entries
        self._definitions = None

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
                            "doc_id": context.doc_id,
                            "correlation_id": context.correlation_id,
                            "metadata": context.metadata,
                            "pipeline_name": pipeline,
                            "pipeline_version": self.pipeline_loader.load(pipeline).version,
                        },
                        "adapter_request": adapter_request.model_dump(),
                        "payload": dict(payload),
                    }
                }
            }
        }

        result = job.job_definition.execute_in_process(run_config=run_config)
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


def build_default_orchestrator() -> DagsterOrchestrator:
    """Construct a Dagster orchestrator with default stage builders."""

    pipeline_loader = PipelineConfigLoader()
    resilience_loader = ResiliencePolicyLoader()
    stage_builders = build_default_stage_factory(get_plugin_manager())
    stage_factory = StageFactory(stage_builders)
    return DagsterOrchestrator(pipeline_loader, resilience_loader, stage_factory)


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
]
