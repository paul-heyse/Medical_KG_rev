"""Generic pipeline orchestration primitives used by ingestion and retrieval."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextvars import Token
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, TypeVar, overload
from uuid import uuid4

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from Medical_KG_rev.observability.metrics import (
    observe_orchestration_duration,
    observe_orchestration_stage,
    record_orchestration_error,
    record_orchestration_operation,
)
from Medical_KG_rev.utils.errors import ProblemDetail
from Medical_KG_rev.utils.logging import bind_correlation_id, get_correlation_id, reset_correlation_id

from .resilience import TimeoutManager

try:  # pragma: no cover - optional tracing dependency
    from opentelemetry import trace
except Exception:  # pragma: no cover - tracing optional
    trace = None  # type: ignore

import structlog

T = TypeVar("T")


class StageFailure(RuntimeError):
    """Wraps a stage failure with retry metadata and RFC 7807 details."""

    def __init__(
        self,
        message: str,
        *,
        status: int = 500,
        detail: str | None = None,
        stage: str | None = None,
        error_type: str | None = None,
        retriable: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.retriable = retriable
        self.error_type = error_type or ("transient" if retriable else "permanent")
        self.problem = ProblemDetail(
            title=message,
            status=status,
            detail=detail,
            extra=extra or {},
        )


@dataclass(slots=True)
class PipelineContext:
    """Mutable context threaded through pipeline stages."""

    tenant_id: str
    operation: str
    data: dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: uuid4().hex)
    pipeline_version: str | None = None
    errors: list[ProblemDetail] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)
    partial: bool = False
    degraded: bool = False
    degradation_events: list[dict[str, Any]] = field(default_factory=list)

    def with_data(self, **values: Any) -> PipelineContext:
        self.data.update(values)
        return self

    def copy(self) -> PipelineContext:
        return PipelineContext(
            tenant_id=self.tenant_id,
            operation=self.operation,
            data=dict(self.data),
            correlation_id=self.correlation_id,
            pipeline_version=self.pipeline_version,
            errors=list(self.errors),
            stage_timings=dict(self.stage_timings),
            partial=self.partial,
            degraded=self.degraded,
            degradation_events=list(self.degradation_events),
        )


class PipelineStage(Protocol):
    """Protocol implemented by pipeline stages."""

    name: str
    timeout_ms: int | None

    def execute(self, context: PipelineContext) -> PipelineContext: ...


class StageConfig(BaseModel):
    """Declarative representation of a pipeline stage loaded from YAML."""

    name: str
    kind: str
    timeout_ms: int | None = Field(default=None, ge=1)
    options: dict[str, Any] = Field(default_factory=dict)


class PipelineDefinition(BaseModel):
    """List of stages composing a named pipeline."""

    name: str
    stages: list[StageConfig]

    @model_validator(mode="after")
    def _validate_unique_stage_names(self) -> PipelineDefinition:
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate stage names in pipeline '{self.name}'")
        return self


class ProfileDefinition(BaseModel):
    """Definition of profile-specific overrides for pipelines."""

    name: str
    extends: str | None = None
    ingestion: str | None = None
    query: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Configuration describing ingestion/query pipelines and profiles."""

    version: str
    ingestion: dict[str, PipelineDefinition] = Field(default_factory=dict)
    query: dict[str, PipelineDefinition] = Field(default_factory=dict)
    profiles: dict[str, ProfileDefinition] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path | None, *, text: str | None = None) -> PipelineConfig:
        if path is None and text is None:
            raise ValueError("Either path or text must be provided")
        raw: Any
        if text is not None:
            raw = yaml.safe_load(text) or {}
        else:
            resolved = Path(path).expanduser()
            raw = yaml.safe_load(resolved.read_text()) if resolved.exists() else {}
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:  # pragma: no cover - validation formatting
            raise ValueError(f"Invalid pipeline configuration: {exc}") from exc


class PipelineExecutor:
    """Sequentially executes pipeline stages with timing and error recording."""

    def __init__(self, stages: Sequence[PipelineStage], *, operation: str, pipeline: str) -> None:
        self.stages = list(stages)
        self.operation = operation
        self.pipeline = pipeline
        self._timeout = TimeoutManager()

    def run(self, context: PipelineContext) -> PipelineContext:
        token = None
        current = get_correlation_id()
        if current != context.correlation_id:
            token = bind_correlation_id(context.correlation_id)
        started = perf_counter()
        pipeline_span = (
            _TRACER.start_as_current_span(f"{self.operation}.{self.pipeline}") if _TRACER else nullcontext()
        )
        logger.info(
            "orchestration.pipeline.start",
            operation=self.operation,
            pipeline=self.pipeline,
            tenant_id=context.tenant_id,
            correlation_id=context.correlation_id,
        )
        try:
            with pipeline_span as span:
                if span is not None:  # pragma: no cover - tracing optional
                    span.set_attribute("pipeline.operation", self.operation)
                    span.set_attribute("pipeline.name", self.pipeline)
                    span.set_attribute("tenant_id", context.tenant_id)
                    span.set_attribute("correlation_id", context.correlation_id)
                for stage in self.stages:
                    stage_started = perf_counter()
                    failure: StageFailure | None = None
                    status = "success"
                    stage_span = (
                        _TRACER.start_as_current_span(f"{self.operation}.{stage.name}")
                        if _TRACER
                        else nullcontext()
                    )
                    with stage_span as stage_otel:
                        if stage_otel is not None:  # pragma: no cover - tracing optional
                            stage_otel.set_attribute("pipeline.stage", stage.name)
                            stage_otel.set_attribute("pipeline.operation", self.operation)
                            stage_otel.set_attribute("pipeline.name", self.pipeline)
                            stage_otel.set_attribute("correlation_id", context.correlation_id)
                        logger.info(
                            "orchestration.stage.start",
                            stage=stage.name,
                            operation=self.operation,
                            pipeline=self.pipeline,
                            correlation_id=context.correlation_id,
                        )
                        try:
                            context = stage.execute(context)
                        except StageFailure as exc:
                            failure = exc
                            status = "error"
                        except Exception as exc:  # pragma: no cover - defensive guard
                            failure = StageFailure(
                                "Stage execution failed",
                                detail=str(exc),
                                stage=stage.name,
                                retriable=False,
                            )
                            status = "error"
                        finally:
                            duration = perf_counter() - stage_started
                            context.stage_timings[stage.name] = duration
                            try:
                                self._timeout.ensure(
                                    operation=self.operation,
                                    stage=stage.name,
                                    duration_seconds=duration,
                                    timeout_ms=getattr(stage, "timeout_ms", None),
                                )
                            except StageFailure as timeout_failure:
                                failure = timeout_failure
                                status = "error"
                            observe_orchestration_stage(self.operation, stage.name, duration)
                            if stage_otel is not None:  # pragma: no cover - tracing optional
                                stage_otel.set_attribute("stage.duration_ms", round(duration * 1000, 3))
                                stage_otel.set_attribute("stage.status", status)
                        if failure:
                            context.errors.append(failure.problem)
                            context.partial = context.partial or failure.retriable or bool(context.data)
                            context.degraded = True
                            context.degradation_events.append(
                                {
                                    "stage": failure.stage or stage.name,
                                    "reason": failure.problem.title,
                                    "retriable": failure.retriable,
                                }
                            )
                            record_orchestration_error(
                                self.operation,
                                failure.stage or stage.name,
                                failure.error_type,
                            )
                            record_orchestration_operation(
                                self.operation,
                                context.tenant_id,
                                "partial" if context.partial else "failed",
                            )
                            logger.warning(
                                "orchestration.stage.failure",
                                stage=stage.name,
                                operation=self.operation,
                                pipeline=self.pipeline,
                                correlation_id=context.correlation_id,
                                error=failure.problem.title,
                            )
                            raise failure
                        logger.info(
                            "orchestration.stage.success",
                            stage=stage.name,
                            operation=self.operation,
                            pipeline=self.pipeline,
                            correlation_id=context.correlation_id,
                            duration_ms=round(context.stage_timings[stage.name] * 1000, 3),
                        )
                total = perf_counter() - started
                context.stage_timings.setdefault("total", total)
                observe_orchestration_duration(self.operation, self.pipeline, total)
                record_orchestration_operation(self.operation, context.tenant_id, "success")
                context.pipeline_version = context.pipeline_version or self.pipeline
                if span is not None:  # pragma: no cover - tracing optional
                    span.set_attribute("pipeline.duration_ms", round(total * 1000, 3))
                logger.info(
                    "orchestration.pipeline.complete",
                    operation=self.operation,
                    pipeline=self.pipeline,
                    correlation_id=context.correlation_id,
                    duration_ms=round(total * 1000, 3),
                )
                return context
        finally:
            if token is not None:
                reset_correlation_id(token)


@dataclass(slots=True)
class ParallelResult:
    name: str
    value: Any | None = None
    duration: float = 0.0
    error: ProblemDetail | None = None
    timed_out: bool = False


class ParallelExecutor:
    """Executes callables concurrently while propagating correlation IDs."""

    def __init__(self, *, max_workers: int = 8) -> None:
        self._max_workers = max_workers
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @overload
    def run(
        self,
        tasks: Mapping[str, Callable[[], T]],
        *,
        timeout_ms: int | None = None,
        correlation_id: str | None = None,
    ) -> dict[str, ParallelResult]:
        ...

    def run(
        self,
        tasks: Mapping[str, Callable[[], Any]],
        *,
        timeout_ms: int | None = None,
        correlation_id: str | None = None,
    ) -> dict[str, ParallelResult]:
        token = None
        current = get_correlation_id()
        if correlation_id and current != correlation_id:
            token = bind_correlation_id(correlation_id)
        try:
            futures: dict[Future[Any], str] = {}
            for name, task in tasks.items():
                futures[self._pool.submit(_call_with_correlation, task, correlation_id)] = name
            timeout_seconds = None if timeout_ms is None else timeout_ms / 1000.0
            done, pending = wait(futures.keys(), timeout=timeout_seconds)
            results: dict[str, ParallelResult] = {}
            for future in done:
                name = futures[future]
                elapsed = 0.0
                try:
                    value, elapsed = future.result()
                    results[name] = ParallelResult(name=name, value=value, duration=elapsed)
                except StageFailure as failure:
                    results[name] = ParallelResult(
                        name=name,
                        duration=elapsed,
                        error=failure.problem,
                    )
                except Exception as exc:  # pragma: no cover - safety
                    results[name] = ParallelResult(
                        name=name,
                        duration=elapsed,
                        error=ProblemDetail(
                            title="Parallel task failure",
                            status=500,
                            detail=str(exc),
                        ),
                    )
            for future in pending:
                name = futures[future]
                future.cancel()
                results[name] = ParallelResult(
                    name=name,
                    timed_out=True,
                    error=ProblemDetail(
                        title="Parallel task timeout",
                        status=504,
                        detail=f"Task '{name}' exceeded timeout",
                    ),
                )
            return results
        finally:
            if token is not None:
                reset_correlation_id(token)


def _call_with_correlation(task: Callable[[], Any], correlation_id: str | None) -> Any:
    token = None
    if correlation_id and get_correlation_id() != correlation_id:
        token = bind_correlation_id(correlation_id)
    started = perf_counter()
    try:
        return task(), perf_counter() - started
    finally:
        if token is not None:
            reset_correlation_id(token)


def ensure_correlation_id(value: str | None = None) -> tuple[str, Token | None]:
    """Ensure a correlation identifier exists and bind it to the context."""

    identifier = value or uuid4().hex
    token = bind_correlation_id(identifier)
    return identifier, token


def iter_stages(pipeline: PipelineDefinition) -> Iterable[str]:
    for stage in pipeline.stages:
        yield stage.name


__all__ = [
    "ParallelExecutor",
    "ParallelResult",
    "PipelineConfig",
    "PipelineContext",
    "PipelineDefinition",
    "PipelineExecutor",
    "PipelineStage",
    "ProfileDefinition",
    "StageConfig",
    "StageFailure",
    "ensure_correlation_id",
    "iter_stages",
]
logger = structlog.get_logger(__name__)
_TRACER = trace.get_tracer(__name__) if trace else None
