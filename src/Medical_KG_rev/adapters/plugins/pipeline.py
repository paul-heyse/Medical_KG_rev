"""Execution pipeline primitives for adapter plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Iterable, Tuple
from uuid import uuid4

from .errors import AdapterPluginError
from .models import AdapterMetadata, AdapterRequest, AdapterResponse, ValidationOutcome

StageCallable = Callable[["AdapterExecutionContext"], "AdapterExecutionContext"]


@dataclass(slots=True)
class StageResult:
    """Timing and outcome telemetry for a single pipeline stage."""

    name: str
    started_at: float
    completed_at: float
    duration_ms: float
    status: str = "success"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AdapterExecutionMetrics:
    """Aggregated metrics captured during adapter execution."""

    started_at: float
    completed_at: float
    duration_ms: float
    stages: tuple[StageResult, ...]


@dataclass(slots=True)
class AdapterExecutionContext:
    """Mutable context propagated through adapter pipeline stages."""

    request: AdapterRequest
    metadata: AdapterMetadata
    plugin: Any
    execution_id: str = field(default_factory=lambda: uuid4().hex)
    pipeline_name: str = "default"
    response: AdapterResponse | None = None
    validation: ValidationOutcome | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    error: AdapterPluginError | None = None
    _stage_results: list[StageResult] = field(default_factory=list)
    _started_at: float = field(default_factory=perf_counter)
    _completed_at: float | None = None

    @property
    def canonical_metadata(self) -> dict[str, Any]:
        return {
            "adapter": self.metadata.name,
            "adapter_version": self.metadata.version,
            "adapter_domain": self.metadata.domain.value,
            "execution_id": self.execution_id,
            "pipeline": self.pipeline_name,
        }

    def record_response(self, response: AdapterResponse) -> "AdapterExecutionContext":
        enriched_metadata = {**response.metadata, **self.canonical_metadata}
        response.metadata = enriched_metadata
        self.response = response
        return self

    def record_validation(self, outcome: ValidationOutcome) -> "AdapterExecutionContext":
        self.validation = outcome
        return self

    def ensure_response(self) -> AdapterResponse:
        if self.response is None:  # pragma: no cover - defensive guard
            raise AdapterPluginError(
                f"Adapter '{self.metadata.name}' pipeline did not yield a response"
            )
        return self.response

    def raise_for_validation(self) -> None:
        if self.validation is None or self.validation.valid:
            return
        errors = ", ".join(self.validation.errors) or "unknown validation error"
        raise AdapterPluginError(f"Validation failed for adapter '{self.metadata.name}': {errors}")

    def record_stage_result(
        self,
        name: str,
        started_at: float,
        completed_at: float,
        *,
        status: str = "success",
        details: dict[str, Any] | None = None,
    ) -> None:
        duration_ms = (completed_at - started_at) * 1000.0
        self._stage_results.append(
            StageResult(
                name=name,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                status=status,
                details=details or {},
            )
        )
        self._completed_at = completed_at

    def record_failure(self, error: AdapterPluginError) -> None:
        self.error = error

    def record_success(self) -> None:
        self.error = None

    def finalize(self) -> None:
        if self._completed_at is None:
            self._completed_at = perf_counter()

    @property
    def metrics(self) -> AdapterExecutionMetrics:
        completed_at = self._completed_at or self._started_at
        duration_ms = (completed_at - self._started_at) * 1000.0
        return AdapterExecutionMetrics(
            started_at=self._started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            stages=tuple(self._stage_results),
        )


# Backwards compatibility export
AdapterExecutionState = AdapterExecutionContext


@dataclass(slots=True)
class AdapterStage:
    """A single executable stage within an adapter pipeline."""

    name: str
    handler: StageCallable

    def __call__(self, context: AdapterExecutionContext) -> AdapterExecutionContext:
        started_at = perf_counter()
        try:
            result = self.handler(context)
        except AdapterPluginError as exc:
            context.record_stage_result(
                self.name,
                started_at,
                perf_counter(),
                status="error",
                details={"error": str(exc)},
            )
            raise
        except Exception as exc:  # pragma: no cover - unexpected failures
            context.record_stage_result(
                self.name,
                started_at,
                perf_counter(),
                status="error",
                details={"error": str(exc), "exception": type(exc).__name__},
            )
            raise AdapterPluginError(f"Adapter stage '{self.name}' failed: {exc}") from exc

        if not isinstance(result, AdapterExecutionContext):
            context.record_stage_result(
                self.name,
                started_at,
                perf_counter(),
                status="error",
                details={"error": "Stage returned unexpected result"},
            )
            raise AdapterPluginError(
                f"Adapter stage '{self.name}' must return AdapterExecutionContext"
            )

        context.record_stage_result(self.name, started_at, perf_counter())
        return result


class AdapterPipeline:
    """Composable execution pipeline for adapter plugins."""

    def __init__(self, stages: Iterable[AdapterStage], *, name: str | None = None) -> None:
        stages_tuple: Tuple[AdapterStage, ...] = tuple(stages)
        if not stages_tuple:
            raise AdapterPluginError("Adapter pipelines must contain at least one stage")
        self._stages = stages_tuple
        self.name = name or "default"
        self._stage_names = tuple(stage.name for stage in stages_tuple)

    def execute(self, context: AdapterExecutionContext) -> AdapterExecutionContext:
        context.pipeline_name = self.name
        try:
            for stage in self._stages:
                context = stage(context)
            return context
        finally:
            context.finalize()

    def clone(self, *, name: str | None = None) -> "AdapterPipeline":
        return AdapterPipeline(self._stages, name=name or self.name)

    @property
    def stage_names(self) -> Tuple[str, ...]:
        return self._stage_names

    @classmethod
    def default(cls, plugin: Any, metadata: AdapterMetadata) -> "AdapterPipeline":
        return cls(
            (
                cls._fetch_stage(plugin, metadata),
                cls._parse_stage(plugin, metadata),
                cls._validate_stage(plugin, metadata),
            )
        )

    @staticmethod
    def _fetch_stage(plugin: Any, metadata: AdapterMetadata) -> AdapterStage:
        if not hasattr(plugin, "fetch"):
            raise AdapterPluginError(
                f"Plugin '{metadata.name}' does not implement required 'fetch' stage"
            )

        def _handler(state: AdapterExecutionState) -> AdapterExecutionState:
            response = plugin.fetch(state.request)
            if not isinstance(response, AdapterResponse):
                raise AdapterPluginError(
                    f"Adapter '{metadata.name}' fetch must return AdapterResponse, got {type(response)!r}"
                )
            return state.record_response(response)

        return AdapterStage("fetch", _handler)

    @staticmethod
    def _parse_stage(plugin: Any, metadata: AdapterMetadata) -> AdapterStage:
        if not hasattr(plugin, "parse"):
            raise AdapterPluginError(
                f"Plugin '{metadata.name}' does not implement required 'parse' stage"
            )

        def _handler(state: AdapterExecutionState) -> AdapterExecutionState:
            response = state.response
            if response is None:
                raise AdapterPluginError(
                    f"Adapter '{metadata.name}' parse stage requires a response from fetch"
                )
            parsed = plugin.parse(response, state.request)
            if not isinstance(parsed, AdapterResponse):
                raise AdapterPluginError(
                    f"Adapter '{metadata.name}' parse must return AdapterResponse, got {type(parsed)!r}"
                )
            return state.record_response(parsed)

        return AdapterStage("parse", _handler)

    @staticmethod
    def _validate_stage(plugin: Any, metadata: AdapterMetadata) -> AdapterStage:
        if not hasattr(plugin, "validate"):
            raise AdapterPluginError(
                f"Plugin '{metadata.name}' does not implement required 'validate' stage"
            )

        def _handler(state: AdapterExecutionState) -> AdapterExecutionState:
            response = state.response
            if response is None:
                raise AdapterPluginError(
                    f"Adapter '{metadata.name}' validate stage requires a response"
                )
            outcome = plugin.validate(response, state.request)
            if not isinstance(outcome, ValidationOutcome):
                raise AdapterPluginError(
                    f"Adapter '{metadata.name}' validate must return ValidationOutcome, got {type(outcome)!r}"
                )
            return state.record_validation(outcome)

        return AdapterStage("validate", _handler)


class AdapterPipelineFactory:
    """Factory responsible for constructing pipelines for registered plugins."""

    def build(self, plugin: Any, metadata: AdapterMetadata) -> AdapterPipeline:
        if hasattr(plugin, "build_pipeline"):
            pipeline = plugin.build_pipeline(metadata=metadata)
            if not isinstance(pipeline, AdapterPipeline):
                raise AdapterPluginError(
                    f"Adapter '{metadata.name}' build_pipeline must return AdapterPipeline"
                )
            return pipeline

        custom = getattr(plugin, "pipeline", None)
        if isinstance(custom, AdapterPipeline):
            return custom.clone()

        return AdapterPipeline.default(plugin, metadata)


__all__ = [
    "AdapterExecutionContext",
    "AdapterExecutionMetrics",
    "AdapterExecutionState",
    "AdapterPipeline",
    "AdapterPipelineFactory",
    "AdapterStage",
    "StageResult",
]
