"""Execution pipeline primitives for adapter plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Iterable, Tuple
import time

from .errors import AdapterPluginError
from .models import (
    AdapterConfig,
    AdapterMetadata,
    AdapterRequest,
    AdapterResponse,
    ValidationOutcome,
)

StageCallable = Callable[["AdapterExecutionState"], "AdapterExecutionState"]


@dataclass(slots=True)
class AdapterStageTiming:
    """Timing information captured for a single stage execution."""

    name: str
    started_at: datetime
    duration_ms: float


@dataclass(slots=True)
class AdapterExecutionContext:
    """Immutable context shared across pipeline stages."""

    request: AdapterRequest
    metadata: AdapterMetadata
    base_config: AdapterConfig | None = None
    override_config: AdapterConfig | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    _config: AdapterConfig | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._config = self._merge_configs()

    @property
    def config(self) -> AdapterConfig | None:
        """Return the resolved configuration for this invocation."""

        return self._config

    @property
    def canonical_metadata(self) -> dict[str, Any]:
        """Canonical metadata injected into adapter responses."""

        return {
            "adapter": self.metadata.name,
            "adapter_version": self.metadata.version,
            "adapter_domain": self.metadata.domain.value,
            "correlation_id": self.request.correlation_id,
        }

    def _merge_configs(self) -> AdapterConfig | None:
        base = self.base_config
        override = self.override_config
        if override is None:
            return base
        if base is None:
            return override
        merged: dict[str, Any] = base.model_dump()
        merged.update(override.model_dump(exclude_unset=True))
        target_cls = base.__class__
        if base.__class__ is not override.__class__:
            target_cls = override.__class__
        return target_cls(**merged)

    @classmethod
    def from_request(
        cls,
        *,
        request: AdapterRequest,
        metadata: AdapterMetadata,
        base_config: AdapterConfig | None = None,
    ) -> "AdapterExecutionContext":
        return cls(
            request=request,
            metadata=metadata,
            base_config=base_config,
            override_config=request.config,
        )


@dataclass(slots=True)
class AdapterExecutionState:
    """Mutable state propagated through adapter pipeline stages."""

    context: AdapterExecutionContext
    plugin: Any
    response: AdapterResponse | None = None
    validation: ValidationOutcome | None = None
    timings: list[AdapterStageTiming] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def request(self) -> AdapterRequest:
        return self.context.request

    @property
    def metadata(self) -> AdapterMetadata:
        return self.context.metadata

    @property
    def config(self) -> AdapterConfig | None:
        return self.context.config

    def record_response(self, response: AdapterResponse) -> "AdapterExecutionState":
        self.response = response
        return self

    def record_validation(self, outcome: ValidationOutcome) -> "AdapterExecutionState":
        self.validation = outcome
        return self

    def record_timing(
        self, stage_name: str, started_at: datetime, duration_ms: float
    ) -> "AdapterExecutionState":
        self.timings.append(
            AdapterStageTiming(name=stage_name, started_at=started_at, duration_ms=duration_ms)
        )
        return self

    def add_extra(self, key: str, value: Any) -> "AdapterExecutionState":
        self.extras[key] = value
        return self

    def apply_canonical_metadata(self) -> None:
        if self.response is None:
            return
        for key, value in self.context.canonical_metadata.items():
            self.response.metadata.setdefault(key, value)

    def finalize(self) -> "AdapterExecutionState":
        if self.response is not None:
            self.apply_canonical_metadata()
        if self.validation is None:
            warnings: list[str] = []
            if self.response is not None:
                warnings = list(self.response.warnings)
            self.validation = ValidationOutcome(valid=True, warnings=warnings)
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
        raise AdapterPluginError(
            f"Validation failed for adapter '{self.metadata.name}': {errors}"
        )

    @property
    def total_duration_ms(self) -> float:
        return sum(timing.duration_ms for timing in self.timings)


@dataclass(slots=True)
class AdapterStage:
    """A single executable stage within an adapter pipeline."""

    name: str
    handler: StageCallable

    def __call__(self, state: AdapterExecutionState) -> AdapterExecutionState:
        started_at = datetime.now(UTC)
        start = time.perf_counter()
        try:
            new_state = self.handler(state)
        except AdapterPluginError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            raise AdapterPluginError(
                f"Adapter '{state.metadata.name}' stage '{self.name}' failed: {exc}"
            ) from exc
        duration_ms = (time.perf_counter() - start) * 1000
        new_state.record_timing(self.name, started_at, duration_ms)
        new_state.apply_canonical_metadata()
        return new_state


class AdapterPipeline:
    """Composable execution pipeline for adapter plugins."""

    def __init__(self, stages: Iterable[AdapterStage], *, name: str | None = None) -> None:
        stages_tuple: Tuple[AdapterStage, ...] = tuple(stages)
        if not stages_tuple:
            raise AdapterPluginError("Adapter pipelines must contain at least one stage")
        self._stages = stages_tuple
        self.name = name or "default"

    def execute(self, state: AdapterExecutionState) -> AdapterExecutionState:
        for stage in self._stages:
            state = stage(state)
        return state.finalize()

    def clone(self, *, name: str | None = None) -> "AdapterPipeline":
        return AdapterPipeline(self._stages, name=name or self.name)

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
    "AdapterExecutionState",
    "AdapterPipeline",
    "AdapterPipelineFactory",
    "AdapterStage",
    "AdapterStageTiming",
]

