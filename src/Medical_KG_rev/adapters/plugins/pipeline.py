"""Execution pipeline primitives for adapter plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Tuple

from .errors import AdapterPluginError
from .models import AdapterMetadata, AdapterRequest, AdapterResponse, ValidationOutcome

StageCallable = Callable[["AdapterExecutionState"], "AdapterExecutionState"]


@dataclass(slots=True)
class AdapterExecutionState:
    """Mutable state propagated through adapter pipeline stages."""

    request: AdapterRequest
    metadata: AdapterMetadata
    plugin: Any
    response: AdapterResponse | None = None
    validation: ValidationOutcome | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def record_response(self, response: AdapterResponse) -> "AdapterExecutionState":
        self.response = response
        return self

    def record_validation(self, outcome: ValidationOutcome) -> "AdapterExecutionState":
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
        raise AdapterPluginError(
            f"Validation failed for adapter '{self.metadata.name}': {errors}"
        )


@dataclass(slots=True)
class AdapterStage:
    """A single executable stage within an adapter pipeline."""

    name: str
    handler: StageCallable

    def __call__(self, state: AdapterExecutionState) -> AdapterExecutionState:
        return self.handler(state)


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
        return state

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
    "AdapterExecutionState",
    "AdapterPipeline",
    "AdapterPipelineFactory",
    "AdapterStage",
]

