"""Runtime representations of registered adapter plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .domains.metadata import DomainAdapterMetadata
from .pipeline import (
    AdapterExecutionContext,
    AdapterExecutionState,
    AdapterPipeline,
    AdapterStageTiming,
)
from .models import AdapterConfig, AdapterMetadata, AdapterRequest, AdapterResponse, ValidationOutcome


@dataclass(slots=True)
class AdapterInvocationResult:
    """Structured execution result enriched with telemetry."""

    metadata: AdapterMetadata
    response: AdapterResponse
    validation: ValidationOutcome
    timings: tuple[AdapterStageTiming, ...]
    extras: Mapping[str, Any]

    @property
    def successful(self) -> bool:
        return self.validation.valid

    @property
    def total_duration_ms(self) -> float:
        return sum(timing.duration_ms for timing in self.timings)


@dataclass(slots=True)
class AdapterExecutionPlan:
    """Execution plan binding metadata to a pipeline."""

    metadata: AdapterMetadata
    pipeline: AdapterPipeline

    def build_state(
        self,
        plugin: Any,
        request: AdapterRequest,
        *,
        base_config: AdapterConfig | None = None,
    ) -> AdapterExecutionState:
        context = AdapterExecutionContext.from_request(
            request=request,
            metadata=self.metadata,
            base_config=base_config,
        )
        return AdapterExecutionState(context=context, plugin=plugin)


@dataclass(slots=True)
class RegisteredAdapter:
    """Aggregate record describing a registered adapter plugin."""

    plugin: Any
    metadata: AdapterMetadata
    pipeline: AdapterPipeline
    domain_metadata: DomainAdapterMetadata
    plan: AdapterExecutionPlan = field(init=False)
    base_config: AdapterConfig | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.plan = AdapterExecutionPlan(metadata=self.metadata, pipeline=self.pipeline)
        self.base_config = getattr(self.plugin, "config", None)

    def execute(self, request: AdapterRequest) -> AdapterExecutionState:
        current_config = getattr(self.plugin, "config", None)
        if current_config is not None and current_config is not self.base_config:
            self.base_config = current_config
        state = self.plan.build_state(self.plugin, request, base_config=self.base_config)
        override = state.context.config
        if override is not None and override is not self.base_config:
            had_attribute = hasattr(self.plugin, "config")
            original = getattr(self.plugin, "config", None)
            try:
                setattr(self.plugin, "config", override)
                return self.pipeline.execute(state)
            finally:
                if had_attribute:
                    setattr(self.plugin, "config", original)
                else:  # pragma: no cover - defensive
                    if hasattr(self.plugin, "config"):
                        delattr(self.plugin, "config")
        return self.pipeline.execute(state)

    def build_result(self, state: AdapterExecutionState) -> AdapterInvocationResult:
        response = state.ensure_response()
        validation = state.validation or ValidationOutcome.success()
        return AdapterInvocationResult(
            metadata=self.metadata,
            response=response,
            validation=validation,
            timings=tuple(state.timings),
            extras=dict(state.extras),
        )


__all__ = [
    "AdapterExecutionPlan",
    "AdapterInvocationResult",
    "RegisteredAdapter",
]

