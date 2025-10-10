"""Runtime representations of registered adapter plugins."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .domains.metadata import DomainAdapterMetadata
from .errors import AdapterPluginError
from .models import (
    AdapterMetadata,
    AdapterRequest,
    AdapterResponse,
    ValidationOutcome,
)
from .pipeline import AdapterExecutionContext, AdapterExecutionMetrics, AdapterPipeline


@dataclass(slots=True)
class AdapterExecutionPlan:
    """Describes how a registered adapter will be executed."""

    pipeline: AdapterPipeline

    @property
    def name(self) -> str:
        return self.pipeline.name

    @property
    def stage_names(self) -> tuple[str, ...]:
        return self.pipeline.stage_names

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "stages": self.stage_names}


@dataclass(slots=True)
class AdapterInvocationResult:
    """Structured outcome for an adapter invocation."""

    adapter: AdapterMetadata
    request: AdapterRequest
    context: AdapterExecutionContext
    response: AdapterResponse | None
    validation: ValidationOutcome | None
    metrics: AdapterExecutionMetrics
    pipeline: Mapping[str, Any]
    extras: Mapping[str, Any]
    error: AdapterPluginError | None
    strict: bool

    @property
    def ok(self) -> bool:
        if self.error is not None:
            return False
        if self.validation is None:
            return True
        return self.validation.valid


@dataclass(slots=True)
class RegisteredAdapter:
    """Aggregate record describing a registered adapter plugin."""

    plugin: Any
    metadata: AdapterMetadata
    plan: AdapterExecutionPlan
    domain_metadata: DomainAdapterMetadata

    def new_context(self, request: AdapterRequest) -> AdapterExecutionContext:
        return AdapterExecutionContext(
            request=request,
            metadata=self.metadata,
            plugin=self.plugin,
            pipeline_name=self.plan.name,
        )

    def build_result(
        self,
        context: AdapterExecutionContext,
        *,
        strict: bool,
        error: AdapterPluginError | None,
    ) -> AdapterInvocationResult:
        response = context.response
        if response is not None:
            response.metadata = {**response.metadata, "pipeline": context.pipeline_name}
        return AdapterInvocationResult(
            adapter=self.metadata,
            request=context.request,
            context=context,
            response=response,
            validation=context.validation,
            metrics=context.metrics,
            pipeline=self.plan.describe(),
            extras=dict(context.extras),
            error=error,
            strict=strict,
        )


__all__ = [
    "AdapterExecutionPlan",
    "AdapterInvocationResult",
    "RegisteredAdapter",
]
