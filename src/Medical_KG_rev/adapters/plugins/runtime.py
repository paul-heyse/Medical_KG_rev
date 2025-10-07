"""Runtime representations of registered adapter plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .pipeline import AdapterExecutionState, AdapterPipeline
from .models import AdapterMetadata, AdapterRequest
from .domains.metadata import DomainAdapterMetadata


@dataclass(slots=True)
class RegisteredAdapter:
    """Aggregate record describing a registered adapter plugin."""

    plugin: Any
    metadata: AdapterMetadata
    pipeline: AdapterPipeline
    domain_metadata: DomainAdapterMetadata

    def new_state(self, request: AdapterRequest) -> AdapterExecutionState:
        return AdapterExecutionState(request=request, metadata=self.metadata, plugin=self.plugin)


__all__ = ["RegisteredAdapter"]

