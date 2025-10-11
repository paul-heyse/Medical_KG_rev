"""Type definitions for pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PipelineContext:
    """Context for pipeline execution."""

    tenant_id: str
    operation: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PipelineState:
    """State of a pipeline execution."""

    stage: str
    data: Any
    metadata: Dict[str, Any]
    status: str = "pending"

    def update(self, **kwargs) -> PipelineState:
        """Update pipeline state."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
