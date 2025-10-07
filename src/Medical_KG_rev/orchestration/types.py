"""Shared types for orchestration pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .pipeline import PipelineContext


class PipelineStage(Protocol):
    """Protocol implemented by pipeline stages."""

    name: str
    timeout_ms: int | None

    def execute(self, context: "PipelineContext") -> "PipelineContext": ...


class StageConfig(BaseModel):
    """Declarative representation of a pipeline stage loaded from YAML."""

    name: str
    kind: str
    timeout_ms: int | None = Field(default=None, ge=1)
    options: dict[str, Any] = Field(default_factory=dict)


__all__ = ["PipelineStage", "StageConfig"]
