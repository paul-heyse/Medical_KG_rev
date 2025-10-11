"""Placeholder ingestion pipeline helpers."""

from __future__ import annotations

from typing import Any

from Medical_KG_rev.orchestration.stages import StageFailure
from Medical_KG_rev.orchestration.stages.types import PipelineContext


def handle_stage_failure(exc: Exception) -> StageFailure:
    """Wrap a generic exception into a :class:`StageFailure` instance."""
    message = str(exc)
    error_type = "gpu_unavailable" if "gpu" in message.lower() else "validation"
    return StageFailure(message, error_type=error_type)


def execute_pipeline(context: PipelineContext) -> Any:
    """Execute the ingestion pipeline (placeholder implementation)."""
    raise StageFailure("Ingestion pipeline is not available in this build", error_type="unimplemented")


__all__ = ["execute_pipeline", "handle_stage_failure"]
