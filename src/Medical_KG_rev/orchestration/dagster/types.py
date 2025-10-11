"""Dagster type definitions used by the orchestration runtime."""

from __future__ import annotations

from dagster import PythonObjectDagsterType, TypeCheck

from Medical_KG_rev.orchestration.stages.contracts import PipelineState


def _pipeline_state_type_check(_context, value: object) -> TypeCheck:
    if not isinstance(value, PipelineState):
        return TypeCheck(
            success=False,
            description=f"Expected PipelineState instance, received {type(value)!r}",
        )
    try:
        value.ensure_tenant_scope(value.context.tenant_id)
    except Exception as exc:  # pragma: no cover - defensive guard
        return TypeCheck(success=False, description=str(exc))
    return TypeCheck(success=True)


PIPELINE_STATE_DAGSTER_TYPE = PythonObjectDagsterType(
    python_type=PipelineState,
    name="PipelineState",
    description=(
        "Typed orchestration state capturing stage outputs, metrics, and"
        " lifecycle checkpoints for Dagster runs."
    ),
)


__all__ = ["PIPELINE_STATE_DAGSTER_TYPE"]
