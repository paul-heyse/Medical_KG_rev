"""Illustrative plugin demonstrating custom stage registration."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition
from Medical_KG_rev.orchestration.dagster.stage_registry import (
    StageMetadata,
    StageRegistration,
)
from Medical_KG_rev.orchestration.stages.contracts import StageContext


def _handle_output(state: dict[str, Any], stage_name: str, output: Iterable[int]) -> None:
    values = list(output)
    state.setdefault("example", {})[stage_name] = values


def _count_output(output: Iterable[int]) -> int:
    if isinstance(output, Iterable):
        return len(list(output))
    return 0


@dataclass(slots=True)
class ExampleTransformStage:
    """Simple stage that scales numeric inputs by a configured factor."""

    name: str
    factor: int = 2

    def execute(self, ctx: StageContext, upstream: Iterable[int]) -> list[int]:
        del ctx  # context unused in the example implementation
        return [int(value) * self.factor for value in upstream]


def register_example_stage() -> StageRegistration:
    """Return a :class:`StageRegistration` for the example transform stage."""

    def _builder(definition: StageDefinition) -> ExampleTransformStage:
        factor = int(definition.config.get("factor", 2)) if definition.config else 2
        return ExampleTransformStage(name=definition.name, factor=factor)

    metadata = StageMetadata(
        stage_type="example-transform",
        state_key="example",
        output_handler=_handle_output,
        output_counter=_count_output,
        description="Scales integer inputs using a configurable multiplier",
    )
    return StageRegistration(metadata=metadata, builder=_builder)
