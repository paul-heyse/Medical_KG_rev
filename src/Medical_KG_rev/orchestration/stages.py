"""Registries for mapping declarative pipeline stages to implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from .pipeline import PipelineStage, StageConfig


class StageBuilder(Protocol):
    """Callable that materialises a pipeline stage from configuration."""

    def __call__(self, stage: StageConfig, options: Mapping[str, object]) -> PipelineStage: ...


@dataclass(slots=True)
class StageRegistry:
    """Registry mapping stage kinds to builder callables."""

    _builders: dict[str, StageBuilder] = field(default_factory=dict)

    def register(self, kind: str, builder: StageBuilder) -> None:
        """Register a builder for the supplied stage kind."""

        self._builders[kind] = builder

    def unregister(self, kind: str) -> None:
        """Remove a previously registered builder if present."""

        self._builders.pop(kind, None)

    def build(self, stage: StageConfig, options: Mapping[str, object]) -> PipelineStage:
        """Materialise a stage instance for the provided declarative config."""

        try:
            builder = self._builders[stage.kind]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unsupported stage kind '{stage.kind}'") from exc
        return builder(stage, options)

    def clone(self) -> StageRegistry:
        """Return a shallow copy of the registry."""

        return StageRegistry(dict(self._builders))


__all__ = ["StageBuilder", "StageRegistry"]
