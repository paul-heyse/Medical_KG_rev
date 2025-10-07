"""Registries for mapping declarative pipeline stages to implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from Medical_KG_rev.utils.errors import ProblemDetail

from ..types import PipelineStage, StageConfig


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


class StageFailure(RuntimeError):
    """Wraps a stage failure with retry metadata and RFC 7807 details."""

    def __init__(
        self,
        message: str,
        *,
        status: int = 500,
        detail: str | None = None,
        stage: str | None = None,
        error_type: str | None = None,
        retriable: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.retriable = retriable
        self.error_type = error_type or ("transient" if retriable else "permanent")
        self.problem = ProblemDetail(
            title=message,
            status=status,
            detail=detail,
            extra=extra or {},
        )


__all__ = ["StageBuilder", "StageRegistry", "StageFailure"]
