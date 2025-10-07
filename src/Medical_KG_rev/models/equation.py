from __future__ import annotations

from pydantic import Field

from .artifact import StructuredArtifact


class Equation(StructuredArtifact):
    """Structured representation of an extracted equation."""

    model_config = StructuredArtifact.model_config

    latex: str
    mathml: str | None = Field(default=None)
    display: bool = Field(default=True)


__all__ = ["Equation"]
