from __future__ import annotations

from pydantic import Field

from .artifact import Artifact


class Equation(Artifact):
    """Structured representation of an extracted equation."""

    model_config = Artifact.model_config

    latex: str
    mathml: str | None = Field(default=None)
    display: bool = Field(default=True)


__all__ = ["Equation"]
