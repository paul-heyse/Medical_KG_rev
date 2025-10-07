from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Equation(BaseModel):
    """Structured representation of an extracted equation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    page: int = Field(ge=1)
    latex: str
    mathml: str | None = Field(default=None)
    bbox: tuple[float, float, float, float] | None = Field(default=None)
    display: bool = Field(default=True)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ["Equation"]
