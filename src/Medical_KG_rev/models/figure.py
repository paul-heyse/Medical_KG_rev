from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Figure(BaseModel):
    """Metadata describing a figure extracted from a PDF."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    page: int = Field(ge=1)
    image_path: str
    caption: str | None = None
    bbox: tuple[float, float, float, float] | None = Field(default=None)
    figure_type: str | None = Field(default=None)
    mime_type: str | None = Field(default=None)
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ["Figure"]
