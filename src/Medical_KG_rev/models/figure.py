from __future__ import annotations

from pydantic import Field

from .artifact import Artifact


class Figure(Artifact):
    """Metadata describing a figure extracted from a PDF."""

    model_config = Artifact.model_config

    image_path: str
    caption: str | None = None
    figure_type: str | None = Field(default=None)
    mime_type: str | None = Field(default=None)
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)


__all__ = ["Figure"]
