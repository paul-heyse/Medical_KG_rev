"""Shared structures for page-level artefacts extracted from documents."""

from __future__ import annotations

from typing import Any, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class Artifact(BaseModel):
    """Base model for structured artefacts extracted from documents."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    id: str
    page: int = Field(ge=1)
    bbox: tuple[float, float, float, float] | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def with_metadata(self, **updates: Any) -> Self:
        """Return a copy with metadata merged with provided keyword arguments."""

        merged = dict(self.metadata)
        merged.update({key: value for key, value in updates.items() if value is not None})
        return self.model_copy(update={"metadata": merged})


ArtifactType = TypeVar("ArtifactType", bound="Artifact")


__all__ = ["Artifact", "ArtifactType"]

