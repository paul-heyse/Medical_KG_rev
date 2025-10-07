"""Shared structures for page-level artefacts extracted from documents."""

from __future__ import annotations

from typing import Any, Mapping, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class Artifact(BaseModel):
    """Base model for structured artefacts extracted from documents."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    id: str
    page: int = Field(ge=1)
    bbox: tuple[float, float, float, float] | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StructuredArtifact(Artifact):
    """Provides immutable update helpers for MinerU artefacts."""

    def merge_metadata(
        self, updates: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> Self:
        """Merge metadata with the provided values, returning a new instance."""

        merged = dict(self.metadata)
        for container in (updates, kwargs):
            if not container:
                continue
            for key, value in container.items():
                if value is None:
                    continue
                merged[str(key)] = value
        return self.model_copy(update={"metadata": merged})

    def replace(self, **updates: Any) -> Self:
        """Return a copy with the specified fields replaced."""

        if not updates:
            return self
        return self.model_copy(update=updates)

    def enrich(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        **updates: Any,
    ) -> Self:
        """Return a copy with metadata merged and any additional fields updated."""

        payload = dict(updates)
        extra_metadata = payload.pop("metadata", None)
        if extra_metadata is not None:
            if metadata is None:
                metadata = extra_metadata  # type: ignore[assignment]
            elif isinstance(extra_metadata, Mapping):
                merged_meta = dict(metadata)
                merged_meta.update(extra_metadata)
                metadata = merged_meta
        if metadata:
            payload["metadata"] = {
                **self.metadata,
                **{str(k): v for k, v in metadata.items() if v is not None},
            }
        if not payload:
            return self
        return self.model_copy(update=payload)

    def with_metadata(self, **updates: Any) -> Self:
        """Backward-compatible helper that merges metadata values immutably."""

        return self.merge_metadata(updates)


ArtifactType = TypeVar("ArtifactType", bound="StructuredArtifact")


__all__ = ["Artifact", "ArtifactType", "StructuredArtifact"]

