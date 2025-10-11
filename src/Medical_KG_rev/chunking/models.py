"""Data models shared across chunkers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


Granularity = Literal["window", "paragraph", "section", "document", "table"]


class Chunk(BaseModel):
    """Representation of a coherent text span produced by a chunker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(pattern=r"^[\w:.-]+$")
    doc_id: str
    tenant_id: str
    body: str = Field(min_length=1)
    title_path: tuple[str, ...] = Field(default_factory=tuple)
    section: str | None = None
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    granularity: Granularity
    chunker: str
    chunker_version: str
    page_no: int | None = Field(default=None, ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_offsets(self) -> Chunk:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        if self.granularity not in {"window", "paragraph", "section", "document", "table"}:
            raise ValueError(f"Unsupported granularity '{self.granularity}'")
        return self

    @computed_field  # type: ignore[misc]
    @property
    def length(self) -> int:
        return len(self.body)


class ChunkerConfig(BaseModel):
    """Runtime configuration for instantiating a chunker."""

    model_config = ConfigDict(extra="forbid")

    name: str
    granularity: Granularity | None = None
    params: dict[str, Any] = Field(default_factory=dict)
