"""Profile configuration models for the chunking subsystem."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field



class Profile(BaseModel):
    """Declarative configuration describing a chunking profile."""

    name: str
    domain: str
    chunker_type: str = Field(alias="chunker_type")
    target_tokens: int = Field(default=512, ge=1)
    overlap_tokens: int = Field(default=50, ge=0)
    respect_boundaries: list[str] = Field(default_factory=list)
    sentence_splitter: str = Field(default="syntok")
    preserve_tables_as_html: bool = Field(default=True)
    filters: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)
