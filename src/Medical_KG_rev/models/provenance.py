"""Provenance models that capture data lineage for all ingested artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProvenanceBaseModel(BaseModel):
    """Base configuration shared across provenance models."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataSource(ProvenanceBaseModel):
    """Represents a logical data provider such as ClinicalTrials.gov."""

    id: str
    name: str
    url: Optional[str] = None
    version: str = Field(default="v1")
    credentials_required: bool = Field(default=False)


class ExtractionActivity(ProvenanceBaseModel):
    """Metadata describing how a particular entity/evidence was extracted."""

    id: str
    performed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actor: str = Field(description="System or user responsible for extraction")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    data_source: DataSource
    tool_version: str = Field(default="v1")

    @field_validator("performed_at")
    @classmethod
    def _ensure_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("performed_at must include timezone information")
        return value.astimezone(timezone.utc)
