"""Entity graph models for the federated knowledge graph."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field, field_validator

from .ir import IRBaseModel, Span
from .provenance import DataSource, ExtractionActivity


class Entity(IRBaseModel):
    """A real-world concept mentioned in a document."""

    id: str
    type: str = Field(description="Entity type (e.g. Condition, Drug)")
    canonical_name: str
    aliases: Sequence[str] = Field(default_factory=tuple)
    spans: Sequence[Span] = Field(default_factory=tuple)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("aliases")
    @classmethod
    def _deduplicate_aliases(cls, value: Sequence[str]) -> Sequence[str]:
        seen_keys: set[str] = set()
        deduped: list[str] = []
        for alias in value:
            key = alias.lower()
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(alias)
        return tuple(deduped)


class Claim(IRBaseModel):
    """A semantic assertion extracted from a document."""

    id: str
    subject_id: str
    predicate: str
    object_id: str
    qualifiers: dict[str, str] = Field(default_factory=dict)
    extraction: ExtractionActivity


class Evidence(IRBaseModel):
    """Links claims/entities back to the source material."""

    id: str
    document_id: str
    span: Span
    statement: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    supporting_entities: Sequence[str] = Field(default_factory=tuple)
    extraction: ExtractionActivity


__all__ = [
    "Claim",
    "DataSource",
    "Entity",
    "Evidence",
    "ExtractionActivity",
]
