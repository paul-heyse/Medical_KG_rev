"""Legal domain overlay aligned with LegalDocML concepts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import Field, field_validator

from ..ir import Document, IRBaseModel


class LegalReference(IRBaseModel):
    """Represents a citation to a legal authority."""

    target: str
    locator: str | None = None
    text: str | None = None


class LegalClause(IRBaseModel):
    """Section of a legal document with associated references."""

    id: str
    title: str | None = None
    references: Sequence[LegalReference] = Field(default_factory=tuple)

    @field_validator("references")
    @classmethod
    def _deduplicate_refs(cls, value: Sequence[LegalReference]) -> Sequence[LegalReference]:
        seen = set()
        unique_refs = []
        for ref in value:
            key = (ref.target, ref.locator)
            if key not in seen:
                seen.add(key)
                unique_refs.append(ref)
        return tuple(unique_refs)


class LegalDocument(Document):
    """Document overlay enriched with LegalDocML specific fields."""

    domain: Literal["legal"] = "legal"
    jurisdiction: str | None = None
    clauses: Sequence[LegalClause] = Field(default_factory=tuple)

    @field_validator("jurisdiction")
    @classmethod
    def _normalize_jurisdiction(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.upper()
