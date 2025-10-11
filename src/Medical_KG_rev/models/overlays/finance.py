"""Finance domain overlay aligned with XBRL constructs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator

from ..ir import Document, IRBaseModel



class XBRLContext(IRBaseModel):
    """Represents an XBRL context definition."""

    identifier: str
    period_start: str | None = None
    period_end: str | None = None
    entity_scheme: str
    entity_identifier: str


class FinancialFact(IRBaseModel):
    """An individual XBRL fact extracted from a document."""

    concept: str
    value: str
    unit: str | None = None
    context_ref: str


class FinancialDocument(Document):
    """Document overlay that captures key XBRL constructs."""

    domain: Literal["finance"] = "finance"
    contexts: Sequence[XBRLContext] = Field(default_factory=tuple)
    facts: Sequence[FinancialFact] = Field(default_factory=tuple)

    @field_validator("facts")
    @classmethod
    def _validate_contexts(
        cls, value: Sequence[FinancialFact], info: ValidationInfo
    ) -> Sequence[FinancialFact]:
        contexts = {context.identifier for context in info.data.get("contexts", [])}
        invalid = [fact.context_ref for fact in value if fact.context_ref not in contexts]
        if invalid:
            raise ValueError(f"Facts reference undefined contexts: {sorted(set(invalid))}")
        return tuple(value)
