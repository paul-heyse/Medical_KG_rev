"""Medical domain overlays aligning with FHIR resources."""
from __future__ import annotations

from typing import Literal, Optional, Sequence

from pydantic import Field, field_validator

from ..ir import Document, IRBaseModel


class ResearchStudy(IRBaseModel):
    """Subset of the FHIR ResearchStudy resource used by the platform."""

    identifier: str
    title: str
    status: Literal["active", "completed", "withdrawn", "suspended", "unknown"]
    phase: Optional[str] = None
    condition: Sequence[str] = Field(default_factory=tuple)
    enrollment_count: Optional[int] = Field(default=None, ge=0)


class EvidenceAssessment(IRBaseModel):
    """FHIR Evidence summary used to contextualize claims."""

    certainty: Literal["high", "moderate", "low", "very-low"] = "low"
    certainty_rationale: Optional[str] = None
    quality_rating: Literal["good", "fair", "poor"] = "fair"


class MedicalDocument(Document):
    """Document overlay enriched with medical metadata."""

    domain: Literal["medical"] = "medical"
    research_study: Optional[ResearchStudy] = None
    evidence_assessments: Sequence[EvidenceAssessment] = Field(default_factory=tuple)

    @field_validator("evidence_assessments")
    @classmethod
    def _limit_assessments(cls, value: Sequence[EvidenceAssessment]) -> Sequence[EvidenceAssessment]:
        if len(value) > 10:
            raise ValueError("MedicalDocument supports at most 10 evidence assessments")
        return tuple(value)
