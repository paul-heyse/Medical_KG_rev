"""Legal domain adapter plugins."""

from __future__ import annotations

from datetime import datetime

from Medical_KG_rev.adapters.plugins.base import BaseAdapterPlugin
from Medical_KG_rev.adapters.plugins.models import (
    AdapterConfig,
    AdapterRequest,
    AdapterResponse,
    ValidationOutcome,
)

from ..metadata import LegalAdapterMetadata


class LegalCaseConfig(AdapterConfig):
    """Configuration for the legal precedent adapter."""

    jurisdiction: str = "us"


class LegalPrecedentAdapterPlugin(BaseAdapterPlugin):
    """Synthetic adapter that returns precedent summaries for testing."""

    config_model = LegalCaseConfig
    metadata = LegalAdapterMetadata(
        name="legal-precedent",
        version="0.1.0",
        summary="Synthetic legal precedent adapter",
        capabilities=["precedent"],
        maintainer="Legal Data Team",
        dataset="legal_precedent",
        jurisdictions=["US", "CA"],
        compliance=["GDPR"],
    )

    def fetch(self, request: AdapterRequest) -> AdapterResponse:
        case_number = request.parameters.get("case_number", "2024-XYZ")
        jurisdiction = request.parameters.get("jurisdiction", self.config.jurisdiction)
        item = {
            "case_number": case_number,
            "jurisdiction": jurisdiction,
            "summary": f"Synthetic summary for case {case_number}",
            "decided_at": datetime.utcnow().isoformat(),
        }
        return AdapterResponse(items=[item])

    def parse(self, response: AdapterResponse, request: AdapterRequest) -> AdapterResponse:
        response.metadata.setdefault("jurisdiction", self.config.jurisdiction)
        return response

    def validate(self, response: AdapterResponse, request: AdapterRequest) -> ValidationOutcome:
        if not response.items:
            return ValidationOutcome.failure("No precedent returned")
        return ValidationOutcome(valid=True)


__all__ = [
    "LegalCaseConfig",
    "LegalPrecedentAdapterPlugin",
]
