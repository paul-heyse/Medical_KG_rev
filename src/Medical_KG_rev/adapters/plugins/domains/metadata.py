"""Domain-specific metadata extensions for adapter plugins."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from Medical_KG_rev.adapters.plugins.models import AdapterDomain, AdapterMetadata


class DomainAdapterMetadata(AdapterMetadata):
    """Base metadata type extended with optional domain descriptors."""

    dataset: str | None = Field(
        default=None,
        description="Logical dataset associated with the adapter output.",
    )
    compliance: list[str] = Field(
        default_factory=list,
        description="Compliance regimes satisfied by the adapter (e.g. HIPAA).",
    )

    def with_entry_point(self, entry_point: str) -> "DomainAdapterMetadata":
        meta = self.model_copy(update={"entry_point": entry_point})
        return meta


class BiomedicalAdapterMetadata(DomainAdapterMetadata):
    """Metadata for biomedical adapters."""

    domain: AdapterDomain = Field(default=AdapterDomain.BIOMEDICAL, frozen=True)
    data_products: list[str] = Field(default_factory=list)


class FinancialAdapterMetadata(DomainAdapterMetadata):
    """Metadata for financial adapters."""

    domain: AdapterDomain = Field(default=AdapterDomain.FINANCIAL, frozen=True)
    reporting_frequency: str | None = None


class LegalAdapterMetadata(DomainAdapterMetadata):
    """Metadata for legal adapters."""

    domain: AdapterDomain = Field(default=AdapterDomain.LEGAL, frozen=True)
    jurisdictions: list[str] = Field(default_factory=list)


def as_metadata(metadata: AdapterMetadata) -> DomainAdapterMetadata:
    """Ensure metadata is represented as :class:`DomainAdapterMetadata`."""

    if isinstance(metadata, DomainAdapterMetadata):
        return metadata
    payload: dict[str, Any] = metadata.model_dump()
    return DomainAdapterMetadata(**payload)
