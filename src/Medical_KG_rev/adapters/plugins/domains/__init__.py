"""Domain-specific adapters and metadata helpers."""

from .metadata import (
    BiomedicalAdapterMetadata,
    DomainAdapterMetadata,
    FinancialAdapterMetadata,
    LegalAdapterMetadata,
)
from .registry import DomainAdapterRegistry

__all__ = [
    "BiomedicalAdapterMetadata",
    "DomainAdapterMetadata",
    "DomainAdapterRegistry",
    "FinancialAdapterMetadata",
    "LegalAdapterMetadata",
]
