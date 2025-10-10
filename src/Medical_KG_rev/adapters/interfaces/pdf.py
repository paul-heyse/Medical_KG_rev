"""PDF ingestion interfaces shared across adapters.

This module defines the manifest dataclasses and the protocol adapters must
implement to participate in the PDF ingestion workflow.  The abstraction keeps
adapters lightweight while ensuring downstream orchestration stages can rely on
consistent metadata describing downloadable assets.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from Medical_KG_rev.adapters.base import AdapterContext
from Medical_KG_rev.models import Document


@dataclass(frozen=True)
class PdfAssetManifest:
    """Normalised description of a single downloadable PDF asset."""

    url: str
    landing_page_url: str | None = None
    license: str | None = None
    version: str | None = None
    source: str | None = None
    checksum_hint: str | None = None
    is_open_access: bool | None = None
    content_type: str | None = None

    def as_metadata(self) -> Mapping[str, object | None]:
        """Render the manifest entry as metadata for document storage."""
        return MappingProxyType(
            {
                "url": self.url,
                "landing_page_url": self.landing_page_url,
                "license": self.license,
                "version": self.version,
                "source": self.source,
                "checksum_hint": self.checksum_hint,
                "is_open_access": self.is_open_access,
                "content_type": self.content_type,
            }
        )


@dataclass(frozen=True)
class PdfManifest:
    """Collection of manifest entries emitted by an adapter."""

    connector: str
    assets: tuple[PdfAssetManifest, ...]
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    polite_headers: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def as_metadata(self) -> Mapping[str, object]:
        """Convert the manifest to a serialisable mapping."""
        return MappingProxyType(
            {
                "connector": self.connector,
                "retrieved_at": self.retrieved_at.isoformat(),
                "assets": [asset.as_metadata() for asset in self.assets],
                "polite_headers": dict(self.polite_headers),
            }
        )

    def pdf_urls(self) -> tuple[str, ...]:
        """Expose the ordered list of URLs for backward compatibility."""
        return tuple(asset.url for asset in self.assets)


@runtime_checkable
class PdfCapableAdapter(Protocol):
    """Protocol implemented by adapters that surface downloadable PDFs."""

    pdf_capabilities: Sequence[str]

    def iter_pdf_candidates(
        self,
        documents: Sequence[Document],
        *,
        context: AdapterContext | None = None,
    ) -> Iterable[PdfAssetManifest]:
        """Yield manifest entries for the provided documents."""

    def polite_headers(self) -> Mapping[str, str]:
        """Return polite pool headers associated with the adapter."""


__all__ = [
    "PdfAssetManifest",
    "PdfCapableAdapter",
    "PdfManifest",
]
