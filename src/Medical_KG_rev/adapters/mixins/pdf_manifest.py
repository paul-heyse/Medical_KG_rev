"""Utilities for building PDF manifests from adapter responses."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any
from urllib.parse import urlparse, urlunparse

from Medical_KG_rev.adapters.base import AdapterContext
from Medical_KG_rev.adapters.interfaces.pdf import PdfAssetManifest, PdfManifest
from Medical_KG_rev.models import Document


class PdfManifestMixin:
    """Mixin providing helpers for normalising adapter PDF manifests."""

    pdf_capabilities: Sequence[str] = ("pdf",)

    def build_pdf_manifest(
        self,
        *,
        connector: str,
        assets: Iterable[Mapping[str, Any]],
        retrieved_at: datetime | None = None,
        polite_headers: Mapping[str, str] | None = None,
    ) -> PdfManifest:
        """Create a :class:`PdfManifest` from raw asset dictionaries."""

        normalised_assets = self._normalise_assets(assets)
        timestamp = retrieved_at or datetime.now(timezone.utc)
        headers = MappingProxyType(dict(polite_headers or {}))
        return PdfManifest(
            connector=connector,
            assets=tuple(normalised_assets),
            retrieved_at=timestamp,
            polite_headers=headers,
        )

    def attach_manifest_to_documents(
        self,
        documents: Sequence[Document],
        manifest: PdfManifest,
    ) -> Sequence[Document]:
        """Attach manifest metadata to the provided documents in place."""

        urls = list(manifest.pdf_urls())
        for document in documents:
            metadata = dict(document.metadata)
            metadata["pdf_manifest"] = manifest.as_metadata()
            if urls:
                metadata.setdefault("pdf_urls", urls)
                metadata.setdefault("document_type", "pdf")
            document.metadata = metadata
        return documents

    def iter_pdf_candidates(
        self,
        documents: Sequence[Document],
        *,
        context: AdapterContext | None = None,
    ) -> Iterable[PdfAssetManifest]:
        """Yield manifest entries for the given documents."""

        for document in documents:
            yield from self._iter_manifest_assets(document)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalise_assets(
        self, assets: Iterable[Mapping[str, Any]]
    ) -> list[PdfAssetManifest]:
        deduped: OrderedDict[tuple[str, str], PdfAssetManifest] = OrderedDict()
        for asset in assets:
            url = self._normalise_url(asset.get("url"))
            if not url:
                continue
            version = self._normalise_version(asset.get("version"))
            key = (url, version or "")
            if key in deduped:
                continue
            manifest = PdfAssetManifest(
                url=url,
                landing_page_url=self._normalise_url(asset.get("landing_page_url")),
                license=self._normalise_license(asset.get("license")),
                version=version,
                source=self._normalise_source(asset.get("source")),
                checksum_hint=self._normalise_checksum(asset.get("checksum_hint")),
                is_open_access=self._normalise_flag(asset.get("is_open_access")),
                content_type=self._normalise_content_type(asset.get("content_type")),
            )
            deduped[key] = manifest
        return list(deduped.values())

    @staticmethod
    def _normalise_url(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        candidate = value.strip()
        if not candidate:
            return None
        parsed = urlparse(candidate)
        if parsed.scheme and parsed.netloc:
            scheme = parsed.scheme.lower()
            if scheme == "http":
                scheme = "https"
            path = parsed.path or ""
            normalised = urlunparse(
                (
                    scheme,
                    parsed.netloc.strip(),
                    path,
                    "",
                    parsed.query,
                    "",
                )
            )
            return normalised
        return candidate

    @staticmethod
    def _normalise_license(value: Any) -> str | None:
        if isinstance(value, str):
            candidate = value.strip()
            return candidate or None
        return None

    @staticmethod
    def _normalise_version(value: Any) -> str | None:
        if isinstance(value, str):
            candidate = value.strip()
            return candidate or None
        if isinstance(value, (int, float)):
            return str(value)
        return None

    @staticmethod
    def _normalise_source(value: Any) -> str | None:
        if isinstance(value, str):
            candidate = value.strip()
            return candidate or None
        return None

    @staticmethod
    def _normalise_checksum(value: Any) -> str | None:
        if isinstance(value, str):
            candidate = value.strip()
            return candidate or None
        return None

    @staticmethod
    def _normalise_flag(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        return None

    @staticmethod
    def _normalise_content_type(value: Any) -> str | None:
        if isinstance(value, str):
            candidate = value.strip().lower()
            return candidate or None
        return None

    def _iter_manifest_assets(
        self, document: Document
    ) -> Iterable[PdfAssetManifest]:
        manifest = document.metadata.get("pdf_manifest")
        if not isinstance(manifest, Mapping):
            return
        assets = manifest.get("assets", [])
        if not isinstance(assets, Sequence):
            return
        for asset in assets:
            if not isinstance(asset, Mapping):
                continue
            url = asset.get("url")
            if not isinstance(url, str):
                continue
            yield PdfAssetManifest(
                url=url,
                landing_page_url=asset.get("landing_page_url"),
                license=asset.get("license"),
                version=asset.get("version"),
                source=asset.get("source"),
                checksum_hint=asset.get("checksum_hint"),
                is_open_access=asset.get("is_open_access"),
                content_type=asset.get("content_type"),
            )


__all__ = ["PdfManifestMixin"]
