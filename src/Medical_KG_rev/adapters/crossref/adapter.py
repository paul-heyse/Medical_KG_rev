"""Crossref adapter for bibliographic metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from Medical_KG_rev.adapters.base import AdapterContext, BaseAdapter
from Medical_KG_rev.adapters.mixins import PdfManifestMixin
from Medical_KG_rev.config.settings import ConnectorPdfSettings, get_settings
from Medical_KG_rev.models import Block, BlockType, Document, Section
from Medical_KG_rev.utils.http_client import (
    BackoffStrategy,
    CircuitBreakerConfig,
    HttpClient,
    RateLimitConfig,
    RetryConfig,
)
from Medical_KG_rev.utils.identifiers import build_document_id
from Medical_KG_rev.utils.validation import validate_doi


def _require_parameter(context: AdapterContext, key: str) -> str:
    """Extract and validate a required parameter from context."""
    try:
        value = context.parameters[key]
    except KeyError as exc:
        raise ValueError(f"Missing required parameter '{key}'") from exc
    if not isinstance(value, str):
        raise ValueError(f"Parameter '{key}' must be provided as a string")
    return value


def _to_text(value: Any) -> str:
    """Convert any value to text representation."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _listify(items: Iterable[Any]) -> list[Any]:
    """Convert iterable to list, filtering out falsy values."""
    return [item for item in items if item]


def _linear_retry_config(attempts: int, initial: float, timeout: float) -> RetryConfig:
    """Create a linear retry configuration."""
    initial = max(initial, 0.0)
    timeout = max(timeout, 0.0)
    if initial == 0:
        return RetryConfig(
            attempts=attempts,
            backoff_strategy=BackoffStrategy.NONE,
            jitter=False,
            timeout=timeout,
        )
    return RetryConfig(
        attempts=attempts,
        backoff_strategy=BackoffStrategy.LINEAR,
        backoff_initial=initial,
        backoff_max=max(initial * attempts, initial),
        jitter=False,
        timeout=timeout,
    )


def _select_license(message: Mapping[str, Any]) -> str | None:
    """Extract the most specific license reference available from Crossref."""

    license_field = message.get("license")
    if isinstance(license_field, Sequence):
        for entry in license_field:
            if not isinstance(entry, Mapping):
                continue
            url = entry.get("URL") or entry.get("url")
            if isinstance(url, str) and url.strip():
                return url.strip()
            name = entry.get("content-version") or entry.get("start")
            if isinstance(name, str) and name.strip():
                return name.strip()
    if isinstance(license_field, Mapping):
        url = license_field.get("URL") or license_field.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    if isinstance(license_field, str) and license_field.strip():
        return license_field.strip()
    return None


def _extract_pdf_assets(
    message: Mapping[str, Any], landing_page: str | None
) -> list[Mapping[str, Any]]:
    """Normalise Crossref link metadata into manifest entries."""

    assets: list[dict[str, Any]] = []
    links = message.get("link")
    if not isinstance(links, Sequence):
        return assets
    license_hint = _select_license(message)
    for link in links:
        if not isinstance(link, Mapping):
            continue
        url = link.get("URL") or link.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        content_type = link.get("content-type") or link.get("contentType")
        if content_type and "pdf" not in content_type.lower():
            continue
        asset: dict[str, Any] = {
            "url": url,
            "landing_page_url": landing_page,
            "license": license_hint,
            "version": link.get("content-version") or link.get("version"),
            "source": link.get("intended-application"),
            "content_type": content_type,
        }
        checksum = link.get("checksum")
        if isinstance(checksum, str) and checksum.strip():
            asset["checksum_hint"] = checksum.strip()
        assets.append(asset)
    return assets


class ResilientHTTPAdapter(BaseAdapter):
    """Base adapter that wraps :class:`HttpClient` with sensible defaults."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        rate_limit_per_second: float,
        burst: int | None = None,
        retry: RetryConfig | None = None,
        client: HttpClient | None = None,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(name=name)
        self._owns_client = client is None
        self._default_headers = dict(default_headers or {})
        if client is None:
            self._client = HttpClient(
                base_url=base_url,
                retry=retry or RetryConfig(),
                rate_limit=RateLimitConfig(
                    rate_per_second=rate_limit_per_second,
                    burst=burst,
                ),
                circuit_breaker=CircuitBreakerConfig(),
            )
        else:
            self._client = client

    def _get_json(self, path: str, *, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Make a GET request and return JSON response."""
        response = self._client.request(
            "GET",
            path,
            params=params,
            headers=self._default_headers or None,
        )
        response.raise_for_status()
        return response.json()

    def _get_text(self, path: str, *, params: Mapping[str, Any] | None = None) -> str:
        """Make a GET request and return text response."""
        response = self._client.request(
            "GET",
            path,
            params=params,
            headers=self._default_headers or None,
        )
        response.raise_for_status()
        return response.text

    def write(
        self, documents: Sequence[Document], context: AdapterContext
    ) -> None:  # pragma: no cover - passthrough
        """Persistence is handled by downstream ingestion pipeline; adapters simply return documents."""
        return None

    def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._owns_client:
            self._client.close()


class CrossrefAdapter(PdfManifestMixin, ResilientHTTPAdapter):
    """Adapter for Crossref citation metadata."""

    def __init__(
        self,
        *,
        client: HttpClient | None = None,
        pdf_settings: ConnectorPdfSettings | None = None,
    ) -> None:
        settings = pdf_settings or get_settings().crossref.pdf
        self._pdf_settings = settings
        self._polite_headers = settings.polite_headers()
        super().__init__(
            name="crossref",
            base_url="https://api.crossref.org",
            rate_limit_per_second=settings.requests_per_second,
            burst=settings.burst,
            retry=_linear_retry_config(
                settings.retry_attempts,
                settings.retry_backoff_seconds,
                settings.timeout_seconds,
            ),
            client=client,
            default_headers=self._polite_headers,
        )

    def fetch(self, context: AdapterContext) -> Iterable[dict[str, Any]]:
        """Fetch bibliographic metadata for a DOI."""
        doi = validate_doi(_require_parameter(context, "doi"))
        payload = self._get_json(
            f"/works/{doi}",
            params={"mailto": self._pdf_settings.contact_email},
        )
        message = payload.get("message", {})
        return [message]

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        """Parse Crossref response into documents."""
        documents: list[Document] = []
        for message in payloads:
            doi = message.get("DOI")
            title_list = message.get("title", [])
            title = title_list[0] if title_list else None
            references = [ref.get("DOI") for ref in message.get("reference", []) if ref.get("DOI")]
            landing_page = message.get("URL") if isinstance(message.get("URL"), str) else None
            manifest_assets = _extract_pdf_assets(message, landing_page)

            metadata = {
                "doi": doi,
                "publisher": message.get("publisher"),
                "issued": message.get("issued"),
                "reference_count": message.get("reference-count"),
                "references": _listify(references),
                "is_referenced_by_count": message.get("is-referenced-by-count"),
                "source": "crossref",
            }
            if manifest_assets:
                metadata["pdf_assets"] = manifest_assets

            # Extract abstract if available
            abstract = message.get("abstract")
            abstract_text = _to_text(abstract) if abstract else "No abstract available"

            block = Block(
                id="crossref-block",
                type=BlockType.PARAGRAPH,
                text=abstract_text,
                spans=[],
            )
            section = Section(id="metadata", title="Crossref Metadata", blocks=[block])

            document = Document(
                id=build_document_id("crossref", doi or title or "unknown"),
                source="crossref",
                title=title,
                sections=[section],
                metadata=metadata,
            )
            if manifest_assets:
                manifest = self.build_pdf_manifest(
                    connector="crossref",
                    assets=manifest_assets,
                    polite_headers=self._polite_headers,
                )
                self.attach_manifest_to_documents([document], manifest)
            documents.append(document)
        return documents

    def polite_headers(self) -> Mapping[str, str]:
        """Expose polite pool headers for observability/tests."""

        return self._polite_headers
