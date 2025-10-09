"""PubMed Central adapter for full-text XML retrieval.

This module provides an adapter for the PubMed Central (PMC) API,
which provides access to full-text XML documents from biomedical
and life science journals. The adapter fetches PMC articles and
extracts structured content from XML.

Key Responsibilities:
    - Fetch PMC articles by PMCID identifier
    - Parse XML content into structured sections
    - Extract metadata and full-text content
    - Transform XML into Document objects
    - Handle XML parsing and validation

Collaborators:
    - Upstream: PMC API (external service)
    - Downstream: Document models, XML parser

Side Effects:
    - Makes HTTP requests to PMC API
    - Parses XML content
    - Validates PMCID identifiers
    - Creates Document objects with structured content

Thread Safety:
    - Thread-safe: Stateless adapter with no shared mutable state

Performance Characteristics:
    - Rate limiting: 3 requests per second
    - XML parsing: O(n) where n is XML size
    - Content extraction: O(m) where m is number of sections

Example:
    >>> adapter = PMCAdapter()
    >>> context = AdapterContext(
    ...     tenant_id="tenant1",
    ...     domain="medical",
    ...     correlation_id="corr1",
    ...     parameters={"pmcid": "PMC123456"}
    ... )
    >>> documents = adapter.fetch_and_parse(context)

"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

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
from Medical_KG_rev.utils.validation import validate_pmcid

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


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


def _collect_text(element: Element | None) -> str:
    """Extract text content from XML element."""
    if element is None:
        return ""
    parts = [segment.strip() for segment in element.itertext() if segment and segment.strip()]
    return " ".join(parts)


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


XLINK_NS = "{http://www.w3.org/1999/xlink}"


def _extract_license_from_xml(root: Element) -> str | None:
    """Extract license reference from PMC XML if present."""

    for node in root.findall(".//license"):
        href = node.get(f"{XLINK_NS}href")
        if href and href.strip():
            return href.strip()
        text = "".join(node.itertext()).strip()
        if text:
            return text
    return None


def _collect_pdf_assets(root: Element, pmcid: str | None) -> list[Mapping[str, Any]]:
    """Collect candidate PDF assets from PMC XML."""

    license_hint = _extract_license_from_xml(root)
    landing_page = f"https://europepmc.org/article/pmcid/{pmcid}" if pmcid else None
    assets: list[dict[str, Any]] = []

    def _normalise_href(node: Element) -> str | None:
        href = node.get(f"{XLINK_NS}href") or node.get("href")
        if href and href.strip():
            return href.strip()
        text = (node.text or "").strip()
        if text.lower().startswith("http") and ".pdf" in text.lower():
            return text
        return None

    def _is_pdf(node: Element, href: str | None) -> bool:
        if href and href.lower().endswith(".pdf"):
            return True
        content_type = node.get("content-type") or node.get("type")
        if content_type and "pdf" in content_type.lower():
            return True
        link_type = node.get("ext-link-type") or node.get("link-type")
        if link_type and "pdf" in link_type.lower():
            return True
        return False

    for selector in (".//self-uri", ".//ext-link"):
        for node in root.findall(selector):
            href = _normalise_href(node)
            if not href or not _is_pdf(node, href):
                continue
            asset: dict[str, Any] = {
                "url": href,
                "landing_page_url": landing_page,
                "license": license_hint,
                "source": "pmc",
                "content_type": node.get("content-type") or node.get("type"),
                "version": node.get("version"),
                "is_open_access": True,
            }
            assets.append(asset)

    return assets


# ==============================================================================
# ADAPTER IMPLEMENTATION
# ==============================================================================

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


class PMCAdapter(PdfManifestMixin, ResilientHTTPAdapter):
    """Adapter for Europe PMC full-text XML retrieval."""

    def __init__(
        self,
        *,
        client: HttpClient | None = None,
        pdf_settings: ConnectorPdfSettings | None = None,
    ) -> None:
        settings = pdf_settings or get_settings().pmc.pdf
        self._pdf_settings = settings
        self._polite_headers = settings.polite_headers()
        super().__init__(
            name="pmc",
            base_url="https://www.ebi.ac.uk/europepmc",
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
        """Fetch full-text XML for a PMCID."""
        pmcid = validate_pmcid(_require_parameter(context, "pmcid"))
        xml_text = self._get_text(f"/webservices/rest/{pmcid}/fullTextXML")
        return [{"xml_content": xml_text}]

    def parse(self, payloads: Iterable[dict[str, Any]], context: AdapterContext) -> Sequence[Document]:
        """Parse PMC XML into documents."""
        documents: list[Document] = []
        for payload in payloads:
            xml_text = payload.get("xml_content", "")
            root = ElementTree.fromstring(xml_text)
            pmcid = root.findtext(".//article-id[@pub-id-type='pmcid']")
            title = _collect_text(root.find(".//article-title"))
            abstract_text = _collect_text(root.find(".//abstract"))
            body_paragraphs = [_collect_text(elem) for elem in root.findall(".//body//p")]
            manifest_assets = _collect_pdf_assets(root, pmcid)

            blocks: list[Block] = []
            if abstract_text:
                blocks.append(Block(
                    id="pmc-abstract",
                    type=BlockType.PARAGRAPH,
                    text=abstract_text,
                    spans=[]
                ))
            for idx, paragraph in enumerate(body_paragraphs[:5]):
                blocks.append(Block(
                    id=f"pmc-body-{idx}",
                    type=BlockType.PARAGRAPH,
                    text=_to_text(paragraph),
                    spans=[]
                ))

            section = Section(id="pmc", title="Europe PMC", blocks=blocks)

            metadata = {
                "pmcid": pmcid,
                "source": "pmc",
            }
            if manifest_assets:
                metadata["pdf_assets"] = manifest_assets
            else:
                metadata["document_type"] = "xml"

            document = Document(
                id=pmcid or build_document_id("pmc", title or "article"),
                source="pmc",
                title=title,
                sections=[section],
                metadata=metadata,
                )
            if manifest_assets:
                manifest = self.build_pdf_manifest(
                    connector="pmc",
                    assets=manifest_assets,
                    polite_headers=self._polite_headers,
                )
                self.attach_manifest_to_documents([document], manifest)
            documents.append(document)
        return documents

    def polite_headers(self) -> Mapping[str, str]:
        """Expose polite pool headers for tests and diagnostics."""

        return self._polite_headers


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "PMCAdapter",
    "ResilientHTTPAdapter",
]
