"""PubMed Central adapter for full-text XML retrieval."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

from Medical_KG_rev.adapters.base import AdapterContext, BaseAdapter
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


def _linear_retry_config(attempts: int, initial: float) -> RetryConfig:
    """Create a linear retry configuration."""
    initial = max(initial, 0.0)
    if initial == 0:
        return RetryConfig(attempts=attempts, backoff_strategy=BackoffStrategy.NONE, jitter=False)
    return RetryConfig(
        attempts=attempts,
        backoff_strategy=BackoffStrategy.LINEAR,
        backoff_initial=initial,
        backoff_max=max(initial * attempts, initial),
        jitter=False,
    )


class ResilientHTTPAdapter(BaseAdapter):
    """Base adapter that wraps :class:`HttpClient` with sensible defaults."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        rate_limit_per_second: float,
        retry: RetryConfig | None = None,
        client: HttpClient | None = None,
    ) -> None:
        super().__init__(name=name)
        self._owns_client = client is None
        if client is None:
            self._client = HttpClient(
                base_url=base_url,
                retry=retry or RetryConfig(),
                rate_limit=RateLimitConfig(rate_per_second=rate_limit_per_second),
                circuit_breaker=CircuitBreakerConfig(),
            )
        else:
            self._client = client

    def _get_json(self, path: str, *, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Make a GET request and return JSON response."""
        response = self._client.request("GET", path, params=params)
        response.raise_for_status()
        return response.json()

    def _get_text(self, path: str, *, params: Mapping[str, Any] | None = None) -> str:
        """Make a GET request and return text response."""
        response = self._client.request("GET", path, params=params)
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


class PMCAdapter(ResilientHTTPAdapter):
    """Adapter for Europe PMC full-text XML retrieval."""

    def __init__(self, client: HttpClient | None = None) -> None:
        super().__init__(
            name="pmc",
            base_url="https://www.ebi.ac.uk/europepmc",
            rate_limit_per_second=3,
            retry=_linear_retry_config(4, 1.0),
            client=client,
        )

    def fetch(self, context: AdapterContext) -> Iterable[dict[str, Any]]:
        """Fetch full-text XML for a PMCID."""
        pmcid = validate_pmcid(_require_parameter(context, "pmcid"))
        xml_text = self._get_text(f"/webservices/rest/{pmcid}/fullTextXML")
        return [{"xml_content": xml_text}]

    def parse(
        self, payloads: Iterable[dict[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        """Parse PMC XML into documents."""
        documents: list[Document] = []
        for payload in payloads:
            xml_text = payload.get("xml_content", "")
            root = ElementTree.fromstring(xml_text)
            pmcid = root.findtext(".//article-id[@pub-id-type='pmcid']")
            title = _collect_text(root.find(".//article-title"))
            abstract_text = _collect_text(root.find(".//abstract"))
            body_paragraphs = [_collect_text(elem) for elem in root.findall(".//body//p")]

            blocks: list[Block] = []
            if abstract_text:
                blocks.append(
                    Block(id="pmc-abstract", type=BlockType.PARAGRAPH, text=abstract_text, spans=[])
                )
            for idx, paragraph in enumerate(body_paragraphs[:5]):
                blocks.append(
                    Block(
                        id=f"pmc-body-{idx}",
                        type=BlockType.PARAGRAPH,
                        text=_to_text(paragraph),
                        spans=[],
                    )
                )

            section = Section(id="pmc", title="Europe PMC", blocks=blocks)

            metadata = {
                "pmcid": pmcid,
                "source": "pmc",
                "document_type": "xml",  # PMC provides structured XML content
            }

            documents.append(
                Document(
                    id=pmcid or build_document_id("pmc", title or "article"),
                    source="pmc",
                    title=title,
                    sections=[section],
                    metadata=metadata,
                )
            )
        return documents
