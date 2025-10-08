"""Semantic Scholar adapter for citation enrichment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

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


class SemanticScholarAdapter(ResilientHTTPAdapter):
    """Adapter for Semantic Scholar citation enrichment."""

    def __init__(self, client: HttpClient | None = None) -> None:
        super().__init__(
            name="semantic-scholar",
            base_url="https://api.semanticscholar.org/graph/v1",
            rate_limit_per_second=4,
            retry=_linear_retry_config(4, 0.5),
            client=client,
        )

    def fetch(self, context: AdapterContext) -> Iterable[dict[str, Any]]:
        """Fetch Semantic Scholar data by DOI or paper ID."""
        doi = context.parameters.get("doi")
        if doi:
            identifier = f"DOI:{validate_doi(str(doi))}"
        else:
            identifier = _require_parameter(context, "paper_id")
        payload = self._get_json(
            f"/paper/{identifier}",
            params={"fields": "title,externalIds,citationCount,referenceCount,references"},
        )
        return [payload]

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        """Parse Semantic Scholar response into documents."""
        documents: list[Document] = []
        for payload in payloads:
            paper_id = payload.get("paperId") or payload.get("paper_id")
            doi = (
                payload.get("externalIds", {}).get("DOI")
                if isinstance(payload.get("externalIds"), Mapping)
                else None
            )
            references = [
                ref.get("title") or ref.get("paperId") for ref in payload.get("references", [])
            ]

            metadata = {
                "paper_id": paper_id,
                "doi": doi,
                "citation_count": payload.get("citationCount"),
                "reference_count": payload.get("referenceCount"),
                "references": _listify(references),
                "source": "semantic-scholar",
            }

            block = Block(
                id="semantic-scholar",
                type=BlockType.PARAGRAPH,
                text=f"Citations: {payload.get('citationCount', 0)}",
                spans=[],
            )
            section = Section(id="semantic-scholar", title="Semantic Scholar", blocks=[block])

            documents.append(
                Document(
                    id=paper_id or build_document_id("semantic-scholar", doi or "paper"),
                    source="semantic-scholar",
                    title=payload.get("title"),
                    sections=[section],
                    metadata=metadata,
                )
            )
        return documents
