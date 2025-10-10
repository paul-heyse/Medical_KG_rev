"""CORE adapter for open research papers.

This module provides an adapter for the CORE (COnnecting REpositories) API,
which aggregates open access research papers from repositories worldwide.
The adapter fetches paper metadata and full-text content when available.

Key Responsibilities:
    - Fetch research paper metadata from CORE API
    - Extract full-text content when available
    - Validate DOI identifiers
    - Transform API responses into Document objects
    - Handle rate limiting and retry logic

Collaborators:
    - Upstream: CORE API (external service)
    - Downstream: Document models, HTTP client

Side Effects:
    - Makes HTTP requests to CORE API
    - Validates DOI identifiers
    - Creates Document objects with structured content

Thread Safety:
    - Thread-safe: Stateless adapter with no shared mutable state

Performance Characteristics:
    - Rate limiting: 3 requests per second
    - Retry strategy: Linear backoff with 3 attempts
    - Response parsing: O(n) where n is response size

Example:
    >>> adapter = CoreAdapter()
    >>> context = AdapterContext(
    ...     tenant_id="tenant1",
    ...     domain="research",
    ...     correlation_id="corr1",
    ...     parameters={"doi": "10.1371/journal.pone.0123456"}
    ... )
    >>> documents = adapter.fetch_and_parse(context)

"""

# ==============================================================================
# IMPORTS
# ==============================================================================

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
from Medical_KG_rev.utils.validation import validate_doi

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


class COREAdapter(ResilientHTTPAdapter):
    """Adapter for CORE PDF access."""

    def __init__(self, client: HttpClient | None = None) -> None:
        super().__init__(
            name="core",
            base_url="https://core.ac.uk/api-v3",
            rate_limit_per_second=2,
            retry=_linear_retry_config(3, 0.5),
            client=client,
        )

    def fetch(self, context: AdapterContext) -> Iterable[dict[str, Any]]:
        """Fetch CORE paper data by core_id or DOI."""
        core_id = context.parameters.get("core_id")
        doi = context.parameters.get("doi")
        if core_id:
            payload = self._get_json(f"/works/{core_id}")
        elif doi:
            identifier = validate_doi(str(doi))
            payload = self._get_json("/works/search", params={"doi": identifier})
        else:
            raise ValueError("Either 'core_id' or 'doi' parameter must be provided")
        return [payload]

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        """Parse CORE response into documents."""
        documents: list[Document] = []
        for payload in payloads:
            raw_entries = payload.get("data")
            if isinstance(raw_entries, list):
                entries = raw_entries
            elif raw_entries:
                entries = [raw_entries]
            else:
                entries = [payload]
            for data in entries:
                work_id = data.get("id") or data.get("coreId")
                if not work_id:
                    continue
                full_text = data.get("fullText") or data.get("fullTextLink")

                metadata = {
                    "download_url": data.get("downloadUrl"),
                    "doi": data.get("doi"),
                    "source": "core",
                }

                # Extract PDF URL if available
                download_url = data.get("downloadUrl")
                if download_url and download_url.endswith(".pdf"):
                    metadata["pdf_urls"] = [download_url]
                    metadata["document_type"] = "pdf"

                block = Block(
                    id="core-text", type=BlockType.PARAGRAPH, text=_to_text(full_text), spans=[]
                )
                section = Section(id="core", title="CORE Full Text", blocks=[block])

                documents.append(
                    Document(
                        id=str(work_id),
                        source="core",
                        title=data.get("title"),
                        sections=[section],
                        metadata=metadata,
                    )
                )
        return documents


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "COREAdapter",
    "ResilientHTTPAdapter",
]
