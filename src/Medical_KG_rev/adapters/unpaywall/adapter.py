"""Unpaywall adapter for open access status.

This module provides an adapter for the Unpaywall API, which provides
information about the open access status of academic papers. The adapter
checks whether papers are freely available and provides access URLs
when available.

Key Responsibilities:
    - Check open access status for academic papers
    - Fetch access URLs for freely available papers
    - Validate DOI identifiers
    - Transform API responses into Document objects
    - Handle rate limiting and API key authentication

Collaborators:
    - Upstream: Unpaywall API (external service)
    - Downstream: Document models, HTTP client

Side Effects:
    - Makes HTTP requests to Unpaywall API
    - Validates DOI identifiers
    - Creates Document objects with access information

Thread Safety:
    - Thread-safe: Stateless adapter with no shared mutable state

Performance Characteristics:
    - Rate limiting: 10 requests per second (with API key)
    - Retry strategy: Linear backoff with 3 attempts
    - Response parsing: O(n) where n is response size

Example:
    >>> adapter = UnpaywallAdapter()
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

import os
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


class UnpaywallAdapter(ResilientHTTPAdapter):
    """Adapter for Unpaywall open access status."""

    def __init__(
        self, email: str | None = None, client: HttpClient | None = None
    ) -> None:
        super().__init__(
            name="unpaywall",
            base_url="https://api.unpaywall.org/v2",
            rate_limit_per_second=5,
            retry=_linear_retry_config(3, 0.5),
            client=client,
        )
        self._email = email or os.getenv("UNPAYWALL_EMAIL", "oss@medical-kg.local")

    def fetch(self, context: AdapterContext) -> Iterable[dict[str, Any]]:
        """Fetch open access status for a DOI."""
        doi = validate_doi(_require_parameter(context, "doi"))
        payload = self._get_json(f"/{doi}", params={"email": self._email})
        return [payload]

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        """Parse Unpaywall response into documents."""
        documents: list[Document] = []
        for payload in payloads:
            doi = payload.get("doi")
            metadata = {
                "doi": doi,
                "is_open_access": payload.get("is_oa"),
                "oa_status": payload.get("oa_status"),
                "journal": payload.get("journal_name"),
                "best_oa_location": payload.get("best_oa_location"),
                "source": "unpaywall",
            }

            # Extract PDF URL if available
            location = payload.get("best_oa_location", {})
            pdf_url = location.get("url")
            if pdf_url:
                metadata["pdf_urls"] = [pdf_url]
                metadata["document_type"] = "pdf"

            text = pdf_url or "No open access location available"
            section = Section(
                id="unpaywall",
                title="Open Access",
                blocks=[
                    Block(
                        id="oa-block",
                        type=BlockType.PARAGRAPH,
                        text=_to_text(text),
                        spans=[]
                    )
                ],
            )

            documents.append(
                Document(
                    id=build_document_id("unpaywall", doi or "unknown"),
                    source="unpaywall",
                    title=payload.get("title"),
                    sections=[section],
                    metadata=metadata,
                )
            )
        return documents


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "ResilientHTTPAdapter",
    "UnpaywallAdapter",
]
