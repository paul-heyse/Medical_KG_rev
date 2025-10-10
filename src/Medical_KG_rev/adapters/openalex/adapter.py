"""OpenAlex adapter backed by the official ``pyalex`` client.

This module provides an adapter for the OpenAlex scholarly works API,
which aggregates metadata about academic papers, authors, institutions,
and citations. The adapter uses the official pyalex client for optimal
performance and API compatibility.

Key Responsibilities:
    - Fetch scholarly works metadata from OpenAlex API
    - Extract citation information and PDF availability
    - Transform API responses into Document objects
    - Handle pagination and rate limiting via pyalex
    - Support DOI-based and keyword-based searches

Collaborators:
    - Upstream: OpenAlex API via pyalex client
    - Downstream: Document models, storage helpers

Side Effects:
    - Makes HTTP requests to OpenAlex API
    - Validates DOI identifiers
    - Creates Document objects with structured content

Thread Safety:
    - Thread-safe: Stateless adapter with no shared mutable state

Performance Characteristics:
    - Rate limiting: Handled by pyalex client
    - Pagination: Automatic via pyalex iterator
    - Response parsing: O(n) where n is response size

Example:
    >>> adapter = OpenAlexAdapter(max_results=10)
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
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, cast

import structlog
from Medical_KG_rev.adapters.base import AdapterContext, BaseAdapter
from Medical_KG_rev.adapters.mixins import PdfManifestMixin, StorageHelperMixin
from Medical_KG_rev.config.settings import ConnectorPdfSettings, get_settings
from Medical_KG_rev.models import Block, BlockType, Document, Section
from Medical_KG_rev.utils.identifiers import build_document_id
from Medical_KG_rev.utils.validation import validate_doi

try:  # pragma: no cover - import is validated in __init__
    from pyalex import Works
    from pyalex import config as pyalex_config
except Exception:  # pragma: no cover - handled lazily
    Works = None
    pyalex_config = None


# ==============================================================================
# GLOBAL STATE
# ==============================================================================

logger = structlog.get_logger(__name__)


# ==============================================================================
# ADAPTER IMPLEMENTATION
# ==============================================================================


class OpenAlexAdapter(BaseAdapter, PdfManifestMixin, StorageHelperMixin):
    """Adapter that retrieves scholarly works via the OpenAlex API.

    The adapter relies on the official :mod:`pyalex` client so that we benefit
    from built-in pagination, polite pool handling, and upstream API changes.
    PDF availability is surfaced through structured metadata so the downstream
    PDF ingestion pipeline can persist artefacts for MinerU processing.
    """

    DEFAULT_MAX_RESULTS = 5

    def __init__(
        self,
        *,
        client: Any | None = None,
        contact_email: str | None = None,
        user_agent: str | None = None,
        pdf_settings: ConnectorPdfSettings | None = None,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> None:
        super().__init__(name="openalex")
        settings = pdf_settings or get_settings().openalex.pdf
        overrides: dict[str, Any] = {}
        if contact_email:
            overrides["contact_email"] = contact_email
        if user_agent:
            overrides["user_agent"] = user_agent
        if overrides:
            settings = settings.model_copy(update=overrides)
        self._pdf_settings = settings
        self._polite_headers = settings.polite_headers()
        self._client = client or self._build_client()
        self._max_results = max(1, int(max_results))

    # ------------------------------------------------------------------
    # Adapter lifecycle
    # ------------------------------------------------------------------
    def fetch(self, context: AdapterContext) -> Iterable[dict[str, Any]]:
        parameters = context.parameters
        doi_param = parameters.get("doi")
        work_id_param = parameters.get("openalex_id")
        query_param = parameters.get("query")

        if isinstance(doi_param, str) and doi_param.strip():
            return self._fetch_by_doi(doi_param)
        if isinstance(work_id_param, str) and work_id_param.strip():
            return self._fetch_by_identifier(work_id_param)
        if isinstance(query_param, str) and query_param.strip():
            return self._search(query_param)

        raise ValueError("Either 'doi', 'openalex_id', or 'query' parameter must be provided")

    async def fetch_and_upload_pdf(
        self,
        context: AdapterContext,
        pdf_url: str,
        document_id: str,
    ) -> str | None:
        """Fetch PDF from URL and upload to storage if configured."""
        if not self._pdf_storage:
            return None

        try:
            # Fetch PDF data using HTTP client
            from Medical_KG_rev.utils.http_client import HttpClient

            client = HttpClient()
            response = client.request("GET", pdf_url)
            if response.status_code != 200:
                return None

            pdf_data = response.content

            # Upload to storage
            storage_uri = await self.upload_pdf_if_available(
                tenant_id=context.tenant_id,
                document_id=document_id,
                pdf_data=pdf_data,
            )

            return storage_uri
        except Exception as e:
            logger.warning(
                "openalex_adapter.pdf_fetch_failed",
                pdf_url=pdf_url,
                document_id=document_id,
                error=str(e),
            )
            return None

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        documents: list[Document] = []
        for payload in payloads:
            work = dict(payload)
            manifest_assets = work.get("pdf_assets")

            # Build document with manifest metadata included
            if isinstance(manifest_assets, Sequence) and manifest_assets:
                manifest = self.build_pdf_manifest(
                    connector="openalex",
                    assets=cast(Sequence[Mapping[str, Any]], manifest_assets),
                    polite_headers=self._polite_headers,
                )
                # Include manifest in the work data before building document
                work["pdf_manifest"] = manifest.as_metadata()
                urls = list(manifest.pdf_urls())
                if urls:
                    work.setdefault("pdf_urls", urls)
                    work["document_type"] = "pdf"

            document = self._build_document(work)
            documents.append(document)
        return documents

    def polite_headers(self) -> Mapping[str, str]:
        return self._polite_headers

    def write(
        self, documents: Sequence[Document], context: AdapterContext
    ) -> None:  # pragma: no cover - passthrough
        return None

    # ------------------------------------------------------------------
    # Client helpers
    # ------------------------------------------------------------------
    def _build_client(self) -> Any:
        if Works is None or pyalex_config is None:  # pragma: no cover - validated in tests
            raise RuntimeError("pyalex>=0.18 is required for OpenAlexAdapter")

        email = self._pdf_settings.contact_email or os.getenv("OPENALEX_CONTACT_EMAIL")
        agent = self._pdf_settings.user_agent or os.getenv("OPENALEX_USER_AGENT")

        if email:
            pyalex_config["email"] = email
        if agent:
            pyalex_config["user_agent"] = agent

        return Works()

    def _fetch_by_doi(self, raw_doi: str) -> list[dict[str, Any]]:
        doi = validate_doi(raw_doi)
        identifier = f"https://doi.org/{doi}"
        work = self._client[identifier]
        return [self._normalise_work_payload(work)]

    def _fetch_by_identifier(self, raw_identifier: str) -> list[dict[str, Any]]:
        identifier = _normalise_openalex_id(raw_identifier)
        work = self._client[identifier]
        return [self._normalise_work_payload(work)]

    def _search(self, query: str) -> list[dict[str, Any]]:
        records: list[Mapping[str, Any]] = []
        paginator = self._client.search(query).paginate(per_page=self._max_results)
        for page in paginator:
            for work in page:
                records.append(self._normalise_work_payload(work))
                if len(records) >= self._max_results:
                    return records
        return records

    # ------------------------------------------------------------------
    # Document construction helpers
    # ------------------------------------------------------------------
    def _normalise_work_payload(self, work: Mapping[str, Any]) -> dict[str, Any]:
        data = dict(work)
        pdf_assets = _extract_pdf_assets(data)
        if pdf_assets:
            data["pdf_assets"] = pdf_assets
            data["pdf_urls"] = [asset["url"] for asset in pdf_assets]
            data["document_type"] = "pdf"
        else:
            data.setdefault("pdf_assets", [])
            data.setdefault("pdf_urls", [])
        data.setdefault("source", "openalex")
        return data

    def _build_document(self, payload: Mapping[str, Any]) -> Document:
        work_id = str(payload.get("id") or payload.get("openalex_id"))
        if not work_id:
            raise ValueError("OpenAlex payload missing identifier")
        document_token = work_id.rsplit("/", 1)[-1]
        document_id = build_document_id("openalex", document_token)

        abstract_text = _extract_abstract(payload)
        section = Section(
            id="abstract",
            title="Abstract",
            blocks=[
                Block(
                    id="abstract-block",
                    type=BlockType.PARAGRAPH,
                    text=abstract_text,
                    spans=[],
                )
            ],
        )

        metadata = _build_metadata(payload)
        pdf_urls = metadata.get("pdf_urls")
        if pdf_urls:
            metadata.setdefault("document_type", "pdf")

        title = payload.get("display_name") or payload.get("title")

        return Document(
            id=document_id,
            source="openalex",
            title=title if isinstance(title, str) else None,
            sections=[section],
            metadata=metadata,
        )


# ----------------------------------------------------------------------
# Metadata helpers
# ----------------------------------------------------------------------
def _build_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    work_id = str(payload.get("id") or payload.get("openalex_id"))
    doi = _normalise_doi(payload.get("doi") or _safe_get(payload.get("ids"), "doi"))
    authors = _collect_authors(payload.get("authorships"))
    concepts = _collect_concepts(payload.get("concepts"))
    topics = _collect_topics(payload)
    open_access = payload.get("open_access") if isinstance(payload.get("open_access"), Mapping) else {}
    pdf_assets = payload.get("pdf_assets") if isinstance(payload.get("pdf_assets"), Sequence) else []
    pdf_urls = payload.get("pdf_urls") if isinstance(payload.get("pdf_urls"), Sequence) else []

    metadata: dict[str, Any] = {
        "openalex_id": work_id,
        "doi": doi,
        "title": payload.get("display_name") or payload.get("title"),
        "publication_year": payload.get("publication_year"),
        "publication_date": payload.get("publication_date"),
        "cited_by_count": payload.get("cited_by_count"),
        "authorships": authors,
        "concepts": concepts,
        "topics": topics,
        "is_open_access": bool(open_access.get("is_oa")),
        "oa_status": open_access.get("oa_status"),
        "pdf_assets": list(pdf_assets),
    }

    # Include PDF manifest if present
    pdf_manifest = payload.get("pdf_manifest")
    if pdf_manifest:
        metadata["pdf_manifest"] = pdf_manifest

    if pdf_urls:
        metadata["pdf_urls"] = list(pdf_urls)

    identifiers = payload.get("ids")
    if isinstance(identifiers, Mapping):
        metadata["identifiers"] = dict(identifiers)

    primary_location = payload.get("primary_location")
    if isinstance(primary_location, Mapping):
        metadata["primary_location"] = {
            key: primary_location.get(key)
            for key in ("landing_page_url", "pdf_url", "source", "version", "license")
            if key in primary_location
        }

    return metadata


def _extract_abstract(payload: Mapping[str, Any]) -> str:
    inverted_index = payload.get("abstract_inverted_index")
    if isinstance(inverted_index, Mapping):
        return _flatten_abstract(cast(Mapping[str, Sequence[int]], inverted_index))
    abstract = payload.get("abstract")
    if isinstance(abstract, str):
        return abstract
    return ""


def _collect_authors(authorships: Any) -> list[str]:
    if not isinstance(authorships, Sequence):
        return []
    names: list[str] = []
    for entry in authorships:
        if not isinstance(entry, Mapping):
            continue
        author = entry.get("author")
        if isinstance(author, Mapping):
            display_name = author.get("display_name")
            if isinstance(display_name, str) and display_name:
                names.append(display_name)
    return names


def _collect_concepts(raw_concepts: Any) -> list[str]:
    if not isinstance(raw_concepts, Sequence):
        return []
    concepts: list[str] = []
    for concept in raw_concepts:
        if isinstance(concept, Mapping):
            display_name = concept.get("display_name")
            if isinstance(display_name, str):
                concepts.append(display_name)
    return concepts


def _collect_topics(payload: Mapping[str, Any]) -> list[str]:
    topics: list[str] = []
    primary_topic = payload.get("primary_topic")
    if isinstance(primary_topic, Mapping):
        name = primary_topic.get("display_name")
        if isinstance(name, str) and name:
            topics.append(name)
    raw_topics = payload.get("topics")
    if isinstance(raw_topics, Sequence):
        for entry in raw_topics:
            if isinstance(entry, Mapping):
                name = entry.get("display_name")
                if isinstance(name, str) and name:
                    topics.append(name)
    # Remove duplicates preserving order
    deduped: list[str] = []
    seen: set[str] = set()
    for value in topics:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _extract_pdf_assets(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for location_type, location in _iter_pdf_locations(payload):
        pdf_url = _extract_pdf_url(location)
        if not pdf_url:
            continue
        if pdf_url in seen_urls:
            continue
        seen_urls.add(pdf_url)

        source_info = location.get("source") if isinstance(location.get("source"), Mapping) else {}
        asset = {
            "url": pdf_url,
            "landing_page_url": location.get("landing_page_url"),
            "license": location.get("license") or location.get("license_id"),
            "version": location.get("version"),
            "location_type": location_type,
            "is_open_access": bool(location.get("is_oa")),
            "source": source_info.get("display_name") if isinstance(source_info, Mapping) else None,
            "source_id": source_info.get("id") if isinstance(source_info, Mapping) else None,
        }
        assets.append(asset)
    return assets


def _iter_pdf_locations(payload: Mapping[str, Any]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    primary = payload.get("primary_location")
    if isinstance(primary, Mapping):
        yield "primary", primary
    best = payload.get("best_oa_location")
    if isinstance(best, Mapping):
        yield "best_oa", best
    locations = payload.get("locations")
    if isinstance(locations, Sequence):
        for index, location in enumerate(locations):
            if isinstance(location, Mapping):
                yield f"location[{index}]", location


def _extract_pdf_url(location: Mapping[str, Any]) -> str | None:
    for key in ("pdf_url", "url_for_pdf"):
        value = location.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _flatten_abstract(inverted_index: Mapping[str, Sequence[int]]) -> str:
    pairs: list[tuple[int, str]] = []
    for term, positions in inverted_index.items():
        if not isinstance(positions, Sequence):
            continue
        for position in positions:
            if isinstance(position, int):
                pairs.append((position, term))
    pairs.sort(key=lambda entry: entry[0])
    terms = [term for _, term in pairs]
    return " ".join(terms)


def _normalise_openalex_id(raw_identifier: str) -> str:
    identifier = raw_identifier.strip()
    if identifier.startswith("https://openalex.org/"):
        return identifier
    if identifier.startswith("openalex.org/"):
        return f"https://{identifier}"
    if identifier.startswith("W"):
        return f"https://openalex.org/{identifier}"
    raise ValueError(f"Invalid OpenAlex identifier: {raw_identifier!r}")


def _normalise_doi(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    if candidate.lower().startswith("https://doi.org/"):
        candidate = candidate.split("/", 3)[-1]
    try:
        return validate_doi(candidate)
    except ValueError:
        return candidate


def _safe_get(mapping: Any, key: str) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    return None


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "OpenAlexAdapter",
]

