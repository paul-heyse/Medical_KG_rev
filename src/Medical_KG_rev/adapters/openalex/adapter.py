"""OpenAlex adapter backed by the official ``pyalex`` client."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Iterator, cast

from Medical_KG_rev.adapters.base import AdapterContext, BaseAdapter
from Medical_KG_rev.models import Block, BlockType, Document, Section
from Medical_KG_rev.utils.identifiers import build_document_id
from Medical_KG_rev.utils.validation import validate_doi

try:  # pragma: no cover - import is validated in __init__
    from pyalex import Works, config as pyalex_config
except Exception:  # pragma: no cover - handled lazily
    Works = None  # type: ignore[assignment]
    pyalex_config = None  # type: ignore[assignment]


class OpenAlexAdapter(BaseAdapter):
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
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> None:
        super().__init__(name="openalex")
        self._client = client or self._build_client(contact_email, user_agent)
        self._max_results = max(1, int(max_results))

    # ------------------------------------------------------------------
    # Adapter lifecycle
    # ------------------------------------------------------------------
    def fetch(self, context: AdapterContext) -> Iterable[Mapping[str, Any]]:
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

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        documents: list[Document] = []
        for payload in payloads:
            work = dict(payload)
            document = self._build_document(work)
            documents.append(document)
        return documents

    def write(
        self, documents: Sequence[Document], context: AdapterContext
    ) -> None:  # pragma: no cover - passthrough
        return None

    # ------------------------------------------------------------------
    # Client helpers
    # ------------------------------------------------------------------
    def _build_client(self, contact_email: str | None, user_agent: str | None) -> Any:
        if Works is None or pyalex_config is None:  # pragma: no cover - validated in tests
            raise RuntimeError("pyalex>=0.18 is required for OpenAlexAdapter")

        email = contact_email or os.getenv("OPENALEX_CONTACT_EMAIL")
        agent = user_agent or os.getenv("OPENALEX_USER_AGENT")

        if email:
            pyalex_config["email"] = email  # type: ignore[index]
        if agent:
            pyalex_config["user_agent"] = agent  # type: ignore[index]

        return Works()

    def _fetch_by_doi(self, raw_doi: str) -> list[Mapping[str, Any]]:
        doi = validate_doi(raw_doi)
        identifier = f"https://doi.org/{doi}"
        work = self._client[identifier]
        return [self._normalise_work_payload(work)]

    def _fetch_by_identifier(self, raw_identifier: str) -> list[Mapping[str, Any]]:
        identifier = _normalise_openalex_id(raw_identifier)
        work = self._client[identifier]
        return [self._normalise_work_payload(work)]

    def _search(self, query: str) -> list[Mapping[str, Any]]:
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
    def _normalise_work_payload(self, work: Mapping[str, Any]) -> Mapping[str, Any]:
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
    open_access = (
        payload.get("open_access") if isinstance(payload.get("open_access"), Mapping) else {}
    )
    pdf_assets = (
        payload.get("pdf_assets") if isinstance(payload.get("pdf_assets"), Sequence) else []
    )
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
    return [
        concept.get("display_name")
        for concept in raw_concepts
        if isinstance(concept, Mapping) and concept.get("display_name")
    ]


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
            "source": source_info.get("display_name"),
            "source_id": source_info.get("id"),
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
    terms = [term for _, term in sorted(pairs, key=lambda entry: entry[0])]
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
