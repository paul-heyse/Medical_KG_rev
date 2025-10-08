"""Biomedical adapter implementations for external data sources."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

from Medical_KG_rev.models import Block, BlockType, Document, Section
from Medical_KG_rev.utils.http_client import (
    BackoffStrategy,
    CircuitBreakerConfig,
    HttpClient,
    RateLimitConfig,
    RetryConfig,
)
from Medical_KG_rev.utils.identifiers import build_document_id, normalize_identifier
from Medical_KG_rev.utils.validation import (
    validate_chembl_id,
    validate_doi,
    validate_icd11,
    validate_mesh_id,
    validate_nct_id,
    validate_ndc,
    validate_pmcid,
    validate_rxcui,
    validate_set_id,
)

from .base import AdapterContext, BaseAdapter


logger = structlog.get_logger(__name__)


def _require_parameter(context: AdapterContext, key: str) -> str:
    try:
        value = context.parameters[key]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Missing required parameter '{key}'") from exc
    if not isinstance(value, str):
        raise ValueError(f"Parameter '{key}' must be provided as a string")
    return value


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _flatten_abstract(inverted_index: Mapping[str, Sequence[int]]) -> str:
    terms = sorted(((pos, term) for term, positions in inverted_index.items() for pos in positions))
    ordered = [term for _, term in terms]
    return " ".join(ordered)


def _listify(items: Iterable[Any]) -> list[Any]:
    return [item for item in items if item]


def _collect_text(element: Element | None) -> str:
    if element is None:
        return ""
    parts = [segment.strip() for segment in element.itertext() if segment and segment.strip()]
    return " ".join(parts)


def _linear_retry_config(attempts: int, initial: float) -> RetryConfig:
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

    def _get_json(self, path: str, *, params: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        response = self._client.request("GET", path, params=params)
        response.raise_for_status()
        return response.json()

    def _get_text(self, path: str, *, params: Mapping[str, Any] | None = None) -> str:
        response = self._client.request("GET", path, params=params)
        response.raise_for_status()
        return response.text

    def write(
        self, documents: Sequence[Document], context: AdapterContext
    ) -> None:  # pragma: no cover - passthrough
        # Persistence is handled by downstream ingestion pipeline; adapters simply return documents.
        return None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


class ClinicalTrialsAdapter(ResilientHTTPAdapter):
    """Adapter for ClinicalTrials.gov API v2."""

    def __init__(self, client: HttpClient | None = None) -> None:
        super().__init__(
            name="clinicaltrials",
            base_url="https://clinicaltrials.gov/api/v2",
            rate_limit_per_second=3,
            retry=_linear_retry_config(4, 1.0),
            client=client,
        )

    def fetch(self, context: AdapterContext) -> Iterable[Mapping[str, Any]]:
        nct_id = validate_nct_id(_require_parameter(context, "nct_id"))
        payload = self._get_json(f"/studies/{nct_id}", params={"format": "json"})
        study = payload.get("study") or payload
        return [study]

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        documents: list[Document] = []
        for study in payloads:
            protocol = study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            nct_id = identification.get("nctId")
            if not nct_id:
                raise ValueError("ClinicalTrials payload missing nctId")
            status_module = protocol.get("statusModule", {})
            design_module = protocol.get("designModule", {})
            interventions_module = protocol.get("armsInterventionsModule", {})
            outcomes_module = protocol.get("outcomesModule", {})
            eligibility_module = protocol.get("eligibilityModule", {})
            description_module = protocol.get("descriptionModule", {})

            interventions = [
                f"{item.get('type')}: {item.get('name')}".strip()
                for item in interventions_module.get("interventions", [])
                if item.get("name")
            ]
            outcomes = [item.get("measure") for item in outcomes_module.get("primaryOutcomes", [])]
            metadata: dict[str, Any] = {
                "nct_id": nct_id,
                "brief_title": identification.get("briefTitle"),
                "official_title": identification.get("officialTitle"),
                "overall_status": status_module.get("overallStatus"),
                "study_type": design_module.get("studyType"),
                "phase": design_module.get("phases") or design_module.get("phase"),
                "start_date": status_module.get("startDateStruct", {}).get("date"),
                "completion_date": status_module.get("completionDateStruct", {}).get("date"),
                "interventions": _listify(interventions),
                "outcomes": _listify(outcomes),
                "eligibility": {
                    "criteria": eligibility_module.get("eligibilityCriteria"),
                    "sex": eligibility_module.get("sex"),
                    "minimum_age": eligibility_module.get("minimumAge"),
                    "maximum_age": eligibility_module.get("maximumAge"),
                },
            }

            sections: list[Section] = []
            summary_text = description_module.get("briefSummary")
            if summary_text:
                sections.append(
                    Section(
                        id="summary",
                        title="Brief Summary",
                        blocks=[Block(id="summary-block", text=_to_text(summary_text), spans=[])],
                    )
                )
            detailed_text = description_module.get("detailedDescription")
            if detailed_text:
                sections.append(
                    Section(
                        id="description",
                        title="Detailed Description",
                        blocks=[
                            Block(id="description-block", text=_to_text(detailed_text), spans=[])
                        ],
                    )
                )

            document = Document(
                id=nct_id,
                source="clinicaltrials",
                title=identification.get("briefTitle"),
                sections=sections,
                metadata=metadata,
            )
            documents.append(document)
        return documents


class OpenFDAAdapter(ResilientHTTPAdapter):
    """Base adapter for OpenFDA endpoints."""

    def __init__(self, *, name: str, endpoint: str, client: HttpClient | None = None) -> None:
        super().__init__(
            name=name,
            base_url="https://api.fda.gov",
            rate_limit_per_second=2,
            retry=_linear_retry_config(5, 1.0),
            client=client,
        )
        self._endpoint = endpoint

    def _query(self, params: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        payload = self._get_json(self._endpoint, params=params)
        return payload.get("results", [])


class OpenFDADrugLabelAdapter(OpenFDAAdapter):
    """Adapter for SPL drug labels."""

    def __init__(self, client: HttpClient | None = None) -> None:
        super().__init__(name="openfda-drug-label", endpoint="/drug/label.json", client=client)

    def fetch(self, context: AdapterContext) -> Iterable[Mapping[str, Any]]:
        ndc = context.parameters.get("ndc")
        set_id = context.parameters.get("set_id")
        params: dict[str, Any] = {"limit": 1}
        if ndc:
            params["search"] = f'openfda.package_ndc:"{validate_ndc(str(ndc))}"'
        elif set_id:
            params["search"] = f'set_id:"{validate_set_id(str(set_id))}"'
        else:  # pragma: no cover - defensive branch
            raise ValueError("Either 'ndc' or 'set_id' parameter must be provided")
        return self._query(params)

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        documents: list[Document] = []
        for payload in payloads:
            openfda = payload.get("openfda", {})
            set_id = payload.get("set_id") or payload.get("id")
            if not set_id:
                raise ValueError("OpenFDA label payload missing set identifier")
            document_id = build_document_id("openfda-label", set_id)
            metadata = {
                "set_id": set_id,
                "brand_name": openfda.get("brand_name", [None])[0],
                "generic_name": openfda.get("generic_name", [None])[0],
                "manufacturer": openfda.get("manufacturer_name", [None])[0],
                "spl_version": payload.get("version"),
                "route": openfda.get("route", []),
            }
            blocks: list[Block] = []
            for key in ("indications_and_usage", "dosage_and_administration", "warnings"):
                text = _to_text(payload.get(key))
                if text:
                    blocks.append(
                        Block(
                            id=f"{key}-block",
                            type=BlockType.PARAGRAPH,
                            text=text,
                            spans=[],
                            metadata={"section": key.replace("_", " ").title()},
                        )
                    )
            section = Section(id="spl", title="Structured Product Label", blocks=blocks)
            documents.append(
                Document(
                    id=document_id,
                    source="openfda-drug-label",
                    title=metadata.get("brand_name") or metadata.get("generic_name"),
                    sections=[section],
                    metadata=metadata,
                )
            )
        return documents


class OpenFDADrugEventAdapter(OpenFDAAdapter):
    """Adapter for adverse event data."""

    def __init__(self, client: HttpClient | None = None) -> None:
        super().__init__(name="openfda-drug-event", endpoint="/drug/event.json", client=client)

    def fetch(self, context: AdapterContext) -> Iterable[Mapping[str, Any]]:
        drug_name = normalize_identifier(_require_parameter(context, "drug"))
        params = {"search": f'patient.drug.medicinalproduct:"{drug_name}"', "limit": 5}
        return self._query(params)

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        documents: list[Document] = []
        for payload in payloads:
            report_id = payload.get("safetyreportid")
            if not report_id:
                continue
            patient = payload.get("patient", {})
            reactions = [
                reaction.get("reactionmeddrapt") for reaction in patient.get("reaction", [])
            ]
            indications = [drug.get("drugindication") for drug in patient.get("drug", [])]
            metadata = {
                "safety_report_id": report_id,
                "received_date": payload.get("receivedate"),
                "reactions": _listify(reactions),
                "indications": _listify(indications),
            }
            summary_text = "; ".join(_listify(reactions)) or "No reactions reported"
            section = Section(
                id="adverse-events",
                title="Adverse Events",
                blocks=[Block(id="adverse-block", text=summary_text, spans=[])],
            )
            documents.append(
                Document(
                    id=build_document_id("openfda-event", report_id),
                    source="openfda-drug-event",
                    title=f"Adverse event report {report_id}",
                    sections=[section],
                    metadata=metadata,
                )
            )
        return documents


class OpenFDADeviceAdapter(OpenFDAAdapter):
    """Adapter for medical device classifications."""

    def __init__(self, client: HttpClient | None = None) -> None:
        super().__init__(
            name="openfda-device", endpoint="/device/classification.json", client=client
        )

    def fetch(self, context: AdapterContext) -> Iterable[Mapping[str, Any]]:
        device_id = _require_parameter(context, "device_id")
        params = {"search": f'product_code:"{device_id}"', "limit": 1}
        return self._query(params)

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        documents: list[Document] = []
        for payload in payloads:
            product_code = payload.get("product_code")
            if not product_code:
                continue
            metadata = {
                "product_code": product_code,
                "device_name": payload.get("device_name"),
                "device_class": payload.get("device_class"),
                "medical_specialty": payload.get("medical_specialty_description"),
            }
            description = payload.get("definition")
            block = Block(id="device-description", text=_to_text(description), spans=[])
            section = Section(id="device", title="Device Details", blocks=[block])
            documents.append(
                Document(
                    id=build_document_id("openfda-device", product_code),
                    source="openfda-device",
                    title=metadata.get("device_name"),
                    sections=[section],
                    metadata=metadata,
                )
            )
        return documents


class OpenAlexAdapter(ResilientHTTPAdapter):
    """Adapter for the OpenAlex works API."""

    def __init__(self, client: HttpClient | None = None) -> None:
        super().__init__(
            name="openalex",
            base_url="https://api.openalex.org",
            rate_limit_per_second=5,
            retry=_linear_retry_config(4, 0.5),
            client=client,
        )

    def fetch(self, context: AdapterContext) -> Iterable[Mapping[str, Any]]:
        doi = context.parameters.get("doi")
        work_id = context.parameters.get("openalex_id")
        query = context.parameters.get("query")
        if doi:
            identifier = validate_doi(str(doi))
            payload = self._get_json(f"/works/https://doi.org/{identifier}")
            return [payload]
        if work_id:
            payload = self._get_json(f"/works/{work_id}")
            return [payload]
        if not query:
            raise ValueError("Either 'doi', 'openalex_id', or 'query' parameter must be provided")
        response = self._get_json("/works", params={"search": query, "per-page": 5})
        return response.get("results", [])

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        documents: list[Document] = []
        for payload in payloads:
            work_id = payload.get("id") or payload.get("openalex_id")
            if not work_id:
                continue
            doi = payload.get("doi") or payload.get("ids", {}).get("doi")
            authors = [
                auth.get("author", {}).get("display_name")
                for auth in payload.get("authorships", [])
            ]
            concepts = [concept.get("display_name") for concept in payload.get("concepts", [])]
            pdf_metadata = self._resolve_pdf_metadata(payload)
            metadata = {
                "openalex_id": work_id,
                "doi": doi,
                "title": payload.get("display_name"),
                "publication_year": payload.get("publication_year"),
                "authorships": _listify(authors),
                "concepts": _listify(concepts),
                "is_open_access": payload.get("open_access", {}).get("is_oa"),
                "cited_by_count": payload.get("cited_by_count"),
            }
            if pdf_metadata:
                metadata.setdefault("pdf", pdf_metadata)
            abstract = payload.get("abstract_inverted_index")
            abstract_text = (
                _flatten_abstract(abstract)
                if isinstance(abstract, Mapping)
                else payload.get("abstract")
            )
            section = Section(
                id="abstract",
                title="Abstract",
                blocks=[Block(id="abstract-block", text=_to_text(abstract_text), spans=[])],
            )
            pdf_url = metadata.get("pdf", {}).get("url") if metadata.get("pdf") else None
            pdf_size = metadata.get("pdf", {}).get("size_bytes") if metadata.get("pdf") else None
            pdf_content_type = metadata.get("pdf", {}).get("content_type") if metadata.get("pdf") else None
            pdf_checksum = metadata.get("pdf", {}).get("checksum") if metadata.get("pdf") else None
            documents.append(
                Document(
                    id=work_id,
                    source="openalex",
                    title=payload.get("display_name"),
                    sections=[section],
                    metadata=metadata,
                    pdf_url=pdf_url,
                    pdf_size_bytes=pdf_size,
                    pdf_content_type=pdf_content_type,
                    pdf_checksum=pdf_checksum,
                )
            )
        return documents

    def _resolve_pdf_metadata(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        location = payload.get("best_oa_location") or {}
        if not isinstance(location, Mapping):
            return None
        url = location.get("url_for_pdf") or location.get("pdf_url")
        if not url:
            return None
        parsed = urlparse(str(url))
        if parsed.scheme not in {"http", "https"}:
            logger.debug("adapter.openalex.pdf_url_unsupported", url=url)
            return None
        size = location.get("pdf_size") or location.get("file_size")
        if isinstance(size, str) and size.isdigit():
            size = int(size)
        elif not isinstance(size, int):
            size = None
        content_type = location.get("pdf_content_type")
        checksum = location.get("pdf_checksum")
        headers: dict[str, str] = {}
        try:
            response = self._client.request("HEAD", url, headers={"Accept": "application/pdf"})
            if response.status_code < 400:
                headers = dict(response.headers)
                if size is None:
                    length = response.headers.get("Content-Length")
                    if length and length.isdigit():
                        size = int(length)
                if not content_type:
                    content_type = response.headers.get("Content-Type")
        except httpx.HTTPError as exc:
            logger.debug("adapter.openalex.pdf_head_failed", url=url, error=str(exc))
        metadata = {
            "url": str(url),
            "size_bytes": size,
            "content_type": content_type,
            "checksum": checksum,
            "headers": headers,
        }
        return metadata


class UnpaywallAdapter(ResilientHTTPAdapter):
    """Adapter for Unpaywall open access status."""

    def __init__(
        self, email: str = "oss@medical-kg.local", client: HttpClient | None = None
    ) -> None:
        super().__init__(
            name="unpaywall",
            base_url="https://api.unpaywall.org/v2",
            rate_limit_per_second=5,
            retry=_linear_retry_config(3, 0.5),
            client=client,
        )
        self._email = email

    def fetch(self, context: AdapterContext) -> Iterable[Mapping[str, Any]]:
        doi = validate_doi(_require_parameter(context, "doi"))
        payload = self._get_json(f"/{doi}", params={"email": self._email})
        return [payload]

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        documents: list[Document] = []
        for payload in payloads:
            doi = payload.get("doi")
            metadata = {
                "doi": doi,
                "is_open_access": payload.get("is_oa"),
                "oa_status": payload.get("oa_status"),
                "journal": payload.get("journal_name"),
                "best_oa_location": payload.get("best_oa_location"),
            }
            location = payload.get("best_oa_location", {})
            text = location.get("url") or "No open access location available"
            section = Section(
                id="unpaywall",
                title="Open Access",
                blocks=[Block(id="oa-block", text=_to_text(text), spans=[])],
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


class CrossrefAdapter(ResilientHTTPAdapter):
    """Adapter for Crossref citation metadata."""

    def __init__(self, client: HttpClient | None = None) -> None:
        super().__init__(
            name="crossref",
            base_url="https://api.crossref.org",
            rate_limit_per_second=4,
            retry=_linear_retry_config(4, 0.5),
            client=client,
        )

    def fetch(self, context: AdapterContext) -> Iterable[Mapping[str, Any]]:
        doi = validate_doi(_require_parameter(context, "doi"))
        payload = self._get_json(f"/works/{doi}")
        message = payload.get("message", {})
        return [message]

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        documents: list[Document] = []
        for message in payloads:
            doi = message.get("DOI")
            title_list = message.get("title", [])
            title = title_list[0] if title_list else None
            references = [ref.get("DOI") for ref in message.get("reference", []) if ref.get("DOI")]
            metadata = {
                "doi": doi,
                "publisher": message.get("publisher"),
                "issued": message.get("issued"),
                "reference_count": message.get("reference-count"),
                "references": _listify(references),
                "is_referenced_by_count": message.get("is-referenced-by-count"),
            }
            block = Block(
                id="crossref-block",
                text=_to_text(message.get("abstract")),
                spans=[],
            )
            section = Section(id="metadata", title="Crossref Metadata", blocks=[block])
            documents.append(
                Document(
                    id=build_document_id("crossref", doi or title or "unknown"),
                    source="crossref",
                    title=title,
                    sections=[section],
                    metadata=metadata,
                )
            )
        return documents


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

    def fetch(self, context: AdapterContext) -> Iterable[Mapping[str, Any]]:
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
                block = Block(id="core-text", text=_to_text(full_text), spans=[])
                section = Section(id="core", title="CORE Full Text", blocks=[block])
                documents.append(
                    Document(
                        id=str(work_id),
                        source="core",
                        title=data.get("title"),
                        sections=[section],
                        metadata={
                            "download_url": data.get("downloadUrl"),
                            "doi": data.get("doi"),
                        },
                    )
                )
        return documents


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

    def fetch(self, context: AdapterContext) -> Iterable[str]:
        pmcid = validate_pmcid(_require_parameter(context, "pmcid"))
        xml_text = self._get_text(f"/webservices/rest/{pmcid}/fullTextXML")
        return [xml_text]

    def parse(self, payloads: Iterable[str], context: AdapterContext) -> Sequence[Document]:
        documents: list[Document] = []
        for xml_text in payloads:
            root = ElementTree.fromstring(xml_text)
            pmcid = root.findtext(".//article-id[@pub-id-type='pmcid']")
            title = _collect_text(root.find(".//article-title"))
            abstract_text = _collect_text(root.find(".//abstract"))
            body_paragraphs = [_collect_text(elem) for elem in root.findall(".//body//p")]
            blocks: list[Block] = []
            if abstract_text:
                blocks.append(Block(id="pmc-abstract", text=abstract_text, spans=[]))
            for idx, paragraph in enumerate(body_paragraphs[:5]):
                blocks.append(Block(id=f"pmc-body-{idx}", text=_to_text(paragraph), spans=[]))
            section = Section(id="pmc", title="Europe PMC", blocks=blocks)
            documents.append(
                Document(
                    id=pmcid or build_document_id("pmc", title or "article"),
                    source="pmc",
                    title=title,
                    sections=[section],
                    metadata={"pmcid": pmcid},
                )
            )
        return documents


class RxNormAdapter(ResilientHTTPAdapter):
    """Adapter for RxNorm normalization."""

    def __init__(self, client: HttpClient | None = None) -> None:
        super().__init__(
            name="rxnorm",
            base_url="https://rxnav.nlm.nih.gov",
            rate_limit_per_second=5,
            retry=_linear_retry_config(3, 0.5),
            client=client,
        )

    def fetch(self, context: AdapterContext) -> Iterable[Mapping[str, Any]]:
        drug_name = normalize_identifier(_require_parameter(context, "drug_name"))
        payload = self._get_json("/REST/drugs", params={"name": drug_name})
        return [payload]

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        documents: list[Document] = []
        for payload in payloads:
            concepts = payload.get("rxnormConceptProperties", [])
            normalized = [concept for concept in concepts if concept.get("rxcui")]
            if not normalized:
                continue
            primary = normalized[0]
            rxcui = validate_rxcui(primary.get("rxcui"))
            block = Block(
                id="rxnorm",
                text=", ".join(
                    sorted({concept.get("name") for concept in normalized if concept.get("name")})
                ),
                spans=[],
            )
            section = Section(id="rxnorm", title="RxNorm Concepts", blocks=[block])
            documents.append(
                Document(
                    id=f"RXCUI:{rxcui}",
                    source="rxnorm",
                    title=primary.get("name"),
                    sections=[section],
                    metadata={
                        "drug_name": context.parameters.get("drug_name"),
                        "concepts": normalized,
                    },
                )
            )
        return documents


class ICD11Adapter(ResilientHTTPAdapter):
    """Adapter for ICD-11 terminology search."""

    def __init__(self, client: HttpClient | None = None) -> None:
        super().__init__(
            name="icd11",
            base_url="https://id.who.int/icd/release/11",
            rate_limit_per_second=3,
            retry=_linear_retry_config(3, 0.5),
            client=client,
        )

    def fetch(self, context: AdapterContext) -> Iterable[Mapping[str, Any]]:
        term = context.parameters.get("code")
        if term:
            code = validate_icd11(str(term))
            payload = self._get_json(f"/{code}")
            return [payload]
        query = _require_parameter(context, "term")
        payload = self._get_json("/search", params={"q": query})
        return payload.get("destinationEntities", [])

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        documents: list[Document] = []
        for entity in payloads:
            code = entity.get("theCode") or entity.get("code")
            if not code:
                continue
            validate_icd11(code)
            title = entity.get("title")
            display = title.get("@value") if isinstance(title, Mapping) else title
            metadata = {
                "code": code,
                "title": display,
                "uri": entity.get("browserUrl"),
            }
            section = Section(
                id="icd11",
                title="ICD-11",
                blocks=[Block(id="icd11-block", text=_to_text(display), spans=[])],
            )
            documents.append(
                Document(
                    id=f"ICD11:{code}",
                    source="icd11",
                    title=display,
                    sections=[section],
                    metadata=metadata,
                )
            )
        return documents


class MeSHAdapter(ResilientHTTPAdapter):
    """Adapter for MeSH descriptor lookups."""

    def __init__(self, client: HttpClient | None = None) -> None:
        super().__init__(
            name="mesh",
            base_url="https://id.nlm.nih.gov/mesh",
            rate_limit_per_second=5,
            retry=_linear_retry_config(3, 0.5),
            client=client,
        )

    def fetch(self, context: AdapterContext) -> Iterable[Mapping[str, Any]]:
        descriptor_id = context.parameters.get("descriptor_id")
        if descriptor_id:
            mesh_id = validate_mesh_id(str(descriptor_id))
            payload = self._get_json(f"/descriptor/{mesh_id}.json")
            return [payload]
        term = _require_parameter(context, "term")
        payload = self._get_json("/lookup", params={"label": term, "limit": 5})
        return payload.get("result", [])

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        documents: list[Document] = []
        for descriptor in payloads:
            identifier = descriptor.get("descriptorUI") or descriptor.get("resource")
            if not identifier:
                continue
            if identifier.startswith("http"):
                mesh_id = identifier.rsplit("/", 1)[-1]
            else:
                mesh_id = identifier
            mesh_id = validate_mesh_id(mesh_id)
            name = descriptor.get("descriptorName") or descriptor.get("label")
            if isinstance(name, Mapping):
                name = name.get("@value")
            tree_numbers = descriptor.get("treeNumberList") or descriptor.get("treeNumber") or []
            metadata = {
                "mesh_id": mesh_id,
                "name": name,
                "tree_numbers": tree_numbers,
            }
            section = Section(
                id="mesh",
                title="MeSH Descriptor",
                blocks=[Block(id="mesh-block", text=_to_text(name), spans=[])],
            )
            documents.append(
                Document(
                    id=f"MeSH:{mesh_id}",
                    source="mesh",
                    title=_to_text(name),
                    sections=[section],
                    metadata=metadata,
                )
            )
        return documents


class ChEMBLAdapter(ResilientHTTPAdapter):
    """Adapter for ChEMBL compound data."""

    def __init__(self, client: HttpClient | None = None) -> None:
        super().__init__(
            name="chembl",
            base_url="https://www.ebi.ac.uk/chembl/api/data",
            rate_limit_per_second=3,
            retry=_linear_retry_config(4, 0.5),
            client=client,
        )

    def fetch(self, context: AdapterContext) -> Iterable[Mapping[str, Any]]:
        chembl_id = context.parameters.get("chembl_id")
        smiles = context.parameters.get("smiles")
        if chembl_id:
            identifier = validate_chembl_id(str(chembl_id))
            payload = self._get_json(f"/molecule/{identifier}")
            return [payload]
        if smiles:
            payload = self._get_json(
                "/molecule", params={"molecule_structures__canonical_smiles__iexact": smiles}
            )
            return payload.get("molecules", [])
        raise ValueError("Either 'chembl_id' or 'smiles' parameter must be provided")

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        documents: list[Document] = []
        for molecule in payloads:
            chembl_id = molecule.get("molecule_chembl_id")
            if not chembl_id:
                continue
            validate_chembl_id(chembl_id)
            properties = molecule.get("molecule_properties", {})
            structures = molecule.get("molecule_structures", {})
            targets = molecule.get("target", molecule.get("targets", []))
            metadata = {
                "chembl_id": chembl_id,
                "pref_name": molecule.get("pref_name"),
                "molecule_type": molecule.get("molecule_type"),
                "molecular_formula": properties.get("full_molformula"),
                "molecular_weight": properties.get("full_mwt"),
                "canonical_smiles": structures.get("canonical_smiles"),
                "targets": targets,
            }
            block = Block(
                id="chembl",
                text=_to_text(structures.get("canonical_smiles")),
                spans=[],
            )
            section = Section(id="chembl", title="ChEMBL", blocks=[block])
            documents.append(
                Document(
                    id=chembl_id,
                    source="chembl",
                    title=molecule.get("pref_name"),
                    sections=[section],
                    metadata=metadata,
                )
            )
        return documents


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

    def fetch(self, context: AdapterContext) -> Iterable[Mapping[str, Any]]:
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
            }
            block = Block(
                id="semantic-scholar",
                text=f"Citations: {payload.get('citationCount', 0)}",  # type: ignore[str-format]
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
