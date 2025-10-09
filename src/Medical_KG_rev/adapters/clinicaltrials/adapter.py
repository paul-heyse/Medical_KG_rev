"""ClinicalTrials.gov adapter for clinical trial data."""

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
from Medical_KG_rev.utils.validation import validate_nct_id


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

    def fetch(self, context: AdapterContext) -> Iterable[dict[str, Any]]:
        """Fetch clinical trial data by NCT ID."""
        nct_id = validate_nct_id(_require_parameter(context, "nct_id"))
        payload = self._get_json(f"/studies/{nct_id}", params={"format": "json"})
        study = payload.get("study") or payload
        return [study]

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        """Parse ClinicalTrials.gov response into documents."""
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
                "source": "clinicaltrials",
            }

            sections: list[Section] = []
            summary_text = description_module.get("briefSummary")
            if summary_text:
                sections.append(
                    Section(
                        id="summary",
                        title="Brief Summary",
                        blocks=[Block(
                            id="summary-block",
                            type=BlockType.PARAGRAPH,
                            text=_to_text(summary_text),
                            spans=[]
                        )],
                    )
                )
            detailed_text = description_module.get("detailedDescription")
            if detailed_text:
                sections.append(
                    Section(
                        id="description",
                        title="Detailed Description",
                        blocks=[
                            Block(
                                id="description-block",
                                type=BlockType.PARAGRAPH,
                                text=_to_text(detailed_text),
                                spans=[]
                            )
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
