"""OpenFDA adapters for drug labels, adverse events, and device classifications."""

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
from Medical_KG_rev.utils.identifiers import build_document_id, normalize_identifier
from Medical_KG_rev.utils.validation import validate_ndc, validate_set_id


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
        """Execute a query against the OpenFDA endpoint."""
        payload = self._get_json(self._endpoint, params=params)
        return payload.get("results", [])


class OpenFDADrugLabelAdapter(OpenFDAAdapter):
    """Adapter for SPL drug labels."""

    def __init__(self, client: HttpClient | None = None) -> None:
        super().__init__(name="openfda-drug-label", endpoint="/drug/label.json", client=client)

    def fetch(self, context: AdapterContext) -> Iterable[dict[str, Any]]:
        """Fetch drug label data by NDC or set_id."""
        ndc = context.parameters.get("ndc")
        set_id = context.parameters.get("set_id")
        params: dict[str, Any] = {"limit": 1}
        if ndc:
            params["search"] = f'openfda.package_ndc:"{validate_ndc(str(ndc))}"'
        elif set_id:
            params["search"] = f'set_id:"{validate_set_id(str(set_id))}"'
        else:
            raise ValueError("Either 'ndc' or 'set_id' parameter must be provided")
        return self._query(params)

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        """Parse OpenFDA drug label response into documents."""
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
                "source": "openfda-drug-label",
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

    def fetch(self, context: AdapterContext) -> Iterable[dict[str, Any]]:
        """Fetch adverse event data by drug name."""
        drug_name = normalize_identifier(_require_parameter(context, "drug"))
        params = {"search": f'patient.drug.medicinalproduct:"{drug_name}"', "limit": 5}
        return self._query(params)

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        """Parse OpenFDA adverse event response into documents."""
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
                "source": "openfda-drug-event",
            }

            summary_text = "; ".join(_listify(reactions)) or "No reactions reported"
            section = Section(
                id="adverse-events",
                title="Adverse Events",
                blocks=[Block(
                    id="adverse-block",
                    type=BlockType.PARAGRAPH,
                    text=summary_text,
                    spans=[]
                )],
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

    def fetch(self, context: AdapterContext) -> Iterable[dict[str, Any]]:
        """Fetch device classification data by device ID."""
        device_id = _require_parameter(context, "device_id")
        params = {"search": f'product_code:"{device_id}"', "limit": 1}
        return self._query(params)

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        """Parse OpenFDA device response into documents."""
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
                "source": "openfda-device",
            }

            description = payload.get("definition")
            block = Block(
                id="device-description",
                type=BlockType.PARAGRAPH,
                text=_to_text(description),
                spans=[]
            )
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
