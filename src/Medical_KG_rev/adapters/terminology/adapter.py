"""Terminology adapters for medical vocabularies.

This module provides adapters for medical terminology systems including
RxNorm (drug names), ICD-11 (disease codes), MeSH (medical subject headings),
and ChEMBL (chemical compounds). These adapters fetch terminology data
and transform it into structured Document objects.

Key Responsibilities:
    - Fetch terminology data from medical vocabulary APIs
    - Validate medical identifiers (RxCUI, ICD-11, MeSH, ChEMBL)
    - Transform terminology responses into Document objects
    - Handle rate limiting and API authentication
    - Support multiple terminology systems

Collaborators:
    - Upstream: Medical terminology APIs (RxNorm, WHO ICD-11, PubMed MeSH, ChEMBL)
    - Downstream: Document models, HTTP client

Side Effects:
    - Makes HTTP requests to medical terminology APIs
    - Validates medical identifiers
    - Creates Document objects with terminology data

Thread Safety:
    - Thread-safe: Stateless adapters with no shared mutable state

Performance Characteristics:
    - Rate limiting: 3-5 requests per second depending on API
    - Retry strategy: Linear backoff with 3 attempts
    - Response parsing: O(n) where n is response size

Example:
    >>> adapter = RxNormAdapter()
    >>> context = AdapterContext(
    ...     tenant_id="tenant1",
    ...     domain="medical",
    ...     correlation_id="corr1",
    ...     parameters={"rxcui": "12345"}
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
from Medical_KG_rev.utils.identifiers import normalize_identifier
from Medical_KG_rev.utils.validation import (
    validate_chembl_id,
    validate_icd11,
    validate_mesh_id,
    validate_rxcui,
)

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

# ==============================================================================
# DATA MODELS
# ==============================================================================

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
        response = self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._owns_client:
            self._client.close()


# ==============================================================================
# ERROR HANDLING
# ==============================================================================

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

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

    def fetch(self, context: AdapterContext) -> Iterable[dict[str, Any]]:
        """Fetch RxNorm concept data by drug name."""
        drug_name = normalize_identifier(_require_parameter(context, "drug_name"))
        payload = self._get_json("/REST/drugs", params={"name": drug_name})
        return [payload]

    def parse(
        self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext
    ) -> Sequence[Document]:
        """Parse RxNorm response into documents."""
        documents: list[Document] = []
        for payload in payloads:
            concepts = payload.get("rxnormConceptProperties", [])
            normalized = [concept for concept in concepts if concept.get("rxcui")]
            if not normalized:
                continue
            primary = normalized[0]
            rxcui = validate_rxcui(primary.get("rxcui"))

            metadata = {
                "drug_name": context.parameters.get("drug_name"),
                "concepts": normalized,
                "source": "rxnorm",
            }

            block = Block(
                id="rxnorm",
                type=BlockType.PARAGRAPH,
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
                    metadata=metadata,
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

    def fetch(self, context: AdapterContext) -> Iterable[dict[str, Any]]:
        """Fetch ICD-11 data by code or search term."""
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
        """Parse ICD-11 response into documents."""
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
                "source": "icd11",
            }

            section = Section(
                id="icd11",
                title="ICD-11",
                blocks=[
                    Block(
                        id="icd11-block", type=BlockType.PARAGRAPH, text=_to_text(display), spans=[]
                    )
                ],
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

    def fetch(self, context: AdapterContext) -> Iterable[dict[str, Any]]:
        """Fetch MeSH data by descriptor ID or search term."""
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
        """Parse MeSH response into documents."""
        documents: list[Document] = []
        for descriptor in payloads:
            identifier = descriptor.get("descriptorUI") or descriptor.get("resource")
            if not identifier:
                continue
            mesh_id = identifier.rsplit("/", 1)[-1] if identifier.startswith("http") else identifier
            mesh_id = validate_mesh_id(mesh_id)
            name = descriptor.get("descriptorName") or descriptor.get("label")
            if isinstance(name, Mapping):
                name = name.get("@value")
            tree_numbers = descriptor.get("treeNumberList") or descriptor.get("treeNumber") or []

            metadata = {
                "mesh_id": mesh_id,
                "name": name,
                "tree_numbers": tree_numbers,
                "source": "mesh",
            }

            section = Section(
                id="mesh",
                title="MeSH Descriptor",
                blocks=[
                    Block(id="mesh-block", type=BlockType.PARAGRAPH, text=_to_text(name), spans=[])
                ],
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

    def fetch(self, context: AdapterContext) -> Iterable[dict[str, Any]]:
        """Fetch ChEMBL data by ChEMBL ID or SMILES."""
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
        """Parse ChEMBL response into documents."""
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
                "source": "chembl",
            }

            block = Block(
                id="chembl",
                type=BlockType.PARAGRAPH,
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


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "ChEMBLAdapter",
    "ICD11Adapter",
    "MeSHAdapter",
    "ResilientHTTPAdapter",
    "RxNormAdapter",
]
