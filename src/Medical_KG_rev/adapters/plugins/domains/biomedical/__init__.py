"""Biomedical adapter plugins implemented on top of the legacy adapters."""

from __future__ import annotations

from functools import lru_cache

from Medical_KG_rev.adapters.base import AdapterContext, BaseAdapter
from Medical_KG_rev.adapters.biomedical import (
    COREAdapter,
    ChEMBLAdapter,
    ClinicalTrialsAdapter,
    CrossrefAdapter,
    ICD11Adapter,
    MeSHAdapter,
    OpenAlexAdapter,
    OpenFDADeviceAdapter,
    OpenFDADrugEventAdapter,
    OpenFDADrugLabelAdapter,
    PMCAdapter,
    RxNormAdapter,
    SemanticScholarAdapter,
    UnpaywallAdapter,
)
from Medical_KG_rev.adapters.plugins.base import BaseAdapterPlugin
from Medical_KG_rev.adapters.plugins.manager import AdapterPluginManager
from Medical_KG_rev.adapters.plugins.models import AdapterRequest, AdapterResponse, ValidationOutcome

from ..metadata import BiomedicalAdapterMetadata


def _context_from_request(request: AdapterRequest) -> AdapterContext:
    return AdapterContext(
        tenant_id=request.tenant_id,
        domain=request.domain.value,
        correlation_id=request.correlation_id,
        parameters=request.parameters,
    )


def _fetch_from_adapter(adapter: BaseAdapter, request: AdapterRequest) -> AdapterResponse:
    context = _context_from_request(request)
    payloads = list(adapter.fetch(context))
    metadata = {
        "adapter": adapter.name,
        "tenant_id": request.tenant_id,
        "correlation_id": request.correlation_id,
    }
    return AdapterResponse(items=payloads, metadata=metadata)


def _parse_with_adapter(
    adapter: BaseAdapter, response: AdapterResponse, request: AdapterRequest
) -> AdapterResponse:
    context = _context_from_request(request)
    documents = list(adapter.parse(response.items, context))
    response.items = documents
    document_ids = [getattr(document, "id", None) for document in documents]
    document_ids = [doc_id for doc_id in document_ids if doc_id is not None]
    if document_ids:
        response.metadata.setdefault("documents", document_ids)
    return response


def _validate_with_adapter(
    adapter: BaseAdapter, response: AdapterResponse, request: AdapterRequest
) -> ValidationOutcome:
    context = _context_from_request(request)
    warnings = list(adapter.validate(response.items, context))
    response.warnings.extend(warnings)
    adapter.write(response.items, context)
    return ValidationOutcome(valid=True, warnings=response.warnings)


class _BiomedicalAdapterPlugin(BaseAdapterPlugin):
    """Specialised plugin base class wiring a legacy biomedical adapter."""

    legacy_adapter_cls: type[BaseAdapter]

    def __init__(
        self,
        config=None,
        *,
        adapter: BaseAdapter | None = None,
    ) -> None:
        super().__init__(config=config)
        self._adapter = adapter or self.legacy_adapter_cls()

    @property
    def adapter(self) -> BaseAdapter:
        return self._adapter

    def fetch(self, request: AdapterRequest) -> AdapterResponse:
        return _fetch_from_adapter(self.adapter, request)

    def parse(self, response: AdapterResponse, request: AdapterRequest) -> AdapterResponse:
        return _parse_with_adapter(self.adapter, response, request)

    def validate(self, response: AdapterResponse, request: AdapterRequest) -> ValidationOutcome:
        return _validate_with_adapter(self.adapter, response, request)


class ClinicalTrialsAdapterPlugin(_BiomedicalAdapterPlugin):
    legacy_adapter_cls = ClinicalTrialsAdapter
    metadata = BiomedicalAdapterMetadata(
        name="clinicaltrials",
        version="1.0.0",
        summary="ClinicalTrials.gov API v2 integration",
        capabilities=["studies", "trial-metadata"],
        maintainer="Data Platform",
        dataset="clinical_trials",
        data_products=["ResearchStudy"],
        compliance=["HIPAA"],
    )


class OpenFDADrugLabelAdapterPlugin(_BiomedicalAdapterPlugin):
    legacy_adapter_cls = OpenFDADrugLabelAdapter
    metadata = BiomedicalAdapterMetadata(
        name="openfda-drug-label",
        version="1.0.0",
        summary="OpenFDA structured product label adapter",
        capabilities=["drug-labels"],
        maintainer="Data Platform",
        dataset="openfda_labels",
        data_products=["MedicalDocument"],
        compliance=["FDA"],
    )


class OpenFDADrugEventAdapterPlugin(_BiomedicalAdapterPlugin):
    legacy_adapter_cls = OpenFDADrugEventAdapter
    metadata = BiomedicalAdapterMetadata(
        name="openfda-drug-event",
        version="1.0.0",
        summary="OpenFDA adverse event reporting adapter",
        capabilities=["drug-events"],
        maintainer="Data Platform",
        dataset="openfda_events",
        data_products=["MedicalDocument"],
        compliance=["FDA"],
    )


class OpenFDADeviceAdapterPlugin(_BiomedicalAdapterPlugin):
    legacy_adapter_cls = OpenFDADeviceAdapter
    metadata = BiomedicalAdapterMetadata(
        name="openfda-device",
        version="1.0.0",
        summary="OpenFDA device classification adapter",
        capabilities=["device-classification"],
        maintainer="Data Platform",
        dataset="openfda_device",
        data_products=["MedicalDocument"],
        compliance=["FDA"],
    )


class OpenAlexAdapterPlugin(_BiomedicalAdapterPlugin):
    legacy_adapter_cls = OpenAlexAdapter
    metadata = BiomedicalAdapterMetadata(
        name="openalex",
        version="1.0.0",
        summary="OpenAlex scholarly works adapter",
        capabilities=["works", "authors"],
        maintainer="Data Platform",
        dataset="openalex",
        data_products=["ResearchStudy"],
    )


class UnpaywallAdapterPlugin(_BiomedicalAdapterPlugin):
    legacy_adapter_cls = UnpaywallAdapter
    metadata = BiomedicalAdapterMetadata(
        name="unpaywall",
        version="1.0.0",
        summary="Unpaywall open access metadata adapter",
        capabilities=["oa-status"],
        maintainer="Data Platform",
        dataset="unpaywall",
        data_products=["ResearchStudy"],
    )


class CrossrefAdapterPlugin(_BiomedicalAdapterPlugin):
    legacy_adapter_cls = CrossrefAdapter
    metadata = BiomedicalAdapterMetadata(
        name="crossref",
        version="1.0.0",
        summary="Crossref bibliographic metadata adapter",
        capabilities=["works"],
        maintainer="Data Platform",
        dataset="crossref",
        data_products=["ResearchStudy"],
    )


class COREAdapterPlugin(_BiomedicalAdapterPlugin):
    legacy_adapter_cls = COREAdapter
    metadata = BiomedicalAdapterMetadata(
        name="core",
        version="1.0.0",
        summary="CORE open research adapter",
        capabilities=["articles"],
        maintainer="Data Platform",
        dataset="core",
        data_products=["ResearchStudy"],
    )


class PMCAdapterPlugin(_BiomedicalAdapterPlugin):
    legacy_adapter_cls = PMCAdapter
    metadata = BiomedicalAdapterMetadata(
        name="pmc",
        version="1.0.0",
        summary="PubMed Central adapter",
        capabilities=["articles"],
        maintainer="Data Platform",
        dataset="pmc",
        data_products=["MedicalDocument"],
    )


class RxNormAdapterPlugin(_BiomedicalAdapterPlugin):
    legacy_adapter_cls = RxNormAdapter
    metadata = BiomedicalAdapterMetadata(
        name="rxnorm",
        version="1.0.0",
        summary="RxNorm terminology adapter",
        capabilities=["drug-terminology"],
        maintainer="Data Platform",
        dataset="rxnorm",
        data_products=["MedicalDocument"],
    )


class ICD11AdapterPlugin(_BiomedicalAdapterPlugin):
    legacy_adapter_cls = ICD11Adapter
    metadata = BiomedicalAdapterMetadata(
        name="icd11",
        version="1.0.0",
        summary="ICD-11 terminology adapter",
        capabilities=["disease-terminology"],
        maintainer="Data Platform",
        dataset="icd11",
        data_products=["MedicalDocument"],
    )


class MeSHAdapterPlugin(_BiomedicalAdapterPlugin):
    legacy_adapter_cls = MeSHAdapter
    metadata = BiomedicalAdapterMetadata(
        name="mesh",
        version="1.0.0",
        summary="MeSH terminology adapter",
        capabilities=["subject-headings"],
        maintainer="Data Platform",
        dataset="mesh",
        data_products=["MedicalDocument"],
    )


class ChEMBLAdapterPlugin(_BiomedicalAdapterPlugin):
    legacy_adapter_cls = ChEMBLAdapter
    metadata = BiomedicalAdapterMetadata(
        name="chembl",
        version="1.0.0",
        summary="ChEMBL compound adapter",
        capabilities=["compound"],
        maintainer="Data Platform",
        dataset="chembl",
        data_products=["MedicalDocument"],
    )


class SemanticScholarAdapterPlugin(_BiomedicalAdapterPlugin):
    legacy_adapter_cls = SemanticScholarAdapter
    metadata = BiomedicalAdapterMetadata(
        name="semanticscholar",
        version="1.0.0",
        summary="Semantic Scholar metadata adapter",
        capabilities=["works"],
        maintainer="Data Platform",
        dataset="semanticscholar",
        data_products=["ResearchStudy"],
    )


BIOMEDICAL_PLUGINS: tuple[type[BaseAdapterPlugin], ...] = (
    ClinicalTrialsAdapterPlugin,
    OpenFDADrugLabelAdapterPlugin,
    OpenFDADrugEventAdapterPlugin,
    OpenFDADeviceAdapterPlugin,
    OpenAlexAdapterPlugin,
    UnpaywallAdapterPlugin,
    CrossrefAdapterPlugin,
    COREAdapterPlugin,
    PMCAdapterPlugin,
    RxNormAdapterPlugin,
    ICD11AdapterPlugin,
    MeSHAdapterPlugin,
    ChEMBLAdapterPlugin,
    SemanticScholarAdapterPlugin,
)


@lru_cache(maxsize=1)
def builtin_biomedical_plugins() -> tuple[BaseAdapterPlugin, ...]:
    """Instantiate bundled biomedical plugins once for reuse."""

    return tuple(plugin() for plugin in BIOMEDICAL_PLUGINS)


def register_biomedical_plugins(manager: AdapterPluginManager) -> list[BiomedicalAdapterMetadata]:
    """Register bundled biomedical adapters with a plugin manager."""

    registrations: list[BiomedicalAdapterMetadata] = []
    for plugin in builtin_biomedical_plugins():
        metadata = manager.register(plugin)
        registrations.append(BiomedicalAdapterMetadata(**metadata.model_dump()))
    return registrations


__all__ = [
    "BIOMEDICAL_PLUGINS",
    "builtin_biomedical_plugins",
    "register_biomedical_plugins",
    "ClinicalTrialsAdapterPlugin",
    "OpenFDADrugLabelAdapterPlugin",
    "OpenFDADrugEventAdapterPlugin",
    "OpenFDADeviceAdapterPlugin",
    "OpenAlexAdapterPlugin",
    "UnpaywallAdapterPlugin",
    "CrossrefAdapterPlugin",
    "COREAdapterPlugin",
    "PMCAdapterPlugin",
    "RxNormAdapterPlugin",
    "ICD11AdapterPlugin",
    "MeSHAdapterPlugin",
    "ChEMBLAdapterPlugin",
    "SemanticScholarAdapterPlugin",
]
