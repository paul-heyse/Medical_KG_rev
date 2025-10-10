"""Tests for extracted biomedical adapters."""

import httpx

from Medical_KG_rev.adapters import (
    AdapterDomain,
    AdapterRequest,
)
from Medical_KG_rev.adapters.clinicaltrials import ClinicalTrialsAdapter
from Medical_KG_rev.adapters.core import COREAdapter
from Medical_KG_rev.adapters.crossref import CrossrefAdapter
from Medical_KG_rev.adapters.openfda import (
    OpenFDADeviceAdapter,
    OpenFDADrugEventAdapter,
    OpenFDADrugLabelAdapter,
)
from Medical_KG_rev.adapters.plugins.base import BaseAdapterPlugin
from Medical_KG_rev.adapters.plugins.domains.biomedical import (
    ChEMBLAdapterPlugin,
    ClinicalTrialsAdapterPlugin,
    COREAdapterPlugin,
    CrossrefAdapterPlugin,
    ICD11AdapterPlugin,
    MeSHAdapterPlugin,
    OpenFDADeviceAdapterPlugin,
    OpenFDADrugEventAdapterPlugin,
    OpenFDADrugLabelAdapterPlugin,
    PMCAdapterPlugin,
    RxNormAdapterPlugin,
    SemanticScholarAdapterPlugin,
    UnpaywallAdapterPlugin,
)
from Medical_KG_rev.adapters.plugins.models import AdapterResponse
from Medical_KG_rev.adapters.pmc import PMCAdapter
from Medical_KG_rev.adapters.semanticscholar import SemanticScholarAdapter
from Medical_KG_rev.adapters.terminology import (
    ChEMBLAdapter,
    ICD11Adapter,
    MeSHAdapter,
    RxNormAdapter,
)
from Medical_KG_rev.adapters.unpaywall import UnpaywallAdapter
from Medical_KG_rev.utils.http_client import BackoffStrategy, HttpClient, RetryConfig


def _client(base_url: str, handler: httpx.MockTransport) -> HttpClient:
    return HttpClient(
        base_url=base_url,
        retry=RetryConfig(attempts=2, backoff_strategy=BackoffStrategy.NONE, jitter=False),
        transport=handler,
    )


def _mock_transport(callback) -> httpx.MockTransport:
    return httpx.MockTransport(callback)


def _run_plugin(
    plugin: BaseAdapterPlugin,
    *,
    parameters: dict[str, object],
    domain: AdapterDomain = AdapterDomain.BIOMEDICAL,
) -> AdapterResponse:
    request = AdapterRequest(
        tenant_id="tenant",
        correlation_id="corr",
        domain=domain,
        parameters=parameters,
    )
    response = plugin.fetch(request)
    response = plugin.parse(response, request)
    outcome = plugin.validate(response, request)
    assert outcome.valid
    return response


def test_unpaywall_adapter_v2_surfaces_pdf_metadata() -> None:
    """Test that the extracted Unpaywall adapter surfaces PDF metadata correctly."""
    payload = {
        "doi": "10.1000/example",
        "title": "Sample Paper",
        "is_oa": True,
        "oa_status": "gold",
        "best_oa_location": {"url": "https://example.com/paper.pdf"},
    }

    plugin = UnpaywallAdapterPlugin(
        adapter=UnpaywallAdapter(
            client=_client(
                "https://api.unpaywall.org/v2",
                _mock_transport(lambda request: httpx.Response(200, json=payload)),
            )
        )
    )
    response = _run_plugin(plugin, parameters={"doi": "10.1000/example"})
    document = response.items[0]

    assert document.metadata["oa_status"] == "gold"
    assert document.metadata["pdf_urls"] == ["https://example.com/paper.pdf"]
    assert document.metadata["document_type"] == "pdf"
    assert document.source == "unpaywall"


def test_crossref_adapter_v2_extracts_references() -> None:
    """Test that the extracted Crossref adapter extracts references correctly."""
    payload = {
        "message": {
            "DOI": "10.1000/example",
            "title": ["Study"],
            "publisher": "Publisher",
            "reference-count": 2,
            "reference": [{"DOI": "10.1000/ref1"}, {"DOI": "10.1000/ref2"}],
        }
    }

    plugin = CrossrefAdapterPlugin(
        adapter=CrossrefAdapter(
            client=_client(
                "https://api.crossref.org",
                _mock_transport(lambda request: httpx.Response(200, json=payload)),
            )
        )
    )
    response = _run_plugin(plugin, parameters={"doi": "10.1000/example"})
    document = response.items[0]

    assert document.metadata["references"] == ["10.1000/ref1", "10.1000/ref2"]
    assert document.source == "crossref"


def test_pmc_adapter_v2_parses_xml() -> None:
    """Test that the extracted PMC adapter parses XML correctly."""
    xml_payload = """
    <article>
        <front>
            <article-meta>
                <title-group><article-title>Article Title</article-title></title-group>
            </article-meta>
        </front>
        <abstract><p>Abstract text</p></abstract>
        <body><p>Body paragraph one.</p><p>Body paragraph two.</p></body>
        <article-id pub-id-type="pmcid">PMC123456</article-id>
    </article>
    """

    plugin = PMCAdapterPlugin(
        adapter=PMCAdapter(
            client=_client(
                "https://www.ebi.ac.uk/europepmc",
                _mock_transport(lambda request: httpx.Response(200, text=xml_payload)),
            )
        )
    )
    response = _run_plugin(plugin, parameters={"pmcid": "PMC123456"})
    document = response.items[0]

    assert document.metadata["pmcid"] == "PMC123456"
    assert document.metadata["document_type"] == "xml"
    assert any(block.text == "Body paragraph one." for block in document.sections[0].blocks)
    assert document.source == "pmc"


def test_core_adapter_v2_handles_multiple_entries() -> None:
    """Test that the extracted CORE adapter handles multiple entries correctly."""
    payload = {
        "data": [
            {
                "id": "core-1",
                "title": "Article",
                "fullText": "Full text",
                "downloadUrl": "https://core.example/download.pdf",
                "doi": "10.1000/example",
            }
        ]
    }

    plugin = COREAdapterPlugin(
        adapter=COREAdapter(
            client=_client(
                "https://core.ac.uk/api-v3",
                _mock_transport(lambda request: httpx.Response(200, json=payload)),
            )
        )
    )
    response = _run_plugin(plugin, parameters={"doi": "10.1000/example"})
    document = response.items[0]

    assert document.metadata["download_url"] == "https://core.example/download.pdf"
    assert document.metadata["pdf_urls"] == ["https://core.example/download.pdf"]
    assert document.metadata["document_type"] == "pdf"
    assert document.source == "core"


def test_clinical_trials_adapter_v2_maps_metadata() -> None:
    """Test that the extracted ClinicalTrials adapter maps metadata correctly."""
    study_payload = {
        "study": {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT01234567",
                    "briefTitle": "Immunotherapy Study",
                },
                "statusModule": {
                    "overallStatus": "Recruiting",
                    "startDateStruct": {"date": "2024-01-01"},
                },
                "designModule": {"studyType": "Interventional", "phase": "Phase 2"},
                "armsInterventionsModule": {"interventions": [{"type": "Drug", "name": "Drug A"}]},
                "outcomesModule": {"primaryOutcomes": [{"measure": "Progression-free survival"}]},
                "eligibilityModule": {
                    "eligibilityCriteria": "Adults 18-65",
                    "sex": "ALL",
                    "minimumAge": "18 Years",
                    "maximumAge": "65 Years",
                },
                "descriptionModule": {
                    "briefSummary": "Short overview",
                    "detailedDescription": "Detailed description of the study",
                },
            }
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/studies/NCT01234567")
        return httpx.Response(200, json=study_payload)

    plugin = ClinicalTrialsAdapterPlugin(
        adapter=ClinicalTrialsAdapter(
            client=_client("https://clinicaltrials.gov/api/v2", _mock_transport(handler))
        )
    )
    response = _run_plugin(plugin, parameters={"nct_id": "NCT01234567"})
    document = response.items[0]

    assert document.id == "NCT01234567"
    assert document.metadata["overall_status"] == "Recruiting"
    assert document.metadata["phase"] == "Phase 2"
    assert "Drug: Drug A" in document.metadata["interventions"]
    assert document.sections[0].blocks[0].text == "Short overview"
    assert document.source == "clinicaltrials"


def test_openfda_drug_label_adapter_v2_parses_spl() -> None:
    """Test that the extracted OpenFDA drug label adapter parses SPL correctly."""
    payload = {
        "results": [
            {
                "set_id": "abcd1234",
                "version": "1",
                "openfda": {
                    "brand_name": ["DrugName"],
                    "generic_name": ["Generic"],
                    "manufacturer_name": ["Manufacturer"],
                    "route": ["ORAL"],
                },
                "indications_and_usage": "Used for treatment",
                "warnings": "Warning text",
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert "openfda.package_ndc" in request.url.params["search"]
        return httpx.Response(200, json=payload)

    plugin = OpenFDADrugLabelAdapterPlugin(
        adapter=OpenFDADrugLabelAdapter(
            client=_client("https://api.fda.gov", _mock_transport(handler))
        )
    )
    response = _run_plugin(plugin, parameters={"ndc": "1234-5678-90"})
    document = response.items[0]

    assert document.metadata["brand_name"] == "DrugName"
    assert any(
        block.metadata["section"] == "Indications And Usage"
        for block in document.sections[0].blocks
    )
    assert document.source == "openfda-drug-label"


def test_openfda_drug_event_adapter_v2_maps_reactions() -> None:
    """Test that the extracted OpenFDA drug event adapter maps reactions correctly."""
    payload = {
        "results": [
            {
                "safetyreportid": "123",
                "receivedate": "20240101",
                "patient": {
                    "reaction": [{"reactionmeddrapt": "Headache"}],
                    "drug": [{"drugindication": "Pain"}],
                },
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["search"].startswith("patient.drug.medicinalproduct")
        return httpx.Response(200, json=payload)

    plugin = OpenFDADrugEventAdapterPlugin(
        adapter=OpenFDADrugEventAdapter(
            client=_client("https://api.fda.gov", _mock_transport(handler))
        )
    )
    response = _run_plugin(plugin, parameters={"drug": "Acetaminophen"})
    document = response.items[0]

    assert "Headache" in document.metadata["reactions"]
    assert "Adverse event report" in document.title
    assert document.source == "openfda-drug-event"


def test_openfda_device_adapter_v2_handles_definition() -> None:
    """Test that the extracted OpenFDA device adapter handles definition correctly."""
    payload = {
        "results": [
            {
                "product_code": "ABC",
                "device_name": "Cardiac Monitor",
                "device_class": "2",
                "medical_specialty_description": "Cardiology",
                "definition": "Device description",
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["search"].startswith("product_code:")
        return httpx.Response(200, json=payload)

    plugin = OpenFDADeviceAdapterPlugin(
        adapter=OpenFDADeviceAdapter(
            client=_client("https://api.fda.gov", _mock_transport(handler))
        )
    )
    response = _run_plugin(plugin, parameters={"device_id": "ABC"})
    document = response.items[0]

    assert document.metadata["medical_specialty"] == "Cardiology"
    assert document.sections[0].blocks[0].text == "Device description"
    assert document.source == "openfda-device"


def test_rxnorm_adapter_v2_normalizes_name() -> None:
    """Test that the extracted RxNorm adapter normalizes names correctly."""
    payload = {
        "rxnormConceptProperties": [
            {"rxcui": "83367", "name": "Atorvastatin", "synonym": "Atorvastatin"}
        ]
    }

    plugin = RxNormAdapterPlugin(
        adapter=RxNormAdapter(
            client=_client(
                "https://rxnav.nlm.nih.gov",
                _mock_transport(lambda request: httpx.Response(200, json=payload)),
            )
        )
    )
    response = _run_plugin(plugin, parameters={"drug_name": "Atorvastatin"})
    document = response.items[0]

    assert document.id == "RXCUI:83367"
    assert document.source == "rxnorm"


def test_icd11_adapter_v2_searches_term() -> None:
    """Test that the extracted ICD11 adapter searches terms correctly."""
    payload = {
        "destinationEntities": [
            {
                "theCode": "5A11",
                "title": {"@value": "Type 2 diabetes mellitus"},
                "browserUrl": "https://icd.who.int/browse11/l-m/en#/http://id.who.int/icd/entity/12345",
            }
        ]
    }

    plugin = ICD11AdapterPlugin(
        adapter=ICD11Adapter(
            client=_client(
                "https://id.who.int/icd/release/11",
                _mock_transport(lambda request: httpx.Response(200, json=payload)),
            )
        )
    )
    response = _run_plugin(plugin, parameters={"term": "diabetes"})
    document = response.items[0]

    assert document.metadata["code"] == "5A11"
    assert document.source == "icd11"


def test_mesh_adapter_v2_descriptor_lookup() -> None:
    """Test that the extracted MeSH adapter performs descriptor lookup correctly."""
    payload = {
        "result": [
            {
                "descriptorUI": "D000001",
                "descriptorName": {"@value": "Calcimycin"},
                "treeNumberList": ["D03.633.100"],
            }
        ]
    }

    plugin = MeSHAdapterPlugin(
        adapter=MeSHAdapter(
            client=_client(
                "https://id.nlm.nih.gov/mesh",
                _mock_transport(lambda request: httpx.Response(200, json=payload)),
            )
        )
    )
    response = _run_plugin(plugin, parameters={"term": "Calcimycin"})
    document = response.items[0]

    assert document.metadata["mesh_id"] == "D000001"
    assert document.source == "mesh"


def test_chembl_adapter_v2_fetches_by_identifier() -> None:
    """Test that the extracted ChEMBL adapter fetches by identifier correctly."""
    payload = {
        "molecule_chembl_id": "CHEMBL25",
        "pref_name": "Aspirin",
        "molecule_properties": {"full_molformula": "C9H8O4"},
        "molecule_structures": {"canonical_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
    }

    plugin = ChEMBLAdapterPlugin(
        adapter=ChEMBLAdapter(
            client=_client(
                "https://www.ebi.ac.uk/chembl/api/data",
                _mock_transport(lambda request: httpx.Response(200, json=payload)),
            )
        )
    )
    response = _run_plugin(plugin, parameters={"chembl_id": "CHEMBL25"})
    document = response.items[0]

    assert document.metadata["canonical_smiles"] == "CC(=O)OC1=CC=CC=C1C(=O)O"
    assert document.source == "chembl"


def test_semantic_scholar_adapter_v2_enriches_citations() -> None:
    """Test that the extracted Semantic Scholar adapter enriches citations correctly."""
    payload = {
        "paperId": "abc123",
        "title": "Paper",
        "externalIds": {"DOI": "10.1000/example"},
        "citationCount": 42,
        "referenceCount": 10,
        "references": [{"title": "Ref 1"}, {"paperId": "xyz"}],
    }

    plugin = SemanticScholarAdapterPlugin(
        adapter=SemanticScholarAdapter(
            client=_client(
                "https://api.semanticscholar.org/graph/v1",
                _mock_transport(lambda request: httpx.Response(200, json=payload)),
            )
        )
    )
    response = _run_plugin(plugin, parameters={"doi": "10.1000/example"})
    document = response.items[0]

    assert document.metadata["citation_count"] == 42
    assert document.source == "semantic-scholar"


def test_extracted_adapters_backward_compatibility() -> None:
    """Test that extracted adapters maintain backward compatibility."""
    # Test that we can still import from the old biomedical module
    from Medical_KG_rev.adapters.biomedical import (
        ChEMBLAdapter as LegacyChEMBLAdapter,
    )
    from Medical_KG_rev.adapters.biomedical import (
        ClinicalTrialsAdapter as LegacyClinicalTrialsAdapter,
    )
    from Medical_KG_rev.adapters.biomedical import (
        COREAdapter as LegacyCOREAdapter,
    )
    from Medical_KG_rev.adapters.biomedical import (
        CrossrefAdapter as LegacyCrossrefAdapter,
    )
    from Medical_KG_rev.adapters.biomedical import (
        ICD11Adapter as LegacyICD11Adapter,
    )
    from Medical_KG_rev.adapters.biomedical import (
        MeSHAdapter as LegacyMeSHAdapter,
    )
    from Medical_KG_rev.adapters.biomedical import (
        OpenFDADeviceAdapter as LegacyOpenFDADeviceAdapter,
    )
    from Medical_KG_rev.adapters.biomedical import (
        OpenFDADrugEventAdapter as LegacyOpenFDADrugEventAdapter,
    )
    from Medical_KG_rev.adapters.biomedical import (
        OpenFDADrugLabelAdapter as LegacyOpenFDADrugLabelAdapter,
    )
    from Medical_KG_rev.adapters.biomedical import (
        PMCAdapter as LegacyPMCAdapter,
    )
    from Medical_KG_rev.adapters.biomedical import (
        RxNormAdapter as LegacyRxNormAdapter,
    )
    from Medical_KG_rev.adapters.biomedical import (
        SemanticScholarAdapter as LegacySemanticScholarAdapter,
    )
    from Medical_KG_rev.adapters.biomedical import (
        UnpaywallAdapter as LegacyUnpaywallAdapter,
    )

    # Test that we can import from the new modules
    from Medical_KG_rev.adapters.clinicaltrials import (
        ClinicalTrialsAdapter as NewClinicalTrialsAdapter,
    )
    from Medical_KG_rev.adapters.core import COREAdapter as NewCOREAdapter
    from Medical_KG_rev.adapters.crossref import CrossrefAdapter as NewCrossrefAdapter
    from Medical_KG_rev.adapters.openfda import (
        OpenFDADeviceAdapter as NewOpenFDADeviceAdapter,
    )
    from Medical_KG_rev.adapters.openfda import (
        OpenFDADrugEventAdapter as NewOpenFDADrugEventAdapter,
    )
    from Medical_KG_rev.adapters.openfda import (
        OpenFDADrugLabelAdapter as NewOpenFDADrugLabelAdapter,
    )
    from Medical_KG_rev.adapters.pmc import PMCAdapter as NewPMCAdapter
    from Medical_KG_rev.adapters.semanticscholar import (
        SemanticScholarAdapter as NewSemanticScholarAdapter,
    )
    from Medical_KG_rev.adapters.terminology import (
        ChEMBLAdapter as NewChEMBLAdapter,
    )
    from Medical_KG_rev.adapters.terminology import (
        ICD11Adapter as NewICD11Adapter,
    )
    from Medical_KG_rev.adapters.terminology import (
        MeSHAdapter as NewMeSHAdapter,
    )
    from Medical_KG_rev.adapters.terminology import (
        RxNormAdapter as NewRxNormAdapter,
    )
    from Medical_KG_rev.adapters.unpaywall import UnpaywallAdapter as NewUnpaywallAdapter

    # Verify that both old and new adapters exist and are different classes
    assert LegacyUnpaywallAdapter is not NewUnpaywallAdapter
    assert LegacyCrossrefAdapter is not NewCrossrefAdapter
    assert LegacyPMCAdapter is not NewPMCAdapter
    assert LegacyCOREAdapter is not NewCOREAdapter
    assert LegacyClinicalTrialsAdapter is not NewClinicalTrialsAdapter
    assert LegacyOpenFDADrugLabelAdapter is not NewOpenFDADrugLabelAdapter
    assert LegacyOpenFDADrugEventAdapter is not NewOpenFDADrugEventAdapter
    assert LegacyOpenFDADeviceAdapter is not NewOpenFDADeviceAdapter
    assert LegacyRxNormAdapter is not NewRxNormAdapter
    assert LegacyICD11Adapter is not NewICD11Adapter
    assert LegacyMeSHAdapter is not NewMeSHAdapter
    assert LegacyChEMBLAdapter is not NewChEMBLAdapter
    assert LegacySemanticScholarAdapter is not NewSemanticScholarAdapter
