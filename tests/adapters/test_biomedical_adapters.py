import httpx
import pytest

from Medical_KG_rev.adapters import AdapterDomain, AdapterRequest, create_adapter_from_config, load_adapter_config
from Medical_KG_rev.adapters.biomedical import (
    ChEMBLAdapter,
    ClinicalTrialsAdapter,
    COREAdapter,
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
from Medical_KG_rev.adapters.plugins.domains.biomedical import (
    ChEMBLAdapterPlugin,
    ClinicalTrialsAdapterPlugin,
    COREAdapterPlugin,
    CrossrefAdapterPlugin,
    ICD11AdapterPlugin,
    MeSHAdapterPlugin,
    OpenAlexAdapterPlugin,
    OpenFDADeviceAdapterPlugin,
    OpenFDADrugEventAdapterPlugin,
    OpenFDADrugLabelAdapterPlugin,
    PMCAdapterPlugin,
    RxNormAdapterPlugin,
    SemanticScholarAdapterPlugin,
    UnpaywallAdapterPlugin,
)
from Medical_KG_rev.adapters.plugins.models import AdapterResponse
from Medical_KG_rev.adapters.plugins.base import BaseAdapterPlugin
from Medical_KG_rev.services.pdf import PdfMetadata
from Medical_KG_rev.utils.http_client import BackoffStrategy, HttpClient, RetryConfig


def _client(base_url: str, handler: httpx.MockTransport) -> HttpClient:
    return HttpClient(
        base_url=base_url,
        retry=RetryConfig(attempts=2, backoff_strategy=BackoffStrategy.NONE, jitter=False),
        transport=handler,
    )


def _mock_transport(callback):
    return httpx.MockTransport(callback)


def _run_plugin(
    plugin: BaseAdapterPlugin, *, parameters: dict[str, object], domain: AdapterDomain = AdapterDomain.BIOMEDICAL
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


def test_clinical_trials_adapter_maps_metadata():
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
    assert len(response.items) == 1
    document = response.items[0]
    assert document.id == "NCT01234567"
    assert document.metadata["overall_status"] == "Recruiting"
    assert document.metadata["phase"] == "Phase 2"
    assert "Drug: Drug A" in document.metadata["interventions"]
    assert document.sections[0].blocks[0].text == "Short overview"


def test_clinical_trials_adapter_validates_identifier():
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("API should not be called for invalid identifiers")

    plugin = ClinicalTrialsAdapterPlugin(
        adapter=ClinicalTrialsAdapter(
            client=_client("https://clinicaltrials.gov/api/v2", _mock_transport(handler))
        )
    )
    with pytest.raises(ValueError):
        _run_plugin(plugin, parameters={"nct_id": "bad"})


def test_openfda_drug_label_adapter_parses_spl():
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


def test_openfda_drug_event_adapter_maps_reactions():
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


def test_openfda_device_adapter_handles_definition():
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


def test_openalex_adapter_flattens_abstract():
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W123",
                "display_name": "Lung cancer immunotherapy",
                "doi": "10.1000/example",
                "publication_year": 2023,
                "authorships": [{"author": {"display_name": "Dr. Doe"}}],
                "concepts": [{"display_name": "Immunotherapy"}],
                "abstract_inverted_index": {"Immune": [0], "therapy": [1]},
                "open_access": {"is_oa": True},
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/works"
        return httpx.Response(200, json=payload)

    plugin = OpenAlexAdapterPlugin(
        adapter=OpenAlexAdapter(
            client=_client("https://api.openalex.org", _mock_transport(handler))
        )
    )
    response = _run_plugin(plugin, parameters={"query": "lung cancer"})
    document = response.items[0]
    assert document.metadata["doi"] == "10.1000/example"
    assert document.sections[0].blocks[0].text == "Immune therapy"


def test_openalex_adapter_extracts_pdf_metadata():
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W999",
                "display_name": "Structured PDF", 
                "abstract": "Summary",
                "best_oa_location": {"url_for_pdf": "https://example.org/paper.pdf"},
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/works"
        return httpx.Response(200, json=payload)

    class _Validator:
        def validate(self, url: str) -> PdfMetadata:
            return PdfMetadata(
                url=url,
                content_type="application/pdf",
                size=2048,
                last_modified=None,
                accessible=True,
                headers={"Accept-Ranges": "bytes"},
            )

    plugin = OpenAlexAdapterPlugin(
        adapter=OpenAlexAdapter(
            client=_client("https://api.openalex.org", _mock_transport(handler)),
            pdf_validator=_Validator(),
        )
    )
    response = _run_plugin(plugin, parameters={"query": "structured"})
    document = response.items[0]
    assert str(document.pdf_url) == "https://example.org/paper.pdf"
    assert document.pdf_size == 2048
    assert document.pdf_metadata["headers"]["Accept-Ranges"] == "bytes"


def test_unpaywall_adapter_returns_metadata():
    payload = {
        "doi": "10.1000/example",
        "title": "Sample",
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


def test_crossref_adapter_extracts_references():
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


def test_core_adapter_handles_multiple_entries():
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


def test_pmc_adapter_parses_xml():
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
    assert any(block.text == "Body paragraph one." for block in document.sections[0].blocks)


def test_rxnorm_adapter_normalizes_name():
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


def test_icd11_adapter_searches_term():
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


def test_mesh_adapter_descriptor_lookup():
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


def test_chembl_adapter_fetches_by_identifier():
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


def test_semantic_scholar_adapter_enriches_citations():
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


def test_yaml_configured_adapter(tmp_path):
    config_content = """
name: example-yaml
source: example
base_url: https://clinicaltrials.gov/api/v2
request:
  method: GET
  path: /studies/{nct_id}
response:
  items_path: study
mapping:
  id: protocolSection.identificationModule.nctId
  title: protocolSection.identificationModule.briefTitle
  summary: protocolSection.descriptionModule.briefSummary
"""
    config_path = tmp_path / "clinicaltrials.yaml"
    config_path.write_text(config_content)
    config = load_adapter_config(config_path)

    payload = {
        "study": {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000001", "briefTitle": "Example"},
                "descriptionModule": {"briefSummary": "Summary"},
            }
        }
    }

    adapter = create_adapter_from_config(
        config,
        client=_client(
            "https://clinicaltrials.gov/api/v2",
            _mock_transport(lambda request: httpx.Response(200, json=payload)),
        ),
    )
    plugin = ClinicalTrialsAdapterPlugin(adapter=adapter)
    response = _run_plugin(plugin, parameters={"nct_id": "NCT00000001"})
    document = response.items[0]
    assert document.title == "Example"
    assert document.sections[0].blocks[0].text == "Summary"
