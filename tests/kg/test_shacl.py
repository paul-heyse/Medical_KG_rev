import pytest

from Medical_KG_rev.kg.shacl import ShaclValidator, ValidationError


def test_shacl_validator_accepts_valid_graph():
    validator = ShaclValidator.default()
    nodes = [
        {
            "id": "doc-1",
            "label": "Document",
            "properties": {
                "document_id": "doc-1",
                "title": "Sample",
                "source": "synthetic",
                "ingested_at": "2024-01-01T00:00:00Z",
                "tenant_id": "tenant",
            },
        },
        {
            "id": "claim-1",
            "label": "Claim",
            "properties": {
                "claim_id": "claim-1",
                "statement": "Sample claim",
            },
        },
        {
            "id": "activity-1",
            "label": "ExtractionActivity",
            "properties": {
                "activity_id": "activity-1",
                "performed_at": "2024-01-01T00:00:00Z",
                "pipeline": "test",
            },
        },
        {
            "id": "entity-1",
            "label": "Entity",
            "properties": {
                "entity_id": "entity-1",
                "name": "Hypertension",
                "type": "Condition",
                "ontology_code": "SNOMED:59621000",
            },
        },
        {
            "id": "evidence-1",
            "label": "Evidence",
            "properties": {
                "evidence_id": "e1",
                "chunk_id": "chunk-1",
                "confidence": 0.9,
            },
        },
    ]
    edges = [
        {"type": "GENERATED_BY", "start": "evidence-1", "end": "activity-1"},
        {"type": "DERIVED_FROM", "start": "evidence-1", "end": "doc-1"},
        {"type": "SUPPORTS", "start": "evidence-1", "end": "claim-1"},
        {"type": "DESCRIBES", "start": "claim-1", "end": "entity-1"},
    ]
    validator.validate_payload(nodes, edges)


def test_shacl_validator_rejects_missing_property():
    validator = ShaclValidator.default()
    nodes = [
        {
            "id": "entity-1",
            "label": "Entity",
            "properties": {"entity_id": "entity-1", "name": "Example", "type": "Condition"},
        }
    ]
    with pytest.raises(ValidationError):
        validator.validate_payload(nodes, [])
