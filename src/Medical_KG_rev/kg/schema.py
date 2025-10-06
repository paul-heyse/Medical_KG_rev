"""Graph schema definitions for the medical knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping


@dataclass(slots=True, frozen=True)
class NodeSchema:
    label: str
    key: str
    properties: Mapping[str, str] = field(default_factory=dict)

    def required_properties(self) -> Iterable[str]:
        yield self.key
        for name, requirement in self.properties.items():
            if requirement == "required":
                yield name


@dataclass(slots=True, frozen=True)
class RelationshipSchema:
    type: str
    start_label: str
    end_label: str
    properties: Mapping[str, str] = field(default_factory=dict)


GRAPH_SCHEMA: Dict[str, NodeSchema] = {
    "Document": NodeSchema(
        label="Document",
        key="document_id",
        properties={
            "title": "required",
            "source": "optional",
            "ingested_at": "required",
        },
    ),
    "Entity": NodeSchema(
        label="Entity",
        key="entity_id",
        properties={
            "name": "required",
            "type": "required",
            "canonical_identifier": "optional",
        },
    ),
    "Claim": NodeSchema(
        label="Claim",
        key="claim_id",
        properties={
            "statement": "required",
            "polarity": "optional",
        },
    ),
    "Evidence": NodeSchema(
        label="Evidence",
        key="evidence_id",
        properties={
            "chunk_id": "required",
            "confidence": "optional",
        },
    ),
    "ExtractionActivity": NodeSchema(
        label="ExtractionActivity",
        key="activity_id",
        properties={
            "performed_at": "required",
            "pipeline": "required",
        },
    ),
}

RELATIONSHIPS: Dict[str, RelationshipSchema] = {
    "MENTIONS": RelationshipSchema(
        type="MENTIONS",
        start_label="Document",
        end_label="Entity",
        properties={"sentence_index": "optional"},
    ),
    "SUPPORTS": RelationshipSchema(
        type="SUPPORTS",
        start_label="Evidence",
        end_label="Claim",
    ),
    "DERIVED_FROM": RelationshipSchema(
        type="DERIVED_FROM",
        start_label="Evidence",
        end_label="Document",
    ),
    "GENERATED_BY": RelationshipSchema(
        type="GENERATED_BY",
        start_label="Evidence",
        end_label="ExtractionActivity",
        properties={"tool": "optional"},
    ),
    "DESCRIBES": RelationshipSchema(
        type="DESCRIBES",
        start_label="Claim",
        end_label="Entity",
    ),
}
