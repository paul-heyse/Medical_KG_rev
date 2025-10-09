"""Graph schema definitions for the medical knowledge graph.

This module provides schema definitions for nodes and relationships in the
medical knowledge graph, defining the structure and validation rules for
graph entities.

Key Responsibilities:
    - Define node schemas with labels, keys, and property requirements
    - Define relationship schemas with types and property constraints
    - Provide canonical schema definitions for core graph entities
    - Support property requirement validation (required vs optional)

Collaborators:
    - Upstream: Graph validation services, SHACL validator
    - Downstream: Neo4j client, graph construction utilities

Side Effects:
    - None (pure data definitions)

Thread Safety:
    - Thread-safe: All classes are immutable dataclasses

Performance Characteristics:
    - O(1) property lookup for schema validation
    - Minimal memory footprint with frozen dataclasses
"""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

# ============================================================================
# SCHEMA DATA MODELS
# ============================================================================


@dataclass(slots=True, frozen=True)
class NodeSchema:
    """Schema definition for a graph node type.

    Defines the structure and validation rules for a specific node type
    in the medical knowledge graph, including required properties and
    the primary key field.

    Attributes:
        label: The node label used in Cypher queries (e.g., "Document", "Entity")
        key: The primary key property name for this node type
        properties: Mapping of property names to requirement levels
            ("required" or "optional")

    Invariants:
        - label is non-empty and matches Neo4j label conventions
        - key is non-empty and corresponds to a unique identifier
        - properties mapping is immutable after creation

    Example:
        >>> schema = NodeSchema(
        ...     label="Document",
        ...     key="document_id",
        ...     properties={"title": "required", "source": "optional"}
        ... )
        >>> list(schema.required_properties())
        ['document_id', 'title']

    """

    label: str
    key: str
    properties: Mapping[str, str] = field(default_factory=dict)

    def required_properties(self) -> Iterable[str]:
        """Yield all required property names for this node schema.

        Returns:
            Iterable of property names that are marked as "required",
            including the primary key.

        Note:
            The primary key is always required regardless of the
            properties mapping.

        Example:
            >>> schema = NodeSchema(
            ...     label="Entity",
            ...     key="entity_id",
            ...     properties={"name": "required", "type": "optional"}
            ... )
            >>> list(schema.required_properties())
            ['entity_id', 'name']

        """
        yield self.key
        for name, requirement in self.properties.items():
            if requirement == "required":
                yield name


@dataclass(slots=True, frozen=True)
class RelationshipSchema:
    """Schema definition for a graph relationship type.

    Defines the structure and validation rules for a specific relationship
    type in the medical knowledge graph, including start/end node types
    and optional properties.

    Attributes:
        type: The relationship type used in Cypher queries (e.g., "MENTIONS")
        start_label: The label of the start node for this relationship
        end_label: The label of the end node for this relationship
        properties: Mapping of property names to requirement levels
            ("required" or "optional")

    Invariants:
        - type is non-empty and matches Neo4j relationship conventions
        - start_label and end_label reference valid node schemas
        - properties mapping is immutable after creation

    Example:
        >>> schema = RelationshipSchema(
        ...     type="MENTIONS",
        ...     start_label="Document",
        ...     end_label="Entity",
        ...     properties={"sentence_index": "optional"}
        ... )

    """

    type: str
    start_label: str
    end_label: str
    properties: Mapping[str, str] = field(default_factory=dict)


# ============================================================================
# SCHEMA CONSTANTS
# ============================================================================


# Canonical node schema definitions for the medical knowledge graph.
# Defines the structure and validation rules for all node types in the
# medical knowledge graph, including core entities like Document, Entity,
# Claim, Evidence, and ExtractionActivity.
#
# Each schema specifies:
# - Node label used in Cypher queries
# - Primary key property name
# - Required vs optional properties
#
# Example:
#     >>> doc_schema = GRAPH_SCHEMA["Document"]
#     >>> doc_schema.label
#     'Document'
#     >>> list(doc_schema.required_properties())
#     ['document_id', 'title', 'ingested_at', 'tenant_id']
GRAPH_SCHEMA: dict[str, NodeSchema] = {
    "Document": NodeSchema(
        label="Document",
        key="document_id",
        properties={
            "title": "required",
            "source": "optional",
            "ingested_at": "required",
            "tenant_id": "required",
        },
    ),
    "Entity": NodeSchema(
        label="Entity",
        key="entity_id",
        properties={
            "name": "required",
            "type": "required",
            "canonical_identifier": "optional",
            "ontology_code": "required",
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

# Canonical relationship schema definitions for the medical knowledge graph.
# Defines the structure and validation rules for all relationship types
# in the medical knowledge graph, including semantic relationships like
# MENTIONS, SUPPORTS, DERIVED_FROM, GENERATED_BY, and DESCRIBES.
#
# Each schema specifies:
# - Relationship type used in Cypher queries
# - Start and end node labels
# - Optional relationship properties
#
# Example:
#     >>> mentions_rel = RELATIONSHIPS["MENTIONS"]
#     >>> mentions_rel.type
#     'MENTIONS'
#     >>> mentions_rel.start_label
#     'Document'
#     >>> mentions_rel.end_label
#     'Entity'
RELATIONSHIPS: dict[str, RelationshipSchema] = {
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
