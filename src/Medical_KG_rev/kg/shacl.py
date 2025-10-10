"""SHACL validation for the knowledge graph write path.

This module provides SHACL (Shapes Constraint Language) validation for
knowledge graph operations, ensuring data integrity and schema compliance
before write operations.

Key Responsibilities:
    - Validate node properties against schema requirements
    - Validate graph structure using SHACL shapes
    - Convert graph data to RDF format for validation
    - Provide detailed validation error messages

Collaborators:
    - Upstream: Neo4jClient, graph construction services
    - Downstream: pyshacl library, RDF graph construction

Side Effects:
    - Reads SHACL shape definitions from shapes.ttl
    - Constructs RDF graphs for validation
    - Raises ValidationError for constraint violations

Thread Safety:
    - Thread-safe: All methods are stateless
    - SHACL validation is read-only

Performance Characteristics:
    - O(n) validation time for n nodes/edges
    - RDF graph construction overhead
    - SHACL validation may be expensive for large graphs
"""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from importlib import resources

from pyshacl import validate
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

from .schema import GRAPH_SCHEMA, RELATIONSHIPS, NodeSchema, RelationshipSchema

# ============================================================================
# EXCEPTION CLASSES
# ============================================================================


class ValidationError(ValueError):
    """Raised when a payload violates SHACL constraints.

    This exception is raised when graph data fails validation against
    the defined SHACL shapes, indicating schema compliance issues.

    Attributes:
        msg: Detailed error message describing the validation failure

    Example:
        >>> raise ValidationError("Missing required properties: document_id, title")

    """


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(slots=True)
class GraphNodePayload:
    """Payload structure for graph node validation.

    Represents a node in the knowledge graph with its identifier,
    label, and properties for SHACL validation.

    Attributes:
        id: Unique identifier for the node
        label: Node label (e.g., "Document", "Entity")
        properties: Mapping of property names to values

    Example:
        >>> node = GraphNodePayload(
        ...     id="doc1",
        ...     label="Document",
        ...     properties={"title": "Test Document", "source": "pubmed"}
        ... )

    """

    id: str
    label: str
    properties: Mapping[str, object]


@dataclass(slots=True)
class GraphEdgePayload:
    """Payload structure for graph edge validation.

    Represents a relationship in the knowledge graph with its type,
    start/end nodes, and optional properties for SHACL validation.

    Attributes:
        type: Relationship type (e.g., "MENTIONS", "SUPPORTS")
        start: Identifier of the start node
        end: Identifier of the end node
        properties: Optional mapping of relationship properties

    Example:
        >>> edge = GraphEdgePayload(
        ...     type="MENTIONS",
        ...     start="doc1",
        ...     end="entity1",
        ...     properties={"sentence_index": 5}
        ... )

    """

    type: str
    start: str
    end: str
    properties: Mapping[str, object] | None = None


# ============================================================================
# SHACL VALIDATOR IMPLEMENTATION
# ============================================================================


@dataclass(slots=True)
class ShaclValidator:
    """SHACL validator for knowledge graph operations.

    Validates graph data against SHACL shapes to ensure schema compliance
    and data integrity before write operations.

    Attributes:
        shapes_graph: RDF graph containing SHACL shape definitions
        namespace: RDF namespace for schema entities
        relationship_predicates: Mapping of relationship types to predicate names
        schema: Node schema definitions for validation

    Invariants:
        - shapes_graph contains valid SHACL shape definitions
        - schema contains all required node types
        - relationship_predicates covers all relationship types

    Thread Safety:
        - Thread-safe: All methods are stateless

    Example:
        >>> validator = ShaclValidator.default()
        >>> validator.validate_node("Document", {"document_id": "doc1", "title": "Test"})

    """

    shapes_graph: Graph
    namespace: Namespace
    relationship_predicates: Mapping[str, str]
    schema: Mapping[str, NodeSchema] = field(default_factory=lambda: GRAPH_SCHEMA)

    @classmethod
    def default(cls) -> ShaclValidator:
        """Create a default ShaclValidator instance.

        Loads SHACL shapes from the shapes.ttl file and creates a validator
        instance with the canonical graph schema and relationship definitions.

        Returns:
            ShaclValidator instance configured with default settings

        Raises:
            FileNotFoundError: If shapes.ttl file is not found
            ValueError: If shapes.ttl contains invalid RDF

        Note:
            This method loads shapes from the package resources, ensuring
            the validator uses the latest shape definitions.

        Example:
            >>> validator = ShaclValidator.default()
            >>> validator.validate_node("Document", {"document_id": "doc1"})

        """
        shape_text = resources.files("Medical_KG_rev.kg").joinpath("shapes.ttl").read_text()
        shapes_graph = Graph().parse(data=shape_text, format="turtle")
        namespace = Namespace("http://medical-kg/schema#")
        relationships = cls._build_relationship_predicates(RELATIONSHIPS)
        return cls(
            shapes_graph=shapes_graph,
            namespace=namespace,
            relationship_predicates=relationships,
            schema=GRAPH_SCHEMA,
        )

    @classmethod
    def from_schema(
        cls,
        schema: Mapping[str, NodeSchema],
        relationships: Mapping[str, RelationshipSchema] | None = None,
    ) -> ShaclValidator:
        """Create a ShaclValidator instance with custom schema.

        Creates a validator instance using custom node and relationship
        schemas while keeping the default SHACL shapes.

        Args:
            schema: Custom node schema definitions
            relationships: Custom relationship schema definitions

        Returns:
            ShaclValidator instance configured with custom schemas

        Note:
            The SHACL shapes remain the same, but validation uses the
            provided schema definitions for property requirements.

        Example:
            >>> custom_schema = {"CustomNode": NodeSchema(...)}
            >>> validator = ShaclValidator.from_schema(custom_schema)

        """
        base = cls.default()
        mapping = cls._build_relationship_predicates(relationships or RELATIONSHIPS)
        return cls(
            shapes_graph=base.shapes_graph,
            namespace=base.namespace,
            relationship_predicates=mapping,
            schema=schema,
        )

    def validate_node(self, label: str, properties: Mapping[str, object]) -> None:
        """Validate a single node against schema requirements and SHACL shapes.

        Checks that the node has all required properties according to its
        schema definition and validates the node against SHACL constraints.

        Args:
            label: Node label (e.g., "Document", "Entity")
            properties: Node properties to validate

        Raises:
            ValidationError: If the node fails schema or SHACL validation

        Note:
            This method validates both schema compliance (required properties)
            and SHACL constraints (data types, value ranges, etc.).

        Example:
            >>> validator.validate_node("Document", {
            ...     "document_id": "doc1",
            ...     "title": "Test Document",
            ...     "tenant_id": "tenant1"
            ... })

        """
        schema = self.schema.get(label)
        if not schema:
            raise ValidationError(f"Unknown label: {label}")
        key_property = schema.key
        missing = [
            prop for prop in schema.required_properties() if not self._has_value(properties, prop)
        ]
        if missing:
            missing.sort()
            raise ValidationError("Missing required properties: " + ", ".join(missing))
        identifier = str(properties.get(key_property) or properties.get("id") or key_property)
        node = GraphNodePayload(id=identifier, label=label, properties=properties)
        self.validate_payload([node], [])

    def validate_payload(
        self,
        nodes: Sequence[Mapping[str, object] | GraphNodePayload],
        edges: Sequence[Mapping[str, object] | GraphEdgePayload],
    ) -> None:
        """Validate a complete graph payload against SHACL shapes.

        Converts the graph data to RDF format and validates it against
        the loaded SHACL shapes, ensuring overall graph structure compliance.

        Args:
            nodes: Sequence of node payloads to validate
            edges: Sequence of edge payloads to validate

        Raises:
            ValidationError: If the graph fails SHACL validation

        Note:
            This method performs comprehensive validation including
            relationship constraints, cardinality rules, and data types.

        Example:
            >>> nodes = [{"id": "doc1", "label": "Document", "properties": {...}}]
            >>> edges = [{"type": "MENTIONS", "start": "doc1", "end": "entity1"}]
            >>> validator.validate_payload(nodes, edges)

        """
        data_graph = self._build_graph(nodes, edges)
        conforms, _, results_text = validate(
            data_graph,
            shacl_graph=self.shapes_graph,
            inference="rdfs",
            abort_on_first=False,
        )
        if not conforms:
            raise ValidationError(str(results_text))

    # ============================================================================
    # PRIVATE HELPERS
    # ============================================================================

    def _build_graph(
        self,
        nodes: Sequence[Mapping[str, object] | GraphNodePayload],
        edges: Sequence[Mapping[str, object] | GraphEdgePayload],
    ) -> Graph:
        """Build an RDF graph from node and edge payloads.

        Converts graph data to RDF format for SHACL validation, including
        node properties, relationships, and reified relationship properties.

        Args:
            nodes: Sequence of node payloads to convert
            edges: Sequence of edge payloads to convert

        Returns:
            RDF graph containing the converted data

        Note:
            This method handles complex relationship patterns including
            evidence-to-activity and claim-support relationships.

        """
        graph = Graph()
        ns = self.namespace
        node_uris: MutableMapping[str, URIRef] = {}
        evidence_to_activity: dict[str, URIRef] = {}
        claim_supports: dict[str, set[str]] = {}
        for raw in nodes:
            node = self._coerce_node(raw)
            uri = URIRef(f"{ns}{node.label}/{node.id}")
            node_uris[node.id] = uri
            graph.add((uri, RDF.type, ns[node.label]))
            for key, value in node.properties.items():
                if value is None:
                    continue
                predicate = ns[key]
                literal = self._to_literal(value)
                graph.add((uri, predicate, literal))
        for raw in edges:  # type: ignore[assignment]
            # Type annotation to help mypy understand this is an edge payload
            raw_edge: Mapping[str, object] | GraphEdgePayload = raw  # type: ignore[assignment]
            edge = self._coerce_edge(raw_edge)
            predicate_name = self.relationship_predicates.get(edge.type)
            if not predicate_name:
                raise ValidationError(f"Unsupported relationship type: {edge.type}")
            try:
                start_uri = node_uris[edge.start]
                end_uri = node_uris[edge.end]
            except KeyError as exc:
                raise ValidationError("Edge references unknown node") from exc
            graph.add((start_uri, ns[predicate_name], end_uri))
            if edge.type == "GENERATED_BY":
                evidence_to_activity[edge.start] = end_uri
            elif edge.type == "SUPPORTS":
                claim_supports.setdefault(edge.end, set()).add(edge.start)
            if edge.properties:
                for prop, value in edge.properties.items():
                    if value is None:
                        continue
                    reified = URIRef(f"{start_uri}_to_{end_uri}_{prop}")
                    graph.add((reified, ns["propertyOf"], start_uri))
                    graph.add((reified, ns[prop], self._to_literal(value)))
        generated_by_predicate = ns["generatedBy"]
        for claim_id, evidence_ids in claim_supports.items():
            claim_uri = node_uris.get(claim_id)
            if not claim_uri:
                continue
            for evidence_id in evidence_ids:
                activity_uri = evidence_to_activity.get(evidence_id)
                if activity_uri is not None:
                    graph.add((claim_uri, generated_by_predicate, activity_uri))
        return graph

    def _coerce_node(self, payload: Mapping[str, object] | GraphNodePayload) -> GraphNodePayload:
        """Convert a raw payload to GraphNodePayload.

        Args:
            payload: Raw node data or GraphNodePayload instance

        Returns:
            GraphNodePayload instance

        Raises:
            ValidationError: If required fields are missing or invalid

        """
        if isinstance(payload, GraphNodePayload):
            return payload
        identifier = str(payload.get("id") or payload.get("key") or payload.get("label"))
        label = str(payload.get("label"))
        properties = payload.get("properties")
        if not label:
            raise ValidationError("Node label is required")
        if not isinstance(properties, Mapping):
            raise ValidationError("Node properties must be an object")
        return GraphNodePayload(id=identifier, label=label, properties=properties)

    def _coerce_edge(self, payload: Mapping[str, object] | GraphEdgePayload) -> GraphEdgePayload:
        """Convert a raw payload to GraphEdgePayload.

        Args:
            payload: Raw edge data or GraphEdgePayload instance

        Returns:
            GraphEdgePayload instance

        Raises:
            ValidationError: If required fields are missing or invalid

        """
        if isinstance(payload, GraphEdgePayload):
            return payload
        edge_type = str(payload.get("type"))
        start = str(payload.get("start"))
        end = str(payload.get("end"))
        properties = payload.get("properties")
        if not edge_type or not start or not end:
            raise ValidationError("Edges require type, start, and end identifiers")
        if properties is not None and not isinstance(properties, Mapping):
            raise ValidationError("Edge properties must be an object when provided")
        return GraphEdgePayload(type=edge_type, start=start, end=end, properties=properties)

    def _to_literal(self, value: object) -> Literal:
        """Convert a Python value to an RDF Literal.

        Args:
            value: Python value to convert

        Returns:
            RDF Literal with appropriate datatype

        Note:
            Automatically detects datetime strings and converts them
            to XSD.dateTime literals.

        """
        if isinstance(value, bool):
            return Literal(value)
        if isinstance(value, int):
            return Literal(value)
        if isinstance(value, float):
            return Literal(Decimal(str(value)), datatype=XSD.decimal)
        if isinstance(value, str):
            if self._looks_like_datetime(value):
                return Literal(value, datatype=XSD.dateTime)
            return Literal(value)
        if isinstance(value, Sequence):
            return Literal(", ".join(str(item) for item in value))
        return Literal(str(value))

    @staticmethod
    def _has_value(properties: Mapping[str, object], key: str) -> bool:
        """Check if a property has a meaningful value.

        Args:
            properties: Property mapping to check
            key: Property key to check

        Returns:
            True if the property has a non-empty value

        """
        if key not in properties:
            return False
        value = properties[key]
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip() != ""
        return True

    @staticmethod
    def _looks_like_datetime(value: str) -> bool:
        """Check if a string looks like a datetime value.

        Args:
            value: String to check

        Returns:
            True if the string appears to be a datetime

        """
        candidate = value.strip()
        if not candidate:
            return False
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            datetime.fromisoformat(candidate)
        except ValueError:
            return False
        return True

    @staticmethod
    def _build_relationship_predicates(
        relationships: Mapping[str, RelationshipSchema],
    ) -> dict[str, str]:
        """Build mapping from relationship types to predicate names.

        Args:
            relationships: Relationship schema definitions

        Returns:
            Mapping of relationship types to predicate names

        """
        mapping: dict[str, str] = {}
        for rel in relationships.values():
            mapping[rel.type] = ShaclValidator._predicate_name(rel.type)
        return mapping

    @staticmethod
    def _predicate_name(rel_type: str) -> str:
        """Convert relationship type to predicate name.

        Args:
            rel_type: Relationship type (e.g., "GENERATED_BY")

        Returns:
            Predicate name (e.g., "generatedBy")

        """
        parts = rel_type.lower().split("_")
        head, *tail = parts
        return head + "".join(segment.title() for segment in tail)


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "GraphEdgePayload",
    "GraphNodePayload",
    "ShaclValidator",
    "ValidationError",
]
