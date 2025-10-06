"""SHACL validation for the knowledge graph write path."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from importlib import resources
from typing import Dict, Mapping, MutableMapping, Sequence

from pyshacl import validate
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

from .schema import GRAPH_SCHEMA, RELATIONSHIPS, NodeSchema, RelationshipSchema


class ValidationError(ValueError):
    """Raised when a payload violates SHACL constraints."""


@dataclass(slots=True)
class GraphNodePayload:
    id: str
    label: str
    properties: Mapping[str, object]


@dataclass(slots=True)
class GraphEdgePayload:
    type: str
    start: str
    end: str
    properties: Mapping[str, object] | None = None


@dataclass(slots=True)
class ShaclValidator:
    shapes_graph: Graph
    namespace: Namespace
    relationship_predicates: Mapping[str, str]
    schema: Mapping[str, NodeSchema] = field(default_factory=lambda: GRAPH_SCHEMA)

    @classmethod
    def default(cls) -> "ShaclValidator":
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
    ) -> "ShaclValidator":
        base = cls.default()
        mapping = cls._build_relationship_predicates(relationships or RELATIONSHIPS)
        return cls(
            shapes_graph=base.shapes_graph,
            namespace=base.namespace,
            relationship_predicates=mapping,
            schema=schema,
        )

    def validate_node(self, label: str, properties: Mapping[str, object]) -> None:
        schema = self.schema.get(label)
        if not schema:
            raise ValidationError(f"Unknown label: {label}")
        key_property = schema.key
        missing = [
            prop
            for prop in schema.required_properties()
            if not self._has_value(properties, prop)
        ]
        if missing:
            missing.sort()
            raise ValidationError(
                "Missing required properties: " + ", ".join(missing)
            )
        identifier = str(properties.get(key_property) or properties.get("id") or key_property)
        node = GraphNodePayload(id=identifier, label=label, properties=properties)
        self.validate_payload([node], [])

    def validate_payload(
        self,
        nodes: Sequence[Mapping[str, object] | GraphNodePayload],
        edges: Sequence[Mapping[str, object] | GraphEdgePayload],
    ) -> None:
        data_graph = self._build_graph(nodes, edges)
        conforms, _, results_text = validate(
            data_graph,
            shacl_graph=self.shapes_graph,
            inference="rdfs",
            abort_on_first=False,
        )
        if not conforms:
            raise ValidationError(str(results_text))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_graph(
        self,
        nodes: Sequence[Mapping[str, object] | GraphNodePayload],
        edges: Sequence[Mapping[str, object] | GraphEdgePayload],
    ) -> Graph:
        graph = Graph()
        ns = self.namespace
        node_uris: MutableMapping[str, URIRef] = {}
        evidence_to_activity: Dict[str, URIRef] = {}
        claim_supports: Dict[str, set[str]] = {}
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
        for raw in edges:
            edge = self._coerce_edge(raw)
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
    ) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for rel in relationships.values():
            mapping[rel.type] = ShaclValidator._predicate_name(rel.type)
        return mapping

    @staticmethod
    def _predicate_name(rel_type: str) -> str:
        parts = rel_type.lower().split("_")
        head, *tail = parts
        return head + "".join(segment.title() for segment in tail)


__all__ = [
    "ShaclValidator",
    "ValidationError",
    "GraphNodePayload",
    "GraphEdgePayload",
]
