"""Utilities for building idempotent Cypher statements."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .schema import GRAPH_SCHEMA, NodeSchema


@dataclass(slots=True, frozen=True)
class CypherTemplates:
    """Pre-built Cypher statements for common graph operations."""

    node_schema: Mapping[str, NodeSchema]

    def merge_node(
        self, label: str, properties: Mapping[str, object]
    ) -> tuple[str, dict[str, object]]:
        schema = self._get_schema(label)
        key_property = schema.key
        if key_property not in properties:
            raise ValueError(f"Missing required key property '{key_property}' for {label}")
        assignments = self._format_assignments(properties)
        query = (
            f"MERGE (n:{label} {{{key_property}: $props.{key_property}}}) "
            f"SET {assignments} RETURN n"
        )
        return query, {"props": dict(properties)}

    def link_nodes(
        self,
        start_label: str,
        end_label: str,
        rel_type: str,
        start_key: object,
        end_key: object,
        properties: Mapping[str, object] | None = None,
    ) -> tuple[str, dict[str, object]]:
        start_schema = self._get_schema(start_label)
        end_schema = self._get_schema(end_label)
        props = properties or {}
        assignments = ""
        if props:
            assignments = " SET r += $props"
        query = (
            f"MATCH (a:{start_label} {{{start_schema.key}: $start}}) "
            f"MATCH (b:{end_label} {{{end_schema.key}: $end}}) "
            f"MERGE (a)-[r:{rel_type}]->(b){assignments} RETURN r"
        )
        parameters = {"start": start_key, "end": end_key}
        if props:
            parameters["props"] = dict(props)
        return query, parameters

    def _get_schema(self, label: str) -> NodeSchema:
        try:
            return self.node_schema[label]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown label: {label}") from exc

    def _format_assignments(self, properties: Mapping[str, object]) -> str:
        assignments: Iterable[str] = (f"n.{key} = $props.{key}" for key in properties)
        return ", ".join(assignments)


DEFAULT_TEMPLATES = CypherTemplates(GRAPH_SCHEMA)
