"""Utilities for building idempotent Cypher statements.

This module provides template-based Cypher query generation for common
graph operations, ensuring idempotent operations and proper parameterization.

The module supports:
- MERGE queries for node creation/updates
- Relationship creation queries with proper matching
- Node label validation against schema definitions
- Parameterized queries to prevent injection attacks

Thread Safety:
    Thread-safe: All methods are stateless and immutable.

Performance:
    O(1) schema lookup for label validation.
    Template-based generation avoids string concatenation overhead.
    Parameterized queries enable query plan caching.

Example:
-------
    >>> templates = CypherTemplates(GRAPH_SCHEMA)
    >>> query, params = templates.merge_node("Document", {"document_id": "doc1"})
    >>> print(query)
    MERGE (n:Document {document_id: $props.document_id}) SET n.document_id = $props.document_id RETURN n

"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .schema import GRAPH_SCHEMA, NodeSchema


# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================


# ==============================================================================
# SCHEMA DATA MODELS
# ==============================================================================


# ==============================================================================
# CLIENT IMPLEMENTATION
# ==============================================================================


@dataclass(slots=True, frozen=True)
class CypherTemplates:
    """Pre-built Cypher statements for common graph operations.

    Generates idempotent Cypher queries for common graph operations like
    node creation and relationship linking, ensuring proper parameterization
    and schema validation.

    Attributes:
    ----------
        node_schema: Mapping of node labels to their schema definitions

    Invariants:
        - node_schema contains all required node types
        - All generated queries are parameterized
        - MERGE operations ensure idempotency

    Thread Safety:
        - Thread-safe: Immutable dataclass with stateless methods

    Example:
    -------
        >>> templates = CypherTemplates(GRAPH_SCHEMA)
        >>> query, params = templates.merge_node("Document", {"document_id": "doc1"})
        >>> print(query)
        MERGE (n:Document {document_id: $props.document_id}) SET n.document_id = $props.document_id RETURN n

    """

    node_schema: Mapping[str, NodeSchema]

    def merge_node(
        self, label: str, properties: Mapping[str, object]
    ) -> tuple[str, dict[str, object]]:
        """Generate a MERGE query for creating or updating a node.

        Creates an idempotent MERGE query that will create a node if it doesn't
        exist or update it if it does, based on the primary key property.

        Args:
        ----
            label: Node label (e.g., "Document", "Entity")
            properties: Node properties including the primary key

        Returns:
        -------
            Tuple of (query_string, parameters_dict)

        Raises:
        ------
            ValueError: If the primary key property is missing from properties
            ValueError: If the label is not defined in the schema

        Note:
        ----
            The primary key property must be present in properties.
            MERGE ensures idempotency by matching on the primary key.

        Example:
        -------
            >>> query, params = templates.merge_node(
            ...     "Document",
            ...     {"document_id": "doc1", "title": "Test"}
            ... )
            >>> print(query)
            MERGE (n:Document {document_id: $props.document_id}) SET n.document_id = $props.document_id, n.title = $props.title RETURN n

        """
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
        """Generate a MERGE query for creating a relationship between nodes.

        Creates an idempotent MERGE query that will create a relationship
        between two nodes if it doesn't exist, or update it if it does.

        Args:
        ----
            start_label: Label of the start node
            end_label: Label of the end node
            rel_type: Relationship type (e.g., "MENTIONS", "SUPPORTS")
            start_key: Primary key value of the start node
            end_key: Primary key value of the end node
            properties: Optional relationship properties

        Returns:
        -------
            Tuple of (query_string, parameters_dict)

        Raises:
        ------
            ValueError: If either label is not defined in the schema

        Note:
        ----
            Both nodes must exist in the database before creating the relationship.
            MERGE ensures idempotency by matching on both nodes and relationship type.

        Example:
        -------
            >>> query, params = templates.link_nodes(
            ...     "Document", "Entity", "MENTIONS",
            ...     "doc1", "entity1", {"sentence_index": 5}
            ... )
            >>> print(query)
            MATCH (a:Document {document_id: $start}) MATCH (b:Entity {entity_id: $end}) MERGE (a)-[r:MENTIONS]->(b) SET r += $props RETURN r

        """
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
        """Get the schema definition for a node label.

        Args:
        ----
            label: Node label to look up

        Returns:
        -------
            NodeSchema instance for the label

        Raises:
        ------
            ValueError: If the label is not defined in the schema

        Note:
        ----
            This is a private method used internally for schema validation.

        """
        try:
            return self.node_schema[label]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown label: {label}") from exc

    def _format_assignments(self, properties: Mapping[str, object]) -> str:
        """Format property assignments for SET clauses.

        Args:
        ----
            properties: Mapping of property names to values

        Returns:
        -------
            Comma-separated string of assignment expressions

        Note:
        ----
            This is a private method used internally for query generation.
            Generates assignments in the form "n.key = $props.key".

        """
        assignments: Iterable[str] = (f"n.{key} = $props.{key}" for key in properties)
        return ", ".join(assignments)


# ==============================================================================
# TEMPLATES
# ==============================================================================

# Default CypherTemplates instance using the canonical graph schema.
# Provides pre-configured templates for all standard node types and
# operations without requiring manual schema configuration.
#
# Example:
#     >>> query, params = DEFAULT_TEMPLATES.merge_node("Document", {"document_id": "doc1"})
DEFAULT_TEMPLATES = CypherTemplates(GRAPH_SCHEMA)


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = ["DEFAULT_TEMPLATES", "CypherTemplates"]
