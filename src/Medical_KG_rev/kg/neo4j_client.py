"""Thin wrapper around the Neo4j Python driver.

This module provides a high-level interface to Neo4j operations, including
convenience methods for common graph operations like node creation, relationship
linking, and transaction management.

Key Responsibilities:
    - Provide session management with automatic cleanup
    - Validate graph operations using SHACL constraints
    - Generate Cypher queries using templates
    - Execute write operations with proper transaction handling

Collaborators:
    - Upstream: Graph construction services, ingestion pipelines
    - Downstream: Neo4j driver, CypherTemplates, ShaclValidator

Side Effects:
    - Creates Neo4j sessions and transactions
    - Executes write operations on the graph database
    - Validates graph structure using SHACL

Thread Safety:
    - Not thread-safe: Each instance should be used by a single thread
    - Session management is per-operation

Performance Characteristics:
    - Connection pooling handled by Neo4j driver
    - Automatic session cleanup prevents resource leaks
    - Transaction batching supported via with_transaction
"""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .cypher_templates import CypherTemplates
from .schema import GRAPH_SCHEMA
from .shacl import ShaclValidator

# ============================================================================
# NEO4J CLIENT IMPLEMENTATION
# ============================================================================


@dataclass(slots=True)
class Neo4jClient:
    """Provides convenience helpers for common graph operations.

    A high-level wrapper around the Neo4j Python driver that provides
    convenient methods for common graph operations while ensuring proper
    session management and validation.

    Attributes:
        driver: Neo4j driver instance for database connections
        templates: CypherTemplates instance for query generation
        validator: ShaclValidator instance for graph structure validation

    Invariants:
        - driver is a valid Neo4j driver instance
        - templates and validator are properly initialized
        - All operations use proper session management

    Thread Safety:
        - Not thread-safe: Each instance should be used by a single thread
        - Session management is per-operation to avoid conflicts

    Example:
        >>> client = Neo4jClient(driver=neo4j_driver)
        >>> result = client.merge_node("Document", {"document_id": "doc1", "title": "Test"})
        >>> client.link("Document", "Entity", "MENTIONS", "doc1", "entity1")
    """

    driver: Any
    templates: CypherTemplates = field(default_factory=lambda: CypherTemplates(GRAPH_SCHEMA))
    validator: ShaclValidator = field(default_factory=ShaclValidator.default)

    @contextmanager
    def _session(self) -> Iterator[Any]:
        """Create and manage a Neo4j session with automatic cleanup.

        Creates a new Neo4j session and ensures it is properly closed
        when the context manager exits, preventing resource leaks.

        Yields:
            Neo4j session instance for database operations

        Note:
            This is a private method used internally for session management.
            External code should use the public methods that handle sessions
            automatically.

        Example:
            >>> with client._session() as session:
            ...     result = session.run("MATCH (n) RETURN count(n)")
        """
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()

    def write(
        self, query: str, parameters: Mapping[str, object] | None = None
    ) -> Iterable[Mapping[str, Any]]:
        """Execute a write operation with automatic session management.

        Executes a Cypher write query using a managed session and returns
        the result data. The session is automatically created and cleaned up.

        Args:
            query: Cypher query string to execute
            parameters: Optional mapping of query parameters

        Returns:
            Iterable of result records as dictionaries

        Raises:
            Neo4jError: If the query execution fails

        Note:
            This method is suitable for write operations (CREATE, MERGE, DELETE).
            For read operations, consider using the driver's read methods directly.

        Example:
            >>> result = client.write(
            ...     "CREATE (n:Document {document_id: $id})",
            ...     {"id": "doc1"}
            ... )
        """
        with self._session() as session:
            return session.execute_write(lambda tx: tx.run(query, parameters or {}).data())

    def merge_node(self, label: str, properties: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
        """Create or update a node with validation and template generation.

        Validates the node properties against the schema, generates a MERGE
        query using templates, and executes it with proper session management.

        Args:
            label: Node label (e.g., "Document", "Entity")
            properties: Node properties including the primary key

        Returns:
            Iterable of result records from the MERGE operation

        Raises:
            ValidationError: If node properties don't match schema requirements
            Neo4jError: If the database operation fails

        Note:
            The primary key property must be present in properties.
            Validation ensures the node conforms to the defined schema.

        Example:
            >>> result = client.merge_node(
            ...     "Document",
            ...     {"document_id": "doc1", "title": "Test Document"}
            ... )
        """
        self.validator.validate_node(label, properties)
        query, parameters = self.templates.merge_node(label, properties)
        return self.write(query, parameters)

    def link(
        self,
        start_label: str,
        end_label: str,
        rel_type: str,
        start_key: Any,
        end_key: Any,
        properties: Mapping[str, Any] | None = None,
    ) -> Iterable[Mapping[str, Any]]:
        """Create a relationship between two nodes.

        Generates and executes a MERGE query to create a relationship
        between two nodes using their primary keys.

        Args:
            start_label: Label of the start node
            end_label: Label of the end node
            rel_type: Relationship type (e.g., "MENTIONS", "SUPPORTS")
            start_key: Primary key value of the start node
            end_key: Primary key value of the end node
            properties: Optional relationship properties

        Returns:
            Iterable of result records from the MERGE operation

        Raises:
            Neo4jError: If the database operation fails

        Note:
            Both nodes must exist in the database before creating the relationship.
            The relationship will be created if it doesn't exist, or updated if it does.

        Example:
            >>> result = client.link(
            ...     "Document", "Entity", "MENTIONS",
            ...     "doc1", "entity1",
            ...     {"sentence_index": 5}
            ... )
        """
        query, parameters = self.templates.link_nodes(
            start_label,
            end_label,
            rel_type,
            start_key,
            end_key,
            properties,
        )
        return self.write(query, parameters)

    def with_transaction(self, func: Callable[[Any], Any]) -> Any:
        """Execute a function within a Neo4j transaction.

        Provides a managed transaction context for executing multiple
        operations atomically with automatic rollback on failure.

        Args:
            func: Function to execute within the transaction.
                Receives a transaction object as its only argument.

        Returns:
            Return value of the executed function

        Raises:
            Neo4jError: If the transaction fails and is rolled back

        Note:
            The transaction is automatically committed if the function
            completes successfully, or rolled back if an exception occurs.

        Example:
            >>> def create_document_and_entity(tx):
            ...     tx.run("CREATE (d:Document {id: 'doc1'})")
            ...     tx.run("CREATE (e:Entity {id: 'entity1'})")
            ...     return "success"
            >>> result = client.with_transaction(create_document_and_entity)
        """
        with self._session() as session:
            return session.execute_write(func)
