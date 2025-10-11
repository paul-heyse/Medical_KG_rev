"""Vector store orchestration service for managing vector operations.

Key Responsibilities:
    - Provide a unified interface for vector store operations
    - Validate security scopes and dimensions before delegating to stores
    - Manage namespace registration and dimension validation
    - Handle vector upsert, query, and deletion operations

Collaborators:
    - Upstream: Gateway services and coordinators request vector operations
    - Downstream: Vector store implementations (Qdrant, FAISS, etc.)

Side Effects:
    - Updates vector store backends with new vectors and metadata
    - Validates and registers namespaces and dimensions
    - Emits metrics for vector operations

Thread Safety:
    - Thread-safe: All operations can be called concurrently
    - Store implementations must handle concurrent access

Performance Characteristics:
    - O(1) namespace lookups for registered namespaces
    - O(n) dimension validation where n is vector count
    - Vector operations delegate to store implementation performance

Example:
    >>> service = VectorStoreService(store=my_store, registry=my_registry)
    >>> result = service.upsert(context=ctx, namespace="docs", records=records)
    >>> print(f"Upserted {result.upserted} vectors")
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from Medical_KG_rev.auth.context import SecurityContext

from .errors import DimensionMismatchError, NamespaceNotFoundError, ScopeError
from .models import HttpClient
from .registry import NamespaceRegistry
from .types import VectorStorePort


class VectorStoreService:
    """Vector store orchestration service with validation and security.

    This service provides a validated interface to vector store operations,
    ensuring proper security scopes, dimension validation, and namespace
    management before delegating to the underlying store implementation.

    Attributes:
        store: The underlying vector store implementation.
        registry: Namespace and dimension registry for validation.

    Invariants:
        - store must be a valid VectorStorePort implementation
        - registry must be a valid NamespaceRegistry instance
        - All operations require proper security context

    Thread Safety:
        - Thread-safe: All operations can be called concurrently

    Lifecycle:
        - Created with store and registry dependencies
        - No explicit cleanup required
        - Can be reused across multiple requests

    Example:
        >>> service = VectorStoreService(store=qdrant_store, registry=namespace_registry)
        >>> context = SecurityContext(tenant_id="tenant1", scopes=["index:write"])
        >>> result = service.upsert(context=context, namespace="docs", records=records)
    """

    def __init__(self, store: VectorStorePort, registry: NamespaceRegistry) -> None:
        """Initialize the vector store service.

        Args:
            store: Vector store implementation to delegate operations to.
            registry: Namespace registry for validation and management.
        """
        self.store = store
        self.registry = registry

    # ------------------------------------------------------------------
    # Namespace management
    # ------------------------------------------------------------------
    def ensure_namespace(
        self,
        *,
        context: SecurityContext,
        config: NamespaceConfig,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Ensure a namespace exists and is properly configured.

        Args:
            context: Security context with tenant and scope information.
            config: Namespace configuration including name, dimensions, etc.
            metadata: Optional metadata to associate with the namespace.

        Raises:
            ScopeError: If context lacks required "index:write" scope.
            NamespaceNotFoundError: If namespace creation fails.
            RuntimeError: If store operation fails.
        """
        self._require_scope(context, "index:write")
        self.registry.register(tenant_id=context.tenant_id, config=config)
        self.store.create_or_update_collection(  # type: ignore[attr-defined]
            tenant_id=context.tenant_id,
            namespace=config.name,
            params=config.params,
            compression=config.compression,
            metadata=dict(metadata or {}),
            named_vectors=config.named_vectors,
        )

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------
    def upsert(
        self,
        *,
        context: SecurityContext,
        namespace: str,
        records: Sequence[VectorRecord],
    ) -> UpsertResult:
        """Upsert vectors into the specified namespace.

        Args:
            context: Security context with tenant and scope information.
            namespace: Target namespace for vector insertion.
            records: Vector records to upsert.

        Returns:
            Result containing upsert count and namespace version.

        Raises:
            ScopeError: If context lacks required "index:write" scope.
            DimensionMismatchError: If vector dimensions don't match namespace.
            RuntimeError: If store operation fails.
        """
        self._require_scope(context, "index:write")
        if not records:
            return UpsertResult(namespace=namespace, upserted=0, version="")
        config = self.registry.get(tenant_id=context.tenant_id, namespace=namespace)
        for record in records:
            if record.values:
                self.registry.ensure_dimension(
                    tenant_id=context.tenant_id,
                    namespace=namespace,
                    vector_length=len(record.values),
                )
            if record.named_vectors:
                for name, values in record.named_vectors.items():
                    self.registry.ensure_dimension(
                        tenant_id=context.tenant_id,
                        namespace=namespace,
                        vector_length=len(values),
                        vector_name=name,
                    )
        self.store.upsert(  # type: ignore[attr-defined]
            tenant_id=context.tenant_id,
            namespace=namespace,
            records=records,
        )
        return UpsertResult(namespace=namespace, upserted=len(records), version=config.version)

    def query(
        self,
        *,
        context: SecurityContext,
        namespace: str,
        query: VectorQuery,
    ) -> Sequence[VectorMatch]:
        """Query vectors in the specified namespace.

        Args:
            context: Security context with tenant and scope information.
            namespace: Target namespace for vector query.
            query: Vector query specification.

        Returns:
            Sequence of matching vectors with scores.

        Raises:
            ScopeError: If context lacks required "index:read" scope.
            DimensionMismatchError: If query vector dimensions don't match namespace.
            RuntimeError: If store operation fails.
        """
        self._require_scope(context, "index:read")
        try:
            self.registry.ensure_dimension(
                tenant_id=context.tenant_id,
                namespace=namespace,
                vector_length=len(query.values),
            )
        except NamespaceNotFoundError:
            raise
        except DimensionMismatchError as exc:
            raise exc
        return self.store.query(  # type: ignore[attr-defined]
            tenant_id=context.tenant_id,
            namespace=namespace,
            query=query,
        )

    def delete(
        self,
        *,
        context: SecurityContext,
        namespace: str,
        ids: Sequence[str],
    ) -> int:
        """Delete vectors from the specified namespace.

        Args:
            context: Security context with tenant and scope information.
            namespace: Target namespace for vector deletion.
            ids: Vector IDs to delete.

        Returns:
            Number of vectors deleted.

        Raises:
            ScopeError: If context lacks required "index:write" scope.
            RuntimeError: If store operation fails.
        """
        self._require_scope(context, "index:write")
        return int(
            self.store.delete(  # type: ignore[attr-defined]
                tenant_id=context.tenant_id,
                namespace=namespace,
                ids=list(ids),
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _require_scope(self, context: SecurityContext, scope: str) -> None:
        """Validate that the security context has the required scope.

        Args:
            context: Security context to validate.
            scope: Required scope for the operation.

        Raises:
            ScopeError: If context lacks the required scope.
        """
        if not context.has_scope(scope):
            raise ScopeError(required_scope=scope)


__all__ = ["VectorStoreService"]
