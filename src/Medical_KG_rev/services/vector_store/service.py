"""High-level service that orchestrates vector store operations."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence

import structlog

from Medical_KG_rev.auth.audit import AuditTrail, get_audit_trail
from Medical_KG_rev.auth.context import SecurityContext

from .errors import (
    BackendUnavailableError,
    NamespaceNotFoundError,
    ResourceExhaustedError,
    ScopeError,
    VectorStoreError,
)
from .models import NamespaceConfig, UpsertResult, VectorMatch, VectorQuery, VectorRecord
from .registry import NamespaceRegistry
from .types import VectorStorePort

logger = structlog.get_logger(__name__)


class VectorStoreService:
    """Coordinates namespace governance, security, and auditing for vector storage."""

    def __init__(
        self,
        store: VectorStorePort,
        registry: NamespaceRegistry,
        *,
        audit_trail: AuditTrail | None = None,
        failure_threshold: int = 5,
        recovery_window_seconds: float = 30.0,
    ) -> None:
        self.store = store
        self.registry = registry
        self.audit = audit_trail or get_audit_trail()
        self.failure_threshold = failure_threshold
        self.recovery_window_seconds = recovery_window_seconds
        self._failure_count = 0
        self._circuit_opened_at: float | None = None

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
        """Register namespace and propagate to the backing store."""

        self._require_scope(context, "index:write")
        logger.info(
            "vector.namespace.ensure",
            tenant_id=context.tenant_id,
            namespace=config.name,
            version=config.version,
        )
        self.registry.register(tenant_id=context.tenant_id, config=config)
        metadata_payload: dict[str, object] = {"version": config.version, **(metadata or {})}
        if config.named_vectors:
            metadata_payload.setdefault(
                "named_vectors",
                {
                    name: {
                        "dimension": params.dimension,
                        "metric": params.metric,
                        "kind": params.kind,
                        "ef_construct": params.ef_construct,
                        "m": params.m,
                    }
                    for name, params in config.named_vectors.items()
                },
            )
        self.store.create_or_update_collection(
            tenant_id=context.tenant_id,
            namespace=config.name,
            params=config.params,
            compression=config.compression,
            metadata=metadata_payload,
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
        self._require_scope(context, "index:write")
        if not records:
            return UpsertResult(namespace=namespace, upserted=0, version="")
        namespace_config = self.registry.get(
            tenant_id=context.tenant_id, namespace=namespace
        )
        for record in records:
            if record.values:
                self.registry.ensure_dimension(
                    tenant_id=context.tenant_id,
                    namespace=namespace,
                    vector_length=len(record.values),
                )
            if record.named_vectors:
                for vector_name, values in record.named_vectors.items():
                    self.registry.ensure_dimension(
                        tenant_id=context.tenant_id,
                        namespace=namespace,
                        vector_length=len(values),
                        vector_name=vector_name,
                    )
        start = time.perf_counter()
        try:
            self._guard_circuit_breaker()
            self.store.upsert(
                tenant_id=context.tenant_id,
                namespace=namespace,
                records=records,
            )
        except VectorStoreError as error:
            self._record_failure(error)
            raise
        else:
            self._reset_failures()
        duration = time.perf_counter() - start
        self.audit.record(
            context=context,
            action="vector.upsert",
            resource=namespace,
            metadata={
                "count": len(records),
                "duration_ms": round(duration * 1000, 3),
                "version": namespace_config.version,
            },
        )
        logger.info(
            "vector.upsert",
            tenant_id=context.tenant_id,
            namespace=namespace,
            count=len(records),
            duration_ms=duration * 1000,
        )
        return UpsertResult(
            namespace=namespace,
            upserted=len(records),
            version=namespace_config.version,
        )

    def query(
        self,
        *,
        context: SecurityContext,
        namespace: str,
        query: VectorQuery,
    ) -> Sequence[VectorMatch]:
        self._require_scope(context, "index:read")
        self.registry.ensure_dimension(
            tenant_id=context.tenant_id,
            namespace=namespace,
            vector_length=len(query.values),
            vector_name=query.vector_name,
        )
        start = time.perf_counter()
        try:
            self._guard_circuit_breaker()
            matches = list(
                self.store.query(
                    tenant_id=context.tenant_id,
                    namespace=namespace,
                    query=query,
                )
            )
        except VectorStoreError as error:
            self._record_failure(error)
            raise
        else:
            self._reset_failures()
        duration = time.perf_counter() - start
        self.audit.record(
            context=context,
            action="vector.query",
            resource=namespace,
            metadata={
                "top_k": query.top_k,
                "duration_ms": round(duration * 1000, 3),
                "returned": len(matches),
            },
        )
        logger.info(
            "vector.query",
            tenant_id=context.tenant_id,
            namespace=namespace,
            top_k=query.top_k,
            returned=len(matches),
            duration_ms=duration * 1000,
        )
        return matches

    def delete(
        self,
        *,
        context: SecurityContext,
        namespace: str,
        vector_ids: Sequence[str],
    ) -> int:
        self._require_scope(context, "index:write")
        self.registry.get(tenant_id=context.tenant_id, namespace=namespace)
        try:
            self._guard_circuit_breaker()
            removed = self.store.delete(
                tenant_id=context.tenant_id,
                namespace=namespace,
                vector_ids=vector_ids,
            )
        except VectorStoreError as error:
            self._record_failure(error)
            raise
        else:
            self._reset_failures()
        logger.info(
            "vector.delete",
            tenant_id=context.tenant_id,
            namespace=namespace,
            removed=removed,
        )
        if removed:
            self.audit.record(
                context=context,
                action="vector.delete",
                resource=namespace,
                metadata={"removed": removed},
            )
        return removed

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _require_scope(self, context: SecurityContext, scope: str) -> None:
        if not context.has_scope(scope):
            raise ScopeError(required_scope=scope)

    def _guard_circuit_breaker(self) -> None:
        if self._circuit_opened_at is None:
            return
        elapsed = time.perf_counter() - self._circuit_opened_at
        if elapsed < self.recovery_window_seconds:
            raise BackendUnavailableError(
                "Vector store circuit breaker open", retry_after=self.recovery_window_seconds - elapsed
            )
        self._circuit_opened_at = None
        self._failure_count = 0

    def _record_failure(self, error: VectorStoreError) -> None:
        logger.warning("vector.store.failure", error=error, error_type=type(error).__name__)
        if isinstance(error, BackendUnavailableError):
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._circuit_opened_at = time.perf_counter()
        elif isinstance(error, ResourceExhaustedError):
            # Resource exhaustion is terminal for the request; reset failures so breaker stays closed
            self._reset_failures()
        else:
            self._failure_count += 1

    def _reset_failures(self) -> None:
        self._failure_count = 0
        self._circuit_opened_at = None

    def bulk_upsert(
        self,
        *,
        context: SecurityContext,
        namespace: str,
        batches: Iterable[Sequence[VectorRecord]],
    ) -> UpsertResult:
        total = 0
        for batch in batches:
            result = self.upsert(context=context, namespace=namespace, records=batch)
            total += result.upserted
        version = self.registry.get(
            tenant_id=context.tenant_id, namespace=namespace
        ).version
        return UpsertResult(namespace=namespace, upserted=total, version=version)
