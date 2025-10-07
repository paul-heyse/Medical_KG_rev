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
from .gpu import GPUFallbackStrategy, GPUResourceManager, get_gpu_stats, plan_batches, summarise_stats
from .models import (
    HealthStatus,
    NamespaceConfig,
    RebuildReport,
    SnapshotInfo,
    UpsertResult,
    VectorMatch,
    VectorQuery,
    VectorRecord,
)
from .monitoring import record_memory_usage, record_vector_operation
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
        gpu_manager: GPUResourceManager | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.audit = audit_trail or get_audit_trail()
        self.failure_threshold = failure_threshold
        self.recovery_window_seconds = recovery_window_seconds
        self._failure_count = 0
        self._circuit_opened_at: float | None = None
        self._gpu_manager = gpu_manager or GPUResourceManager()
        self._gpu_strategy = GPUFallbackStrategy(
            logger=lambda code: logger.info("vector.gpu_fallback", code=code)
        )

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
        records = list(records)
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
        gpu_available = self._gpu_strategy.guard(
            operation="vector_upsert", require_gpu=self._gpu_manager.require_gpu
        )
        batches = plan_batches(
            len(records), manager=self._gpu_manager, logger=lambda msg: logger.info("vector.batch", message=msg)
        )
        try:
            self._guard_circuit_breaker()
            for batch_range in batches:
                batch_records = records[batch_range.start : batch_range.stop]
                if not batch_records:
                    continue
                self.store.upsert(
                    tenant_id=context.tenant_id,
                    namespace=namespace,
                    records=batch_records,
                )
        except VectorStoreError as error:
            self._record_failure(error)
            raise
        else:
            self._reset_failures()
        duration = time.perf_counter() - start
        record_vector_operation("upsert", namespace, duration, len(records))
        record_memory_usage(namespace, self._estimate_memory(records))
        self.audit.record(
            context=context,
            action="vector.upsert",
            resource=namespace,
            metadata={
                "count": len(records),
                "duration_ms": round(duration * 1000, 3),
                "version": namespace_config.version,
                "gpu": {
                    "used": gpu_available,
                    **summarise_stats(get_gpu_stats()),
                },
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
        gpu_available = self._gpu_strategy.guard(
            operation="vector_query", require_gpu=self._gpu_manager.require_gpu
        )
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
        record_vector_operation("query", namespace, duration, len(matches))
        self.audit.record(
            context=context,
            action="vector.query",
            resource=namespace,
            metadata={
                "top_k": query.top_k,
                "duration_ms": round(duration * 1000, 3),
                "returned": len(matches),
                "gpu": {
                    "used": gpu_available,
                    **summarise_stats(get_gpu_stats()),
                },
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

    def create_snapshot(
        self,
        *,
        context: SecurityContext,
        namespace: str,
        destination: str,
        include_payloads: bool = True,
    ) -> SnapshotInfo:
        self._require_scope(context, "index:read")
        self.registry.get(tenant_id=context.tenant_id, namespace=namespace)
        try:
            self._guard_circuit_breaker()
            info = self.store.create_snapshot(
                tenant_id=context.tenant_id,
                namespace=namespace,
                destination=destination,
                include_payloads=include_payloads,
            )
        except VectorStoreError as error:
            self._record_failure(error)
            raise
        else:
            self._reset_failures()
        self.audit.record(
            context=context,
            action="vector.snapshot",
            resource=namespace,
            metadata={
                "path": info.path,
                "size_bytes": info.size_bytes,
                "include_payloads": include_payloads,
            },
        )
        return info

    def restore_snapshot(
        self,
        *,
        context: SecurityContext,
        namespace: str,
        source: str,
        overwrite: bool = False,
    ) -> RebuildReport:
        self._require_scope(context, "index:write")
        self.registry.get(tenant_id=context.tenant_id, namespace=namespace)
        try:
            self._guard_circuit_breaker()
            report = self.store.restore_snapshot(
                tenant_id=context.tenant_id,
                namespace=namespace,
                source=source,
                overwrite=overwrite,
            )
        except VectorStoreError as error:
            self._record_failure(error)
            raise
        else:
            self._reset_failures()
        self.audit.record(
            context=context,
            action="vector.restore",
            resource=namespace,
            metadata={"restored": report.rebuilt, "source": source},
        )
        return report

    def rebuild_namespace(
        self,
        *,
        context: SecurityContext,
        namespace: str,
        force: bool = False,
    ) -> RebuildReport:
        self._require_scope(context, "index:write")
        self.registry.get(tenant_id=context.tenant_id, namespace=namespace)
        try:
            self._guard_circuit_breaker()
            report = self.store.rebuild_index(
                tenant_id=context.tenant_id,
                namespace=namespace,
                force=force,
            )
        except VectorStoreError as error:
            self._record_failure(error)
            raise
        else:
            self._reset_failures()
        self.audit.record(
            context=context,
            action="vector.rebuild",
            resource=namespace,
            metadata={"force": force, "rebuilt": report.rebuilt},
        )
        return report

    def check_health(
        self,
        *,
        context: SecurityContext,
        namespace: str | None = None,
    ) -> Mapping[str, HealthStatus]:
        self._require_scope(context, "index:read")
        return self.store.check_health(
            tenant_id=context.tenant_id,
            namespace=namespace,
        )

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

    def _estimate_memory(self, records: Sequence[VectorRecord]) -> int:
        total = 0
        for record in records:
            if record.values:
                total += len(record.values) * 4
            if record.named_vectors:
                for values in record.named_vectors.values():
                    total += len(values) * 4
        return total

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
