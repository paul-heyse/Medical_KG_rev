"""Minimal vector store orchestration service used for tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from Medical_KG_rev.auth.context import SecurityContext

from .errors import DimensionMismatchError, NamespaceNotFoundError, ScopeError
from .models import NamespaceConfig, UpsertResult, VectorMatch, VectorQuery, VectorRecord
from .registry import NamespaceRegistry
from .types import VectorStorePort


class VectorStoreService:
    """Small wrapper that validates scopes and dimensions before delegating to a store."""

    def __init__(self, store: VectorStorePort, registry: NamespaceRegistry) -> None:
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
        if not context.has_scope(scope):
            raise ScopeError(required_scope=scope)


__all__ = ["VectorStoreService"]
