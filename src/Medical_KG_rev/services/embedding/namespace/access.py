"""Namespace access control helpers for embedding services."""

from __future__ import annotations

from dataclasses import dataclass

from Medical_KG_rev.services.embedding.namespace.registry import EmbeddingNamespaceRegistry


@dataclass(frozen=True)
class NamespaceAccessResult:
    """Result of namespace access validation."""

    namespace: str
    tenant_id: str
    scope: str
    allowed: bool
    reason: str | None = None


def validate_namespace_access(
    registry: EmbeddingNamespaceRegistry,
    *,
    namespace: str,
    tenant_id: str,
    required_scope: str,
) -> NamespaceAccessResult:
    """Validate whether a tenant with a scope may access a namespace."""

    try:
        config = registry.get(namespace)
    except ValueError as exc:  # pragma: no cover - surfaced to caller
        return NamespaceAccessResult(
            namespace=namespace,
            tenant_id=tenant_id,
            scope=required_scope,
            allowed=False,
            reason=str(exc),
        )

    if not config.enabled:
        return NamespaceAccessResult(
            namespace=namespace,
            tenant_id=tenant_id,
            scope=required_scope,
            allowed=False,
            reason="Namespace disabled",
        )

    scopes = set(config.allowed_scopes)
    if required_scope not in scopes and "*" not in scopes:
        return NamespaceAccessResult(
            namespace=namespace,
            tenant_id=tenant_id,
            scope=required_scope,
            allowed=False,
            reason=f"Scope '{required_scope}' not permitted",
        )

    tenants = set(config.allowed_tenants)
    if "all" not in tenants and tenant_id not in tenants:
        return NamespaceAccessResult(
            namespace=namespace,
            tenant_id=tenant_id,
            scope=required_scope,
            allowed=False,
            reason=f"Tenant '{tenant_id}' not allowed",
        )

    return NamespaceAccessResult(
        namespace=namespace,
        tenant_id=tenant_id,
        scope=required_scope,
        allowed=True,
        reason=None,
    )


__all__ = ["NamespaceAccessResult", "validate_namespace_access"]
