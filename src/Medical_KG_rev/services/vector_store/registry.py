"""Namespace registry for vector store collections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import DimensionMismatchError, InvalidNamespaceConfigError, NamespaceNotFoundError
from .models import NamespaceConfig


@dataclass(slots=True)
class NamespaceRegistry:
    """In-memory registry of namespaces keyed by tenant."""

    _namespaces: dict[str, dict[str, NamespaceConfig]]

    def __init__(self) -> None:
        self._namespaces = {}

    def register(self, *, tenant_id: str, config: NamespaceConfig) -> None:
        if not 128 <= config.params.dimension <= 4096:
            raise InvalidNamespaceConfigError(
                config.name,
                detail="Vector dimension must be between 128 and 4096.",
            )
        tenant_namespaces = self._namespaces.setdefault(tenant_id, {})
        existing = tenant_namespaces.get(config.name)
        if existing and existing.params.dimension != config.params.dimension:
            raise DimensionMismatchError(
                existing.params.dimension,
                config.params.dimension,
                namespace=config.name,
            )
        tenant_namespaces[config.name] = config

    def get(self, *, tenant_id: str, namespace: str) -> NamespaceConfig:
        tenant_namespaces = self._namespaces.get(tenant_id)
        if not tenant_namespaces or namespace not in tenant_namespaces:
            raise NamespaceNotFoundError(namespace, tenant_id=tenant_id)
        return tenant_namespaces[namespace]

    def ensure_dimension(
        self,
        *,
        tenant_id: str,
        namespace: str,
        vector_length: int,
    ) -> None:
        config = self.get(tenant_id=tenant_id, namespace=namespace)
        if config.params.dimension != vector_length:
            raise DimensionMismatchError(
                config.params.dimension, vector_length, namespace=namespace
            )

    def list(self, *, tenant_id: str) -> Mapping[str, NamespaceConfig]:
        return dict(self._namespaces.get(tenant_id, {}))
