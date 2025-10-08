"""Runtime registry for embedding namespaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .schema import EmbeddingKind, NamespaceConfig


@dataclass(slots=True)
class EmbeddingNamespaceRegistry:
    """In-memory registry storing namespace configurations by identifier."""

    _namespaces: dict[str, NamespaceConfig] = field(default_factory=dict)

    def register(self, namespace: str, config: NamespaceConfig) -> None:
        self._namespaces[namespace] = config

    def bulk_register(self, configs: Mapping[str, NamespaceConfig]) -> None:
        for namespace, config in configs.items():
            self.register(namespace, config)

    def reset(self) -> None:
        self._namespaces.clear()

    def get(self, namespace: str) -> NamespaceConfig:
        try:
            return self._namespaces[namespace]
        except KeyError as exc:  # pragma: no cover - exercised via tests
            available = ", ".join(sorted(self._namespaces))
            raise ValueError(
                f"Namespace '{namespace}' not found. Available: {available}" if available else "No namespaces registered"
            ) from exc

    def list_namespaces(self) -> list[str]:
        return sorted(self._namespaces)

    def list_by_kind(self, kind: EmbeddingKind) -> list[str]:
        return sorted(namespace for namespace, config in self._namespaces.items() if config.kind == kind)

    def __contains__(self, namespace: str) -> bool:  # pragma: no cover - convenience
        return namespace in self._namespaces


__all__ = ["EmbeddingNamespaceRegistry"]
