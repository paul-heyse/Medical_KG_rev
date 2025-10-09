"""Runtime registry for embedding namespaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import structlog

logger = structlog.get_logger(__name__)

from .schema import EmbeddingKind, NamespaceConfig


@dataclass(slots=True)
class EmbeddingNamespaceRegistry:
    """In-memory registry storing namespace configurations by identifier."""

    _namespaces: dict[str, NamespaceConfig] = field(default_factory=dict)
    _tokenizers: dict[str, object] = field(default_factory=dict, init=False, repr=False)

    def register(self, namespace: str, config: NamespaceConfig) -> None:
        self._namespaces[namespace] = config
        self._tokenizers.pop(namespace, None)

    def bulk_register(self, configs: Mapping[str, NamespaceConfig]) -> None:
        for namespace, config in configs.items():
            self.register(namespace, config)

    def reset(self) -> None:
        self._namespaces.clear()
        self._tokenizers.clear()

    def get(self, namespace: str) -> NamespaceConfig:
        try:
            return self._namespaces[namespace]
        except KeyError as exc:  # pragma: no cover - exercised via tests
            available = ", ".join(sorted(self._namespaces))
            raise ValueError(
                f"Namespace '{namespace}' not found. Available: {available}"
                if available
                else "No namespaces registered"
            ) from exc

    def list_namespaces(self) -> list[str]:
        return sorted(self._namespaces)

    def list_by_kind(self, kind: EmbeddingKind) -> list[str]:
        return sorted(
            namespace for namespace, config in self._namespaces.items() if config.kind == kind
        )

    def exists(self, namespace: str, *, include_disabled: bool = False) -> bool:
        config = self._namespaces.get(namespace)
        if not config:
            return False
        return bool(config.enabled or include_disabled)

    def list_enabled(
        self,
        *,
        tenant_id: str | None = None,
        scope: str | None = None,
    ) -> list[tuple[str, NamespaceConfig]]:
        configs: list[tuple[str, NamespaceConfig]] = []
        for namespace, config in self._namespaces.items():
            if not config.enabled:
                continue
            if scope and scope not in config.allowed_scopes and "*" not in config.allowed_scopes:
                continue
            tenants = config.allowed_tenants
            if tenant_id is not None and "all" not in tenants and tenant_id not in tenants:
                continue
            configs.append((namespace, config))
        return sorted(configs, key=lambda item: item[0])

    def get_provider(self, namespace: str) -> str:
        return self.get(namespace).provider

    def get_dimension(self, namespace: str) -> int | None:
        return self.get(namespace).dim

    def get_max_tokens(self, namespace: str) -> int | None:
        return self.get(namespace).max_tokens

    def get_allowed_scopes(self, namespace: str) -> list[str]:
        return list(self.get(namespace).allowed_scopes)

    def get_allowed_tenants(self, namespace: str) -> list[str]:
        return list(self.get(namespace).allowed_tenants)

    def get_tokenizer(self, namespace: str):  # pragma: no cover - exercised via validation tests
        if namespace in self._tokenizers:
            return self._tokenizers[namespace]
        config = self.get(namespace)
        if not config.tokenizer:
            raise ValueError(f"Namespace '{namespace}' does not define a tokenizer")
        try:
            from transformers import AutoTokenizer  # type: ignore import-not-found
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("transformers package is required for tokenizer validation") from exc
        logger.debug(
            "embedding.namespace.tokenizer.load",
            namespace=namespace,
            tokenizer=config.tokenizer,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        self._tokenizers[namespace] = tokenizer
        return tokenizer

    def __contains__(self, namespace: str) -> bool:  # pragma: no cover - convenience
        return namespace in self._namespaces


__all__ = ["EmbeddingNamespaceRegistry"]
