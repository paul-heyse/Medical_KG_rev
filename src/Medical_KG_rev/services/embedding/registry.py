"""Lightweight registry for embedding models and namespace configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import logging

from .namespace.registry import NamespaceRegistry
from .schema import EmbeddingKind, NamespaceConfig

logger = logging.getLogger(__name__)

_DEFAULT_NAMESPACES: Dict[str, NamespaceConfig] = {
    "default": NamespaceConfig(
        name="default",
        kind=EmbeddingKind.SINGLE_VECTOR,
        model_id="bert-base-uncased",
        tokenizer="bert-base-uncased",
        max_tokens=512,
        dim=768,
        provider="huggingface",
    ),
    "biomedical": NamespaceConfig(
        name="biomedical",
        kind=EmbeddingKind.SINGLE_VECTOR,
        model_id="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        tokenizer="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        max_tokens=512,
        dim=768,
        provider="huggingface",
    ),
}


@dataclass
class EmbeddingModelRegistry:
    """In-memory registry used by the embedding worker components."""

    config_path: Path | None = None
    _namespace_registry: NamespaceRegistry = field(default_factory=NamespaceRegistry)
    _models: Dict[str, Any] = field(default_factory=dict)
    _configs: Dict[str, NamespaceConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._load_defaults()

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------
    def register_model(self, name: str, model: Any) -> None:
        self._models[name] = model
        logger.info("embedding.registry.model.registered", name=name)

    def get_model(self, name: str) -> Any | None:
        return self._models.get(name)

    def unregister_model(self, name: str) -> bool:
        return self._models.pop(name, None) is not None

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    # ------------------------------------------------------------------
    # Namespace management
    # ------------------------------------------------------------------
    def register_namespace(self, namespace: str, config: NamespaceConfig) -> None:
        self._configs[namespace] = config
        self._namespace_registry.register_namespace(namespace, config)
        logger.info("embedding.registry.namespace.registered", namespace=namespace)

    def unregister_namespace(self, namespace: str) -> bool:
        removed = self._configs.pop(namespace, None) is not None
        if removed:
            self._namespace_registry.unregister_namespace(namespace)
        return removed

    def get_namespace_config(self, namespace: str) -> NamespaceConfig | None:
        return self._configs.get(namespace)

    def list_namespaces(self) -> list[str]:
        return list(self._configs.keys())

    def get_namespace_registry(self) -> NamespaceRegistry:
        return self._namespace_registry

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def reload(self, *, config_path: str | Path | None = None) -> None:
        if config_path is not None:
            self.config_path = Path(config_path)
        self.clear()
        self._load_defaults()

    def clear(self) -> None:
        self._models.clear()
        self._configs.clear()
        self._namespace_registry.clear()

    def health_check(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "models": len(self._models),
            "namespaces": len(self._configs),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "models": list(self._models.keys()),
            "namespaces": list(self._configs.keys()),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_defaults(self) -> None:
        for namespace, config in _DEFAULT_NAMESPACES.items():
            self.register_namespace(namespace, config)


# Global singleton -----------------------------------------------------------
_registry_singleton: EmbeddingModelRegistry | None = None


def get_embedding_model_registry() -> EmbeddingModelRegistry:
    global _registry_singleton
    if _registry_singleton is None:
        _registry_singleton = EmbeddingModelRegistry()
    return _registry_singleton


def create_embedding_model_registry(config_path: str | Path | None = None) -> EmbeddingModelRegistry:
    return EmbeddingModelRegistry(config_path=Path(config_path) if config_path else None)


__all__ = [
    "EmbeddingModelRegistry",
    "get_embedding_model_registry",
    "create_embedding_model_registry",
]
