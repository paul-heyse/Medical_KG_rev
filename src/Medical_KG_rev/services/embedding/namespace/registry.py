"""Registry for embedding namespaces."""

from __future__ import annotations

import logging
from typing import Any

import structlog

from .schema import EmbeddingKind, NamespaceConfig

logger = structlog.get_logger(__name__)


class NamespaceRegistry:
    """Registry for managing embedding namespaces."""

    def __init__(self) -> None:
        """Initialize the namespace registry."""
        self.logger = logger
        self._namespaces: dict[str, NamespaceConfig] = {}
        self._tokenizers: dict[str, Any] = {}

    def register_namespace(self, namespace: str, config: NamespaceConfig) -> None:
        """Register a namespace configuration."""
        self._namespaces[namespace] = config
        self.logger.info(f"Registered namespace: {namespace}")

    def get_namespace_config(self, namespace: str) -> NamespaceConfig | None:
        """Get namespace configuration."""
        return self._namespaces.get(namespace)

    def list_namespaces(self) -> list[str]:
        """List all registered namespaces."""
        return list(self._namespaces.keys())

    def unregister_namespace(self, namespace: str) -> bool:
        """Unregister a namespace."""
        if namespace in self._namespaces:
            del self._namespaces[namespace]
            if namespace in self._tokenizers:
                del self._tokenizers[namespace]
            self.logger.info(f"Unregistered namespace: {namespace}")
            return True
        return False

    def get_tokenizer(self, namespace: str) -> Any | None:
        """Get tokenizer for namespace."""
        if namespace in self._tokenizers:
            return self._tokenizers[namespace]

        config = self.get_namespace_config(namespace)
        if not config:
            return None

        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("transformers package is required for tokenizer validation") from exc

        self.logger.debug(
            "embedding.namespace.tokenizer.load",
            namespace=namespace,
            tokenizer=config.tokenizer,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        self._tokenizers[namespace] = tokenizer
        return tokenizer

    def __contains__(self, namespace: str) -> bool:  # pragma: no cover - convenience
        """Check if namespace is registered."""
        return namespace in self._namespaces

    def __len__(self) -> int:  # pragma: no cover - convenience
        """Get number of registered namespaces."""
        return len(self._namespaces)

    def __iter__(self):  # pragma: no cover - convenience
        """Iterate over registered namespaces."""
        return iter(self._namespaces)

    def health_check(self) -> dict[str, Any]:
        """Check registry health."""
        return {
            "registry": "namespace",
            "status": "healthy",
            "namespaces": len(self._namespaces),
            "tokenizers": len(self._tokenizers),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_namespaces": len(self._namespaces),
            "namespaces": list(self._namespaces.keys()),
            "tokenizers_loaded": len(self._tokenizers),
            "tokenizer_namespaces": list(self._tokenizers.keys()),
        }

    def clear(self) -> None:
        """Clear all namespaces and tokenizers."""
        self._namespaces.clear()
        self._tokenizers.clear()
        self.logger.info("Cleared all namespaces and tokenizers")

    def validate_namespace(self, namespace: str) -> bool:
        """Validate namespace configuration."""
        config = self.get_namespace_config(namespace)
        if not config:
            return False

        try:
            # Validate tokenizer
            tokenizer = self.get_tokenizer(namespace)
            if not tokenizer:
                return False

            # Validate embedding kind
            if config.embedding_kind not in [EmbeddingKind.SINGLE_VECTOR, EmbeddingKind.MULTI_VECTOR]:
                return False

            return True

        except Exception as exc:
            self.logger.warning(f"Namespace validation failed for {namespace}: {exc}")
            return False

    def get_namespace_info(self, namespace: str) -> dict[str, Any] | None:
        """Get detailed information about a namespace."""
        config = self.get_namespace_config(namespace)
        if not config:
            return None

        return {
            "namespace": namespace,
            "config": config.model_dump(),
            "tokenizer_loaded": namespace in self._tokenizers,
            "valid": self.validate_namespace(namespace),
        }


# Global namespace registry instance
_namespace_registry: NamespaceRegistry | None = None


def get_namespace_registry() -> NamespaceRegistry:
    """Get the global namespace registry instance."""
    global _namespace_registry

    if _namespace_registry is None:
        _namespace_registry = NamespaceRegistry()

    return _namespace_registry


def create_namespace_registry() -> NamespaceRegistry:
    """Create a new namespace registry instance."""
    return NamespaceRegistry()


def register_default_namespaces(registry: NamespaceRegistry) -> None:
    """Register default namespaces."""
    # Register default namespace
    default_config = NamespaceConfig(
        embedding_kind=EmbeddingKind.SINGLE_VECTOR,
        tokenizer="bert-base-uncased",
        max_tokens=512,
        dimensions=768,
    )
    registry.register_namespace("default", default_config)

    # Register biomedical namespace
    biomedical_config = NamespaceConfig(
        embedding_kind=EmbeddingKind.SINGLE_VECTOR,
        tokenizer="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        max_tokens=512,
        dimensions=768,
    )
    registry.register_namespace("biomedical", biomedical_config)

    # Register clinical namespace
    clinical_config = NamespaceConfig(
        embedding_kind=EmbeddingKind.SINGLE_VECTOR,
        tokenizer="emilyalsentzer/Bio_ClinicalBERT",
        max_tokens=512,
        dimensions=768,
    )
    registry.register_namespace("clinical", clinical_config)

    logger.info("Registered default namespaces")
