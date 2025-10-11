"""Namespace configuration utilities for embedding services."""

from .loader import DEFAULT_NAMESPACE_DIR, load_namespace_configs
from .registry import EmbeddingNamespaceRegistry
from .schema import EmbeddingKind, NamespaceConfig


__all__ = [
    "DEFAULT_NAMESPACE_DIR",
    "EmbeddingKind",
    "EmbeddingNamespaceRegistry",
    "NamespaceConfig",
    "load_namespace_configs",
]
