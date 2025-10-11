"""Universal embedding system covering dense, sparse, and multi-vector paradigms."""

from .namespace import NamespaceConfig, NamespaceManager
from .ports import BaseEmbedder, EmbedderConfig, EmbeddingKind, EmbeddingRecord
from .providers import register_builtin_embedders
from .registry import EmbedderFactory, EmbedderRegistry
from .storage import StorageRouter


__all__ = [
    "BaseEmbedder",
    "EmbedderConfig",
    "EmbedderFactory",
    "EmbedderRegistry",
    "EmbeddingKind",
    "EmbeddingRecord",
    "NamespaceConfig",
    "NamespaceManager",
    "StorageRouter",
    "register_builtin_embedders",
]
