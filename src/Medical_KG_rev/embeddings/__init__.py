"""Universal embedding system covering dense, sparse, and multi-vector paradigms."""

from .ports import BaseEmbedder, EmbedderConfig, EmbeddingRecord, EmbeddingKind
from .registry import EmbedderFactory, EmbedderRegistry
from .namespace import NamespaceManager, NamespaceConfig
from .storage import StorageRouter
from .providers import register_builtin_embedders

__all__ = [
    "BaseEmbedder",
    "EmbedderConfig",
    "EmbeddingRecord",
    "EmbeddingKind",
    "EmbedderFactory",
    "EmbedderRegistry",
    "NamespaceManager",
    "NamespaceConfig",
    "register_builtin_embedders",
    "StorageRouter",
]
