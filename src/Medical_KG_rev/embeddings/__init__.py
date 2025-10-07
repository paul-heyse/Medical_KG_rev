"""Universal embedding system covering dense, sparse, and multi-vector paradigms."""

from .ports import BaseEmbedder, EmbedderConfig, EmbeddingRecord, EmbeddingKind
from .registry import EmbedderFactory, EmbedderRegistry
from .namespace import NamespaceManager, NamespaceConfig
from .storage import StorageRouter

__all__ = [
    "BaseEmbedder",
    "EmbedderConfig",
    "EmbeddingRecord",
    "EmbeddingKind",
    "EmbedderFactory",
    "EmbedderRegistry",
    "NamespaceManager",
    "NamespaceConfig",
    "StorageRouter",
]
