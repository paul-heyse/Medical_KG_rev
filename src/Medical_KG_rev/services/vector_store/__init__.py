"""Vector storage service abstractions and implementations."""

from .errors import (
    BackendUnavailableError,
    DimensionMismatchError,
    InvalidNamespaceConfigError,
    NamespaceNotFoundError,
    ResourceExhaustedError,
    ScopeError,
    VectorStoreError,
)
from .factory import VectorStoreFactory
from .models import (
    CompressionPolicy,
    IndexParams,
    NamespaceConfig,
    UpsertResult,
    VectorMatch,
    VectorQuery,
    VectorRecord,
)
from .registry import NamespaceRegistry
from .service import VectorStoreService
from .stores.faiss import FaissVectorStore
from .stores.memory import InMemoryVectorStore
from .stores.qdrant import QdrantVectorStore
from .types import VectorStorePort

__all__ = [
    "BackendUnavailableError",
    "CompressionPolicy",
    "DimensionMismatchError",
    "InvalidNamespaceConfigError",
    "FaissVectorStore",
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "IndexParams",
    "NamespaceConfig",
    "NamespaceNotFoundError",
    "NamespaceRegistry",
    "ResourceExhaustedError",
    "ScopeError",
    "UpsertResult",
    "VectorMatch",
    "VectorQuery",
    "VectorRecord",
    "VectorStoreError",
    "VectorStoreFactory",
    "VectorStorePort",
    "VectorStoreService",
]
