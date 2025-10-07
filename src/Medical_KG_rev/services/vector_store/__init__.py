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
    HealthStatus,
    IndexParams,
    NamespaceConfig,
    RebuildReport,
    SnapshotInfo,
    UpsertResult,
    VectorMatch,
    VectorQuery,
    VectorRecord,
)
from .registry import NamespaceRegistry
from .service import VectorStoreService
from .stores.external import (
    AnnoyIndex,
    ChromaStore,
    DiskANNStore,
    DuckDBVSSStore,
    HNSWLibIndex,
    LanceDBStore,
    NMSLibIndex,
    PgvectorStore,
    ScaNNIndex,
    VespaStore,
    WeaviateStore,
)
from .stores.faiss import FaissVectorStore
from .stores.memory import InMemoryVectorStore
from .stores.milvus import MilvusVectorStore
from .stores.opensearch import OpenSearchKNNStore
from .stores.qdrant import QdrantVectorStore
from .types import VectorStorePort

__all__ = [
    "BackendUnavailableError",
    "CompressionPolicy",
    "HealthStatus",
    "DimensionMismatchError",
    "InvalidNamespaceConfigError",
    "AnnoyIndex",
    "ChromaStore",
    "DiskANNStore",
    "DuckDBVSSStore",
    "FaissVectorStore",
    "HNSWLibIndex",
    "InMemoryVectorStore",
    "LanceDBStore",
    "MilvusVectorStore",
    "NMSLibIndex",
    "OpenSearchKNNStore",
    "PgvectorStore",
    "QdrantVectorStore",
    "ScaNNIndex",
    "VespaStore",
    "WeaviateStore",
    "IndexParams",
    "NamespaceConfig",
    "NamespaceNotFoundError",
    "NamespaceRegistry",
    "RebuildReport",
    "ResourceExhaustedError",
    "SnapshotInfo",
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
