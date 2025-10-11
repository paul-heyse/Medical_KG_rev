"""Vector storage service abstractions."""

from .errors import HttpClient
from .models import HttpClient
from .registry import NamespaceRegistry
from .service import VectorStoreService
from .types import VectorStorePort


__all__ = [
    "BackendUnavailableError",
    "CompressionPolicy",
    "DimensionMismatchError",
    "HealthStatus",
    "IndexParams",
    "InvalidNamespaceConfigError",
    "NamespaceConfig",
    "NamespaceNotFoundError",
    "NamespaceRegistry",
    "RebuildReport",
    "ResourceExhaustedError",
    "ScopeError",
    "SnapshotInfo",
    "UpsertResult",
    "VectorMatch",
    "VectorQuery",
    "VectorRecord",
    "VectorStoreError",
    "VectorStorePort",
    "VectorStoreService",
]
