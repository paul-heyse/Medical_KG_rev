"""Vector storage service abstractions."""

from .errors import (
    BackendUnavailableError,
    DimensionMismatchError,
    InvalidNamespaceConfigError,
    NamespaceNotFoundError,
    ResourceExhaustedError,
    ScopeError,
    VectorStoreError,
)
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
