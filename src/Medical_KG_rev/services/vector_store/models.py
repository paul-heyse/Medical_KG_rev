"""Domain models for the vector store subsystem."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class CompressionPolicy:
    """Configuration describing how vectors are compressed in the store."""

    kind: str = "none"
    pq_m: int | None = None
    pq_nbits: int | None = None
    opq_m: int | None = None

    def is_enabled(self) -> bool:
        return self.kind != "none"


@dataclass(slots=True, frozen=True)
class IndexParams:
    """Parameters describing a namespace/collection."""

    dimension: int
    metric: str = "cosine"
    kind: str = "hnsw"
    ef_search: int | None = None
    ef_construct: int | None = None
    m: int | None = None
    nlist: int | None = None
    nprobe: int | None = None
    replicas: int | None = None
    use_gpu: bool = False
    gpu_id: int | None = None
    reorder_k: int | None = None
    train_size: int | None = None
    storage_path: str | None = None


@dataclass(slots=True, frozen=True)
class NamespaceConfig:
    """Runtime configuration for a namespace."""

    name: str
    params: IndexParams
    compression: CompressionPolicy = field(default_factory=CompressionPolicy)
    version: str = "v1"
    named_vectors: Mapping[str, IndexParams] | None = None


@dataclass(slots=True, frozen=True)
class VectorRecord:
    """Represents a vector ready to be persisted."""

    vector_id: str
    values: Sequence[float]
    metadata: Mapping[str, object] = field(default_factory=dict)
    vector_version: str | None = None
    named_vectors: Mapping[str, Sequence[float]] | None = None


@dataclass(slots=True, frozen=True)
class VectorQuery:
    """Query payload for vector similarity search."""

    values: Sequence[float]
    top_k: int = 10
    filters: Mapping[str, object] | None = None
    vector_name: str | None = None
    reorder: bool | None = None


@dataclass(slots=True, frozen=True)
class VectorMatch:
    """Result from a vector similarity query."""

    vector_id: str
    score: float
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class UpsertResult:
    """Summary returned after an upsert operation."""

    namespace: str
    upserted: int
    version: str


@dataclass(slots=True, frozen=True)
class SnapshotInfo:
    """Metadata describing a created snapshot or backup artifact."""

    namespace: str
    path: str
    size_bytes: int | None = None
    created_at: float | None = None
    metadata: Mapping[str, object] | None = None


@dataclass(slots=True, frozen=True)
class RebuildReport:
    """Details returned when a namespace index is retrained or rebuilt."""

    namespace: str
    rebuilt: bool
    details: Mapping[str, object] | None = None


@dataclass(slots=True, frozen=True)
class HealthStatus:
    """Represents readiness information for a namespace or backend."""

    name: str
    healthy: bool
    details: Mapping[str, object] | None = None
