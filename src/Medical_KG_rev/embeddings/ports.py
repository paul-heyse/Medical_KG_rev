"""Protocol and data models used by embedding adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable
import re


EmbeddingKind = Literal["single_vector", "multi_vector", "sparse", "neural_sparse"]


NAMESPACE_PATTERN = re.compile(
    r"^(?P<kind>[a-z_]+)\.(?P<model>[a-z0-9_\-]+)\.(?P<dim>\d+|auto)\.(?P<version>v[0-9]+(?:\.[0-9]+)*)$"
)


@dataclass(slots=True, frozen=True)
class EmbeddingRecord:
    """Normalized representation of an embedding produced by any adapter."""

    id: str
    tenant_id: str
    namespace: str
    model_id: str
    model_version: str
    kind: EmbeddingKind
    dim: int | None = None
    vectors: list[list[float]] | None = None
    terms: dict[str, float] | None = None
    neural_fields: dict[str, Any] | None = None
    normalized: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.vectors is not None:
            normalized_vectors = [list(map(float, vector)) for vector in self.vectors]
            object.__setattr__(self, "vectors", normalized_vectors)
        if self.terms is not None:
            object.__setattr__(self, "terms", {str(k): float(v) for k, v in self.terms.items()})
        if self.neural_fields is not None:
            object.__setattr__(self, "neural_fields", dict(self.neural_fields))
        if self.kind in {"single_vector", "multi_vector"}:
            if not self.vectors:
                raise ValueError("Dense embeddings must provide vectors")
            first = self.vectors[0]
            for vector in self.vectors:
                if len(vector) != len(first):
                    raise ValueError("All vectors must share the same dimensionality")
            if self.dim is not None and self.dim != len(first):
                raise ValueError("Vector dimensionality does not match declared dim")
        if self.kind == "sparse" and self.terms is None:
            raise ValueError("Sparse embeddings must provide term weights")
        if self.kind == "neural_sparse" and not (self.neural_fields or self.vectors):
            raise ValueError("Neural sparse embeddings must provide neural fields or vectors")


@dataclass(slots=True)
class EmbedderConfig:
    """Configuration for an embedder instance loaded from YAML or environment."""

    name: str
    provider: str
    kind: EmbeddingKind
    namespace: str
    model_id: str
    model_version: str = "v1"
    dim: int | None = None
    pooling: Literal["mean", "max", "cls", "last_token", "none"] | None = "mean"
    normalize: bool = True
    batch_size: int = 32
    requires_gpu: bool = False
    prefixes: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    tenant_scoped: bool = True

    def __post_init__(self) -> None:
        if not NAMESPACE_PATTERN.match(self.namespace):
            raise ValueError(
                "Namespace must follow {kind}.{model}.{dim}.{version} (e.g. dense.bge.1024.v1)"
            )
        if self.kind in {"single_vector", "multi_vector"} and self.dim is None:
            raise ValueError("Dense embedders must declare expected dimensionality")
        if self.batch_size < 1 or self.batch_size > 4096:
            raise ValueError("Batch size must be between 1 and 4096")
        # Ensure we hold independent copies of mutable defaults
        object.__setattr__(self, "prefixes", dict(self.prefixes))
        object.__setattr__(self, "parameters", dict(self.parameters))

    @property
    def namespace_parts(self) -> dict[str, str]:
        match = NAMESPACE_PATTERN.match(self.namespace)
        if not match:  # pragma: no cover - validated in __post_init__
            raise ValueError("Invalid namespace format")
        return match.groupdict()


@dataclass(slots=True)
class EmbeddingRequest:
    """Request payload sent to an embedder implementation."""

    tenant_id: str
    namespace: str
    texts: Sequence[str]
    ids: Sequence[str] | None = None
    correlation_id: str | None = None
    metadata: Sequence[dict[str, Any]] | None = None


@runtime_checkable
class BaseEmbedder(Protocol):
    """Protocol describing the contract for embedding adapters."""

    name: str
    kind: EmbeddingKind
    config: EmbedderConfig

    def embed_documents(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        """Embed corpus documents/chunks."""

    def embed_queries(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        """Embed search queries."""


__all__ = [
    "BaseEmbedder",
    "EmbedderConfig",
    "EmbeddingKind",
    "EmbeddingRecord",
    "EmbeddingRequest",
]
