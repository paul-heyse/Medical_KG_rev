"""Core interfaces and data models for the universal embedding system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EmbeddingKind = Literal["single_vector", "multi_vector", "sparse", "neural_sparse"]


NAMESPACE_PATTERN = re.compile(
    r"^(?P<kind>[a-z_]+)\.(?P<model>[a-z0-9_\-]+)\.(?P<dim>\d+|auto)\.(?P<version>v[0-9]+(?:\.[0-9]+)*)$"
)


class EmbeddingRecord(BaseModel):
    """Normalized representation of an embedding produced by any adapter."""

    model_config = ConfigDict(frozen=True)

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
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: str | None = None

    @model_validator(mode="after")
    def _validate_payload(self) -> "EmbeddingRecord":
        if self.kind in {"single_vector", "multi_vector"}:
            if not self.vectors:
                raise ValueError("Dense embeddings must provide vectors")
            if any(len(vector) != len(self.vectors[0]) for vector in self.vectors):
                raise ValueError("All vectors must share the same dimensionality")
            if self.dim is not None and self.dim != len(self.vectors[0]):
                raise ValueError("Vector dimensionality does not match declared dim")
        if self.kind == "sparse" and not self.terms:
            raise ValueError("Sparse embeddings must provide term weights")
        if self.kind == "neural_sparse" and not (self.neural_fields or self.vectors):
            raise ValueError("Neural sparse embeddings must provide neural fields or vectors")
        return self


class EmbedderConfig(BaseModel):
    """Configuration for an embedder instance loaded from YAML or environment."""

    model_config = ConfigDict(extra="allow")

    name: str
    provider: str
    kind: EmbeddingKind
    namespace: str
    model_id: str
    model_version: str = "v1"
    dim: int | None = None
    pooling: Literal["mean", "max", "cls", "last_token", "none"] | None = "mean"
    normalize: bool = True
    batch_size: int = Field(default=32, ge=1, le=4096)
    requires_gpu: bool = False
    prefixes: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    tenant_scoped: bool = True

    @field_validator("namespace")
    @classmethod
    def _validate_namespace(cls, value: str) -> str:
        if not NAMESPACE_PATTERN.match(value):
            raise ValueError(
                "Namespace must follow {kind}.{model}.{dim}.{version} (e.g. dense.bge.1024.v1)"
            )
        return value

    @model_validator(mode="after")
    def _validate_dimension(self) -> "EmbedderConfig":
        if self.kind in {"single_vector", "multi_vector"} and self.dim is None:
            raise ValueError("Dense embedders must declare expected dimensionality")
        return self

    @property
    def namespace_parts(self) -> dict[str, str]:
        match = NAMESPACE_PATTERN.match(self.namespace)
        if not match:  # pragma: no cover - validated above
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
