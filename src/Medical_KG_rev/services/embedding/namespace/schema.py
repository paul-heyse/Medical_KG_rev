"""Typed namespace configuration models for embedding registrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

try:  # pragma: no cover - optional dependency guard
    from pydantic import BaseModel, Field  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when pydantic missing
    BaseModel = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]

from Medical_KG_rev.embeddings.ports import EmbedderConfig, EmbeddingKind as AdapterEmbeddingKind


class EmbeddingKind(str, Enum):
    """Supported embedding families for namespace registration."""

    SINGLE_VECTOR = "single_vector"
    SPARSE = "sparse"
    MULTI_VECTOR = "multi_vector"
    NEURAL_SPARSE = "neural_sparse"

    @classmethod
    def from_adapter(cls, value: AdapterEmbeddingKind) -> "EmbeddingKind":
        return cls(value)  # type: ignore[arg-type]


if BaseModel is not None:

    class NamespaceConfig(BaseModel):
        """Validated namespace configuration loaded from YAML files."""

        name: str
        kind: EmbeddingKind
        model_id: str
        model_version: str = "v1"
        dim: int | None = None
        provider: str
        endpoint: str | None = None
        parameters: Mapping[str, Any] = Field(default_factory=dict)
        pooling: str | None = "mean"
        normalize: bool = True
        batch_size: int = 32
        requires_gpu: bool = False

        model_config = {"frozen": True}

        def to_embedder_config(self, namespace: str) -> EmbedderConfig:
            """Convert the namespace definition into an embedder adapter config."""

            param_map = dict(self.parameters)
            if self.endpoint and "endpoint" not in param_map:
                param_map["endpoint"] = self.endpoint
            return EmbedderConfig(
                name=self.name,
                provider=self.provider,
                kind=self.kind.value,
                namespace=namespace,
                model_id=self.model_id,
                model_version=self.model_version,
                dim=self.dim,
                pooling=self.pooling if self.pooling else "mean",
                normalize=self.normalize,
                batch_size=self.batch_size,
                requires_gpu=self.requires_gpu,
                parameters=param_map,
            )

else:

    @dataclass(slots=True, frozen=True)
    class NamespaceConfig:  # type: ignore[no-redef]
        """Minimal fallback configuration when pydantic is unavailable."""

        name: str
        kind: EmbeddingKind | str
        model_id: str
        model_version: str = "v1"
        dim: int | None = None
        provider: str = ""
        endpoint: str | None = None
        parameters: Mapping[str, Any] = field(default_factory=dict)
        pooling: str | None = "mean"
        normalize: bool = True
        batch_size: int = 32
        requires_gpu: bool = False

        def __post_init__(self) -> None:
            value = self.kind.value if isinstance(self.kind, EmbeddingKind) else str(self.kind)
            object.__setattr__(self, "kind", EmbeddingKind(value))
            object.__setattr__(self, "parameters", dict(self.parameters))

        def to_embedder_config(self, namespace: str) -> EmbedderConfig:
            param_map = dict(self.parameters)
            if self.endpoint and "endpoint" not in param_map:
                param_map["endpoint"] = self.endpoint
            return EmbedderConfig(
                name=self.name,
                provider=self.provider,
                kind=self.kind.value,
                namespace=namespace,
                model_id=self.model_id,
                model_version=self.model_version,
                dim=self.dim,
                pooling=self.pooling if self.pooling else "mean",
                normalize=self.normalize,
                batch_size=self.batch_size,
                requires_gpu=self.requires_gpu,
                parameters=param_map,
            )


__all__ = ["EmbeddingKind", "NamespaceConfig"]
