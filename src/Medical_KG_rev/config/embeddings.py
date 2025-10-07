"""Configuration loader for the universal embedding system."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from Medical_KG_rev.embeddings.ports import EmbedderConfig, EmbeddingKind

DEFAULT_EMBEDDING_CONFIG = Path(__file__).resolve().parents[3] / "config" / "embeddings.yaml"


class NamespaceDefinition(BaseModel):
    name: str
    provider: str
    kind: EmbeddingKind
    model_id: str
    model_version: str = "v1"
    dim: int | None = None
    pooling: str | None = "mean"
    normalize: bool = True
    batch_size: int = 32
    requires_gpu: bool = False
    prefixes: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)

    def to_embedder_config(self, namespace: str) -> EmbedderConfig:
        return EmbedderConfig(
            name=self.name,
            provider=self.provider,
            kind=self.kind,
            namespace=namespace,
            model_id=self.model_id,
            model_version=self.model_version,
            dim=self.dim,
            pooling=self.pooling if self.pooling else "mean",
            normalize=self.normalize,
            batch_size=self.batch_size,
            requires_gpu=self.requires_gpu,
            prefixes=self.prefixes,
            parameters=self.parameters,
        )


class EmbeddingsConfiguration(BaseModel):
    active_namespaces: list[str] = Field(default_factory=list)
    namespaces: dict[str, NamespaceDefinition] = Field(default_factory=dict)

    def to_embedder_configs(self) -> list[EmbedderConfig]:
        configs: list[EmbedderConfig] = []
        for namespace, definition in self.namespaces.items():
            configs.append(definition.to_embedder_config(namespace))
        return configs


def load_embeddings_config(path: Path | None = None) -> EmbeddingsConfiguration:
    config_path = path or Path(os.environ.get("MK_EMBEDDINGS_CONFIG", DEFAULT_EMBEDDING_CONFIG))
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return EmbeddingsConfiguration.model_validate(data)
