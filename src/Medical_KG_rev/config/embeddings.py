"""Embedding configuration loader without external dependencies."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if importlib.util.find_spec("yaml") is not None:  # pragma: no cover - depends on PyYAML
    from yaml import safe_load as _safe_load  # type: ignore
else:  # pragma: no cover - fallback when PyYAML unavailable

    def _safe_load(_: str) -> dict[str, Any]:
        return {}


from Medical_KG_rev.embeddings.ports import EmbedderConfig, EmbeddingKind

DEFAULT_EMBEDDING_CONFIG = Path(__file__).resolve().parents[3] / "config" / "embeddings.yaml"


@dataclass(slots=True)
class NamespaceDefinition:
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
    max_tokens: int | None = None
    tokenizer: str | None = None
    enabled: bool = True
    allowed_scopes: list[str] = field(default_factory=lambda: ["embed:read", "embed:write"])
    allowed_tenants: list[str] = field(default_factory=lambda: ["all"])
    prefixes: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> NamespaceDefinition:
        return cls(
            name=str(data["name"]),
            provider=str(data["provider"]),
            kind=data.get("kind", "single_vector"),
            model_id=str(data["model_id"]),
            model_version=str(data.get("model_version", "v1")),
            dim=data.get("dim"),
            pooling=data.get("pooling"),
            normalize=bool(data.get("normalize", True)),
            batch_size=int(data.get("batch_size", 32)),
            requires_gpu=bool(data.get("requires_gpu", False)),
            max_tokens=(int(data["max_tokens"]) if data.get("max_tokens") is not None else None),
            tokenizer=(str(data["tokenizer"]) if data.get("tokenizer") else None),
            enabled=bool(data.get("enabled", True)),
            allowed_scopes=list(data.get("allowed_scopes", ["embed:read", "embed:write"])),
            allowed_tenants=list(data.get("allowed_tenants", ["all"])),
            prefixes=dict(data.get("prefixes", {})),
            parameters=dict(data.get("parameters", {})),
        )

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
            max_tokens=self.max_tokens,
            tokenizer=self.tokenizer,
            enabled=self.enabled,
            allowed_scopes=list(self.allowed_scopes),
            allowed_tenants=list(self.allowed_tenants),
            prefixes=self.prefixes,
            parameters=self.parameters,
        )


@dataclass(slots=True)
class EmbeddingsConfiguration:
    active_namespaces: list[str] = field(default_factory=list)
    namespaces: dict[str, NamespaceDefinition] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> EmbeddingsConfiguration:
        namespaces: dict[str, NamespaceDefinition] = {}
        for namespace, raw in (data.get("namespaces") or {}).items():
            namespaces[namespace] = NamespaceDefinition.from_mapping(raw)
        return cls(
            active_namespaces=list(data.get("active_namespaces", [])),
            namespaces=namespaces,
        )

    def to_embedder_configs(self) -> list[EmbedderConfig]:
        return [
            definition.to_embedder_config(namespace)
            for namespace, definition in self.namespaces.items()
        ]


def load_embeddings_config(path: Path | None = None) -> EmbeddingsConfiguration:
    config_path = path or Path(os.environ.get("MK_EMBEDDINGS_CONFIG", DEFAULT_EMBEDDING_CONFIG))
    data: dict[str, Any] = {}
    if config_path.exists():
        data = _safe_load(config_path.read_text()) or {}
    return EmbeddingsConfiguration.from_mapping(data)


__all__ = [
    "DEFAULT_EMBEDDING_CONFIG",
    "EmbeddingsConfiguration",
    "NamespaceDefinition",
    "load_embeddings_config",
]
