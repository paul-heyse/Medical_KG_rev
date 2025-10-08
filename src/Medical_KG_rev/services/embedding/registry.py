"""Configuration-backed embedding model registry for service usage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from Medical_KG_rev.config.embeddings import (
    EmbeddingsConfiguration,
    load_embeddings_config,
)
from Medical_KG_rev.embeddings.namespace import NamespaceManager
from Medical_KG_rev.embeddings.ports import BaseEmbedder, EmbedderConfig
from Medical_KG_rev.embeddings.providers import register_builtin_embedders
from Medical_KG_rev.embeddings.registry import EmbedderFactory, EmbedderRegistry
from Medical_KG_rev.embeddings.storage import StorageRouter
from Medical_KG_rev.services.embedding.namespace.loader import load_namespace_configs
from Medical_KG_rev.services.embedding.namespace.registry import (
    EmbeddingNamespaceRegistry,
)
from Medical_KG_rev.services.embedding.namespace.schema import NamespaceConfig

logger = structlog.get_logger(__name__)


_DEFAULT_NAMESPACES: dict[str, NamespaceConfig] = {
    "single_vector.qwen3.4096.v1": NamespaceConfig(
        name="qwen3-embedding",
        provider="vllm",
        kind="single_vector",
        model_id="Qwen/Qwen2.5-Embedding-8B-Instruct",
        model_version="v1",
        dim=4096,
        pooling="mean",
        normalize=True,
        batch_size=64,
        requires_gpu=True,
        endpoint="http://vllm-qwen3:8001/v1",
        parameters={"timeout": 60, "max_tokens": 8192},
    ),
    "sparse.splade_v3.400.v1": NamespaceConfig(
        name="splade-v3",
        provider="pyserini",
        kind="sparse",
        model_id="naver/splade-v3",
        model_version="v3",
        dim=400,
        normalize=False,
        batch_size=32,
        parameters={"top_k": 400, "mode": "document", "max_terms": 400},
    ),
    "multi_vector.colbert_v2.128.v1": NamespaceConfig(
        name="colbert-v2",
        provider="colbert",
        kind="multi_vector",
        model_id="colbert-v2",
        model_version="v2",
        dim=128,
        normalize=False,
        batch_size=16,
        parameters={"max_doc_tokens": 180},
    ),
}


@dataclass(slots=True)
class EmbeddingModelRegistry:
    """Loads embedding configs and instantiates adapters on demand."""

    gpu_manager: Any | None = None
    namespace_manager: NamespaceManager | None = None
    config_path: str | Path | None = None
    storage_router: StorageRouter | None = None
    namespace_registry: EmbeddingNamespaceRegistry | None = None
    _namespace_configs: dict[str, NamespaceConfig] = field(init=False, default_factory=dict)
    _config: EmbeddingsConfiguration = field(init=False)
    _registry: EmbedderRegistry = field(init=False)
    _factory: EmbedderFactory = field(init=False)
    _configs_by_name: dict[str, EmbedderConfig] = field(init=False, default_factory=dict)
    _configs_by_namespace: dict[str, EmbedderConfig] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self.namespace_manager = self.namespace_manager or NamespaceManager()
        self.storage_router = self.storage_router or StorageRouter()
        self._config = self._load_configuration(self.config_path)
        self.namespace_registry = self.namespace_registry or EmbeddingNamespaceRegistry()
        self._registry = EmbedderRegistry(namespace_manager=self.namespace_manager)
        register_builtin_embedders(self._registry)
        self._factory = EmbedderFactory(self._registry)
        self._namespace_configs = self._load_namespace_configs(self.config_path)
        self._prime_configs()
        self._register_namespaces()

    def _load_configuration(self, config_path: str | Path | None) -> EmbeddingsConfiguration:
        path_obj = Path(config_path) if config_path else None
        try:
            return load_embeddings_config(path_obj)
        except FileNotFoundError:
            logger.warning(
                "embedding.registry.config_missing",
                path=str(path_obj) if path_obj else "<default>",
                fallback=True,
            )
            return EmbeddingsConfiguration(
                active_namespaces=list(_DEFAULT_NAMESPACES.keys()),
                namespaces={},
            )

    def _load_namespace_configs(
        self, config_path: str | Path | None
    ) -> dict[str, NamespaceConfig]:
        path_obj = Path(config_path) if config_path else None
        directory = None
        if path_obj is not None:
            directory = path_obj.parent / "embedding" / "namespaces"
        return load_namespace_configs(directory, fallback_config=self._config) or dict(_DEFAULT_NAMESPACES)

    def _prime_configs(self) -> None:
        self.namespace_registry.reset()
        self._configs_by_namespace.clear()
        self._configs_by_name.clear()
        for namespace, config in self._namespace_configs.items():
            self.namespace_registry.register(namespace, config)
            embedder_config = config.to_embedder_config(namespace)
            self._configs_by_namespace[namespace] = embedder_config
            self._configs_by_name[embedder_config.name] = embedder_config
        if not self._configs_by_namespace:
            for namespace, config in _DEFAULT_NAMESPACES.items():
                self.namespace_registry.register(namespace, config)
                embedder_config = config.to_embedder_config(namespace)
                self._configs_by_namespace[namespace] = embedder_config
                self._configs_by_name[embedder_config.name] = embedder_config

    def _register_namespaces(self) -> None:
        self.namespace_manager.reset()
        for config in self._configs_by_namespace.values():
            self.namespace_manager.register(config)

    @property
    def factory(self) -> EmbedderFactory:
        return self._factory

    @property
    def registry(self) -> EmbedderRegistry:
        return self._registry

    @property
    def configuration(self) -> EmbeddingsConfiguration:
        return self._config

    def list_configs(self) -> list[EmbedderConfig]:
        return list(self._configs_by_namespace.values())

    def active_configs(self) -> list[EmbedderConfig]:
        actives = [
            self._configs_by_namespace[ns]
            for ns in self._config.active_namespaces
            if ns in self._configs_by_namespace
        ]
        if actives:
            return actives
        return self.list_configs()

    def resolve(
        self,
        *,
        models: Sequence[str] | None = None,
        namespaces: Sequence[str] | None = None,
    ) -> list[EmbedderConfig]:
        if namespaces:
            missing = [ns for ns in namespaces if ns not in self._configs_by_namespace]
            if missing:
                available = ", ".join(self.namespace_registry.list_namespaces())
                raise ValueError(
                    f"Namespaces {', '.join(missing)} not found. Available: {available}"
                )
            return [self._configs_by_namespace[ns] for ns in namespaces]
        if models:
            missing = [name for name in models if name not in self._configs_by_name]
            if missing:
                available = ", ".join(sorted(self._configs_by_name))
                raise ValueError(
                    f"Models {', '.join(missing)} not found. Available: {available}"
                )
            return [self._configs_by_name[name] for name in models]
        return self.active_configs()

    def get(self, key: str | EmbedderConfig) -> BaseEmbedder:
        config = self._resolve_config(key)
        return self._factory.get(config)

    def config_for(self, key: str) -> EmbedderConfig:
        config = self._resolve_config(key)
        return config

    def _resolve_config(self, key: str | EmbedderConfig) -> EmbedderConfig:
        if isinstance(key, EmbedderConfig):
            return key
        if key in self._configs_by_namespace:
            return self._configs_by_namespace[key]
        if key in self._configs_by_name:
            return self._configs_by_name[key]
        available = ", ".join(sorted(self._configs_by_namespace))
        msg = f"Unknown embedder configuration '{key}'. Available namespaces: {available}"
        logger.error("embedding.registry.config_missing_entry", key=key)
        raise ValueError(msg)

    def reload(self, *, config_path: str | Path | None = None) -> None:
        """Reload embedding configurations and refresh namespace registrations."""

        if config_path is not None:
            self.config_path = config_path
        self._config = self._load_configuration(self.config_path)
        self._namespace_configs = self._load_namespace_configs(self.config_path)
        self._prime_configs()
        self._register_namespaces()
        logger.info(
            "embedding.registry.reloaded",
            namespaces=list(self._configs_by_namespace.keys()),
        )


__all__ = ["EmbeddingModelRegistry"]
