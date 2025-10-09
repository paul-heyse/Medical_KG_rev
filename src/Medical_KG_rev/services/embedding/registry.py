"""Configuration-backed embedding model registry for service usage.

This module provides a centralized registry for embedding models and configurations,
loading them from configuration files and providing on-demand instantiation of
embedders. It manages namespace configurations, model registrations, and provides
resolution capabilities for embedding operations.

Key Responsibilities:
    - Load embedding configurations from files and environment
    - Register embedding models and namespaces
    - Provide embedder instantiation on demand
    - Resolve models by name or namespace
    - Manage configuration reloading and updates
    - Support fallback to default configurations

Collaborators:
    - Upstream: Embedding coordinators, policy evaluators
    - Downstream: EmbedderFactory, EmbedderRegistry, NamespaceManager

Side Effects:
    - Loads configuration files from disk
    - Registers embedders with global registries
    - Updates namespace manager with configurations
    - Logs configuration loading and errors

Thread Safety:
    - Thread-safe: All operations use immutable configurations
    - Safe for concurrent access after initialization

Performance Characteristics:
    - O(1) lookup time for registered models and namespaces
    - Configuration loading occurs during initialization
    - Embedder instantiation is lazy and cached

Example:
    >>> from Medical_KG_rev.services.embedding.registry import EmbeddingModelRegistry
    >>> registry = EmbeddingModelRegistry()
    >>> embedder = registry.get("single_vector.qwen3.4096.v1")
    >>> configs = registry.active_configs()
    >>> print(f"Active models: {len(configs)}")
"""

# ============================================================================
# IMPORTS
# ============================================================================

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

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = structlog.get_logger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================


_DEFAULT_NAMESPACES: dict[str, NamespaceConfig] = {
    """Default namespace configurations for embedding models.

    Provides fallback configurations when no external configuration
    files are available. Includes configurations for different embedding
    types: single vector (dense), sparse, and multi-vector models.

    The configurations include:
        - Qwen3: Dense single-vector embeddings with 4096 dimensions
        - SPLADE v3: Sparse embeddings with 400 dimensions
        - ColBERT v2: Multi-vector embeddings with 128 dimensions

    Example:
        >>> from Medical_KG_rev.services.embedding.registry import _DEFAULT_NAMESPACES
        >>> qwen_config = _DEFAULT_NAMESPACES["single_vector.qwen3.4096.v1"]
        >>> print(f"Model: {qwen_config.model_id}, Dims: {qwen_config.dim}")
    """
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


# ============================================================================
# REGISTRY IMPLEMENTATION
# ============================================================================


@dataclass(slots=True)
class EmbeddingModelRegistry:
    """Loads embedding configs and instantiates adapters on demand.

    Central registry for embedding models that loads configurations from
    files and provides on-demand instantiation of embedders. Manages
    namespace configurations, model registrations, and provides resolution
    capabilities for embedding operations.

    Attributes:
        gpu_manager: Optional GPU resource manager for GPU-required models
        namespace_manager: Manager for embedding namespaces
        config_path: Path to configuration file or directory
        storage_router: Router for storage operations
        namespace_registry: Registry for namespace configurations
        _namespace_configs: Internal cache of namespace configurations
        _config: Loaded embeddings configuration
        _registry: Embedder registry for model instantiation
        _factory: Factory for creating embedder instances
        _configs_by_name: Cache of configurations by model name
        _configs_by_namespace: Cache of configurations by namespace

    Invariants:
        - _config is never None after initialization
        - _registry and _factory are properly initialized
        - Configuration caches are consistent with loaded configs

    Thread Safety:
        - Thread-safe: All operations use immutable configurations
        - Safe for concurrent access after initialization

    Lifecycle:
        - Initialized with optional dependencies
        - Loads configurations during __post_init__
        - Registers models and namespaces
        - Provides embedder instantiation on demand

    Example:
        >>> registry = EmbeddingModelRegistry()
        >>> embedder = registry.get("single_vector.qwen3.4096.v1")
        >>> configs = registry.active_configs()
        >>> print(f"Available models: {len(configs)}")
    """

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
        """Initialize the registry with configurations and dependencies.

        Sets up all required components including namespace manager,
        storage router, embedder registry, and loads configurations
        from files or defaults.

        Note:
            This method is called automatically after dataclass
            initialization and performs all setup operations.
        """
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
        """Load embeddings configuration from file or use defaults.

        Args:
            config_path: Path to configuration file. If None, uses default config.

        Returns:
            Loaded embeddings configuration with active namespaces and settings.

        Note:
            Falls back to default configuration if file is not found,
            logging a warning about the fallback.
        """
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
        """Load namespace configurations from directory or use defaults.

        Args:
            config_path: Path to configuration file. Used to determine
                namespace configuration directory.

        Returns:
            Dictionary mapping namespace names to their configurations.

        Note:
            Looks for namespace configs in embedding/namespaces subdirectory
            relative to the config file path.
        """
        path_obj = Path(config_path) if config_path else None
        directory = None
        if path_obj is not None:
            directory = path_obj.parent / "embedding" / "namespaces"
        return load_namespace_configs(directory, fallback_config=self._config) or dict(_DEFAULT_NAMESPACES)

    def _prime_configs(self) -> None:
        """Initialize configuration caches and register namespaces.

        Clears existing caches and registers all namespace configurations
        with the namespace registry and internal caches.

        Note:
            This method is called during initialization and when
            configurations are reloaded.
        """
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
        """Register all configurations with the namespace manager.

        Resets the namespace manager and registers all embedder
        configurations for namespace resolution.

        Note:
            This method is called during initialization and when
            configurations are reloaded.
        """
        self.namespace_manager.reset()
        for config in self._configs_by_namespace.values():
            self.namespace_manager.register(config)

    @property
    def factory(self) -> EmbedderFactory:
        """Get the embedder factory for creating embedder instances.

        Returns:
            Factory instance for creating embedders from configurations.
        """
        return self._factory

    @property
    def registry(self) -> EmbedderRegistry:
        """Get the embedder registry for model management.

        Returns:
            Registry instance containing registered embedder types.
        """
        return self._registry

    @property
    def configuration(self) -> EmbeddingsConfiguration:
        """Get the current embeddings configuration.

        Returns:
            Current configuration with active namespaces and settings.
        """
        return self._config

    def list_configs(self) -> list[EmbedderConfig]:
        """List all available embedder configurations.

        Returns:
            List of all registered embedder configurations.
        """
        return list(self._configs_by_namespace.values())

    def active_configs(self) -> list[EmbedderConfig]:
        """Get configurations for active namespaces.

        Returns:
            List of configurations for namespaces marked as active.
            Falls back to all configurations if no active namespaces
            are specified.

        Example:
            >>> registry = EmbeddingModelRegistry()
            >>> active = registry.active_configs()
            >>> print(f"Active models: {len(active)}")
        """
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
        """Resolve embedder configurations by model names or namespaces.

        Args:
            models: Optional sequence of model names to resolve.
            namespaces: Optional sequence of namespace names to resolve.

        Returns:
            List of embedder configurations matching the specified
            models or namespaces.

        Raises:
            ValueError: If any specified models or namespaces are not found.

        Example:
            >>> registry = EmbeddingModelRegistry()
            >>> configs = registry.resolve(namespaces=["single_vector.qwen3.4096.v1"])
            >>> print(f"Resolved {len(configs)} configurations")
        """
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
        """Get an embedder instance for the specified key.

        Args:
            key: Namespace name, model name, or embedder configuration.

        Returns:
            Embedder instance for the specified configuration.

        Raises:
            ValueError: If the key is not found in registered configurations.

        Example:
            >>> registry = EmbeddingModelRegistry()
            >>> embedder = registry.get("single_vector.qwen3.4096.v1")
            >>> embeddings = embedder.embed(["sample text"])
        """
        config = self._resolve_config(key)
        return self._factory.get(config)

    def config_for(self, key: str) -> EmbedderConfig:
        """Get embedder configuration for the specified key.

        Args:
            key: Namespace name or model name.

        Returns:
            Embedder configuration for the specified key.

        Raises:
            ValueError: If the key is not found in registered configurations.

        Example:
            >>> registry = EmbeddingModelRegistry()
            >>> config = registry.config_for("single_vector.qwen3.4096.v1")
            >>> print(f"Model: {config.name}, Dimensions: {config.dim}")
        """
        config = self._resolve_config(key)
        return config

    def _resolve_config(self, key: str | EmbedderConfig) -> EmbedderConfig:
        """Resolve embedder configuration from key.

        Args:
            key: Namespace name, model name, or embedder configuration.

        Returns:
            Embedder configuration for the specified key.

        Raises:
            ValueError: If the key is not found in registered configurations.
        """
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
        """Reload embedding configurations and refresh namespace registrations.

        Args:
            config_path: Optional new path to configuration file.
                If provided, updates the registry's config path.

        Note:
            This method reloads all configurations from files and
            reinitializes the registry with the new settings.

        Example:
            >>> registry = EmbeddingModelRegistry()
            >>> registry.reload(config_path="/new/config/path")
            >>> print("Configuration reloaded")
        """
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


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["EmbeddingModelRegistry"]
