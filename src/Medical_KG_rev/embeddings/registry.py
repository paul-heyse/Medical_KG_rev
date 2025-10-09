"""Registry for embedding adapters with lazy loading support."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .namespace import NamespaceManager
from .ports import BaseEmbedder, EmbedderConfig

EmbedderFactoryCallable = Callable[[EmbedderConfig], BaseEmbedder]


@dataclass(slots=True)
class EmbedderRegistry:
    """Stores embedder factories keyed by provider identifier."""

    namespace_manager: NamespaceManager
    _factories: dict[str, EmbedderFactoryCallable] = field(default_factory=dict)

    def register(self, provider: str, factory: EmbedderFactoryCallable) -> None:
        self._factories[provider] = factory

    def create(self, config: EmbedderConfig) -> BaseEmbedder:
        try:
            factory = self._factories[config.provider]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown embedder provider '{config.provider}'") from exc
        embedder = factory(config)
        self.namespace_manager.register(config)
        return embedder


@dataclass(slots=True)
class EmbedderFactory:
    """High level factory that instantiates embedders from configuration blocks."""

    registry: EmbedderRegistry
    cache: dict[str, BaseEmbedder] = field(default_factory=dict)

    def get(self, config: EmbedderConfig) -> BaseEmbedder:
        cache_key = config.namespace
        if cache_key not in self.cache:
            self.cache[cache_key] = self.registry.create(config)
        return self.cache[cache_key]
