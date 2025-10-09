"""Factory for config-driven chunker instantiation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from threading import RLock

from .configuration import ChunkerSettings
from .exceptions import ChunkerConfigurationError
from .models import ChunkerConfig
from .ports import BaseChunker
from .registry import ChunkerRegistry, default_registry


@dataclass(slots=True)
class RegisteredChunker:
    instance: BaseChunker
    granularity: str | None


class ChunkerFactory:
    """Instantiates chunkers from configuration using the registry."""

    def __init__(self, registry: ChunkerRegistry | None = None) -> None:
        self.registry = registry or default_registry()
        self._cache: dict[tuple[object, ...], BaseChunker] = {}
        self._lock = RLock()

    def create(self, config: ChunkerConfig, *, allow_experimental: bool = False) -> RegisteredChunker:
        entry = self.registry.list_chunkers(include_experimental=True).get(config.name)
        if entry is None:
            raise ChunkerConfigurationError(f"Chunker '{config.name}' is not registered")
        if entry.experimental and not allow_experimental:
            raise ChunkerConfigurationError(
                f"Chunker '{config.name}' is experimental and not enabled"
            )
        key = self._cache_key(config)
        with self._lock:
            chunker = self._cache.get(key)
            if chunker is None:
                chunker = entry.factory(**config.params)
                self._cache[key] = chunker
        return RegisteredChunker(instance=chunker, granularity=config.granularity)

    def create_many(
        self, settings: Iterable[ChunkerSettings], *, allow_experimental: bool = False
    ) -> list[RegisteredChunker]:
        registered: list[RegisteredChunker] = []
        for setting in settings:
            registered.append(
                self.create(
                    setting.to_config(), allow_experimental=allow_experimental
                )
            )
        if not registered:
            raise ChunkerConfigurationError("At least one chunker must be configured")
        return registered

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def _cache_key(self, config: ChunkerConfig) -> tuple[object, ...]:
        return (
            config.name,
            self._freeze_value(config.params),
        )

    def _freeze_value(self, value: object) -> object:
        if isinstance(value, Mapping):
            return tuple(
                (key, self._freeze_value(subvalue))
                for key, subvalue in sorted(value.items(), key=lambda item: item[0])
            )
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_value(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(self._freeze_value(item) for item in value))
        return value
