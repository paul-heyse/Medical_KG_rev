"""Factory for config-driven chunker instantiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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

    def create(self, config: ChunkerConfig, *, allow_experimental: bool = False) -> RegisteredChunker:
        chunker = self.registry.create(config, allow_experimental=allow_experimental)
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
