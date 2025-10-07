"""Chunker registry and factory helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Type

from .exceptions import ChunkerConfigurationError, ChunkerRegistryError
from .models import ChunkerConfig
from .ports import BaseChunker


@dataclass(slots=True)
class ChunkerEntry:
    name: str
    factory: Callable[..., BaseChunker]
    experimental: bool = False


class ChunkerRegistry:
    """Registry storing mappings from names to chunker factories."""

    def __init__(self) -> None:
        self._entries: dict[str, ChunkerEntry] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., BaseChunker],
        *,
        experimental: bool = False,
    ) -> None:
        if name in self._entries:
            raise ChunkerRegistryError(f"Chunker '{name}' already registered")
        self._entries[name] = ChunkerEntry(name, factory, experimental=experimental)

    def create(self, config: ChunkerConfig, *, allow_experimental: bool = False) -> BaseChunker:
        entry = self._entries.get(config.name)
        if entry is None:
            raise ChunkerConfigurationError(f"Chunker '{config.name}' is not registered")
        if entry.experimental and not allow_experimental:
            raise ChunkerConfigurationError(
                f"Chunker '{config.name}' is experimental and not enabled"
            )
        return entry.factory(**config.params)

    def list_chunkers(self, *, include_experimental: bool = False) -> dict[str, ChunkerEntry]:
        if include_experimental:
            return dict(self._entries)
        return {name: entry for name, entry in self._entries.items() if not entry.experimental}


def default_registry() -> ChunkerRegistry:
    from .chunkers import (
        ClinicalRoleChunker,
        SectionAwareChunker,
        SemanticSplitterChunker,
        SlidingWindowChunker,
        TableChunker,
    )

    registry = ChunkerRegistry()
    registry.register("section_aware", SectionAwareChunker)
    registry.register("sliding_window", SlidingWindowChunker)
    registry.register("table", TableChunker)
    registry.register("semantic_splitter", SemanticSplitterChunker)
    registry.register("clinical_role", ClinicalRoleChunker)
    return registry
