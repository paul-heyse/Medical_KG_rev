"""Chunker port and registry utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from Medical_KG_rev.models.ir import Document

from .validation import ensure_valid_chunks


@dataclass(slots=True)
class Chunk:
    """Normalized representation of a chunk."""

    chunk_id: str
    doc_id: str
    text: str
    char_offsets: tuple[int, int]
    section_label: str
    intent_hint: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ChunkerPort(Protocol):
    """Protocol describing required chunking behaviour."""

    def chunk(self, document: Document, *, profile: str) -> list[Chunk]:
        ...


ChunkerFactory = Callable[..., ChunkerPort]
CHUNKER_REGISTRY: dict[str, ChunkerFactory] = {}


class UnknownChunkerError(RuntimeError):
    pass


class ChunkerRegistrationError(RuntimeError):
    pass


def _coerce_factory(factory: ChunkerFactory | type[ChunkerPort]) -> ChunkerFactory:
    if isinstance(factory, type):
        return lambda **kwargs: factory(**kwargs)  # type: ignore[misc]
    return factory


def register_chunker(name: str, factory: ChunkerFactory | type[ChunkerPort]) -> None:
    if name in CHUNKER_REGISTRY:
        raise ChunkerRegistrationError(f"Chunker '{name}' already registered")
    CHUNKER_REGISTRY[name] = _coerce_factory(factory)


def get_chunker(name: str, **factory_kwargs: Any) -> ChunkerPort:
    try:
        factory = CHUNKER_REGISTRY[name]
    except KeyError as exc:
        raise UnknownChunkerError(f"Chunker '{name}' is not registered") from exc
    return factory(**factory_kwargs)


def chunk_document(
    document: Document,
    *,
    profile_name: str,
    profile_loader: Callable[[str], dict[str, Any]],
) -> list[Chunk]:
    profile = profile_loader(profile_name)
    chunker_type = profile.get("chunker_type")
    if not isinstance(chunker_type, str):
        raise UnknownChunkerError("profile missing 'chunker_type'")
    chunker = get_chunker(chunker_type, profile=profile)
    chunks = chunker.chunk(document, profile=profile_name)
    ensure_valid_chunks(chunks)
    return chunks


def reset_registry() -> None:
    CHUNKER_REGISTRY.clear()


__all__ = [
    "Chunk",
    "ChunkerPort",
    "ChunkerFactory",
    "register_chunker",
    "get_chunker",
    "chunk_document",
    "reset_registry",
]
