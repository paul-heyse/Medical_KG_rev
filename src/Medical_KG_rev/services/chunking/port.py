"""Chunker port and helper utilities.

This module defines a protocol-based port that all chunking implementations
must implement. It also provides a lightweight registry used by the new
profile-driven chunking system introduced by the `add-parsing-chunking-
normalization` change.  The implementation intentionally keeps the registry
stateful in module scope so that integrations can perform registration during
import without requiring a global service locator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Protocol

from Medical_KG_rev.models.ir import Document

if False:  # pragma: no cover - imported for typing only
    from typing import Type


@dataclass(slots=True)
class Chunk:
    """Normalized representation of a chunk returned by chunkers."""

    chunk_id: str
    doc_id: str
    text: str
    char_offsets: tuple[int, int]
    section_label: str | None = None
    intent_hint: str | None = None
    page_bbox: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ChunkerPort(Protocol):
    """Protocol describing the required chunking interface."""

    def chunk(self, document: Document, *, profile: str) -> list[Chunk]:
        """Split *document* according to the configuration for *profile*."""


CHUNKER_REGISTRY: Dict[str, Callable[..., ChunkerPort]] = {}


class UnknownChunkerError(RuntimeError):
    """Raised when requesting a chunker that has not been registered."""


class ChunkerRegistrationError(RuntimeError):
    """Raised when attempting to register a duplicate chunker name."""


def register_chunker(name: str, factory: Callable[..., ChunkerPort]) -> None:
    """Register *factory* for *name*.

    The registry is intentionally simple: it stores callables that return
    `ChunkerPort` instances.  This indirection allows implementations to defer
    heavy imports until they are actually required.
    """

    if name in CHUNKER_REGISTRY:
        raise ChunkerRegistrationError(f"Chunker '{name}' already registered")
    CHUNKER_REGISTRY[name] = factory


def get_chunker(name: str, **factory_kwargs: Any) -> ChunkerPort:
    """Return the chunker identified by *name*.

    Args:
        name: Registered chunker identifier.
        **factory_kwargs: Keyword arguments forwarded to the factory.

    Raises:
        UnknownChunkerError: If *name* is not present in the registry.
    """

    try:
        factory = CHUNKER_REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - error path exercised in tests
        raise UnknownChunkerError(f"Chunker '{name}' is not registered") from exc
    return factory(**factory_kwargs)


def chunk_document(
    document: Document,
    *,
    profile_name: str,
    profile_loader: Callable[[str], dict[str, Any]],
) -> list[Chunk]:
    """Helper that resolves *profile_name* and invokes the configured chunker."""

    profile = profile_loader(profile_name)
    chunker_type = profile["chunker_type"]
    chunker = get_chunker(chunker_type, profile=profile)
    return chunker.chunk(document, profile=profile_name)


def reset_registry() -> None:
    """Clear the registry.

    This is primarily intended for tests which need to ensure isolated
    registration state.
    """

    CHUNKER_REGISTRY.clear()
