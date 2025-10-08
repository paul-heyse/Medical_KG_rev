"""Service entry-points for chunking operations."""

from __future__ import annotations

from .port import (
    CHUNKER_REGISTRY,
    Chunk,
    ChunkerPort,
    ChunkerRegistrationError,
    UnknownChunkerError,
    chunk_document,
    get_chunker,
    register_chunker,
    reset_registry,
)
from .registry import register_defaults

__all__ = [
    "CHUNKER_REGISTRY",
    "Chunk",
    "ChunkerPort",
    "ChunkerRegistrationError",
    "UnknownChunkerError",
    "chunk_document",
    "get_chunker",
    "register_chunker",
    "register_defaults",
    "reset_registry",
]

# Register the lightweight chunkers eagerly so that callers can immediately use
# the port without performing additional plumbing.  Optional dependencies are
# handled gracefully by the registration helpers.
register_defaults()
