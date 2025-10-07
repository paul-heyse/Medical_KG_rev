"""Exception hierarchy for the chunking system."""

from __future__ import annotations


class ChunkingError(RuntimeError):
    """Base error for chunking related failures."""


class ChunkerConfigurationError(ChunkingError):
    """Raised when chunker configuration is invalid or missing."""


class ChunkerRegistryError(ChunkingError):
    """Raised when registry operations fail."""
