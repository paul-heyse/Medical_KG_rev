"""Exception hierarchy for the chunking system."""

from __future__ import annotations


class ChunkingError(RuntimeError):
    """Base error for chunking related failures."""


class ChunkerConfigurationError(ChunkingError):
    """Raised when chunker configuration is invalid or missing."""


class ChunkerRegistryError(ChunkingError):
    """Raised when registry operations fail."""


class InvalidDocumentError(ChunkingError):
    """Raised when an invalid or unsupported document is provided."""


class ChunkingUnavailableError(ChunkingError):
    """Raised when the chunking circuit breaker is open."""

    def __init__(self, retry_after: float) -> None:
        super().__init__("Chunking temporarily unavailable due to repeated failures")
        self.retry_after = max(retry_after, 0.0)
