"""Exception hierarchy for the chunking system."""

from __future__ import annotations

from typing import Sequence


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


class ProfileNotFoundError(ChunkingError):
    """Raised when a requested chunking profile does not exist."""

    def __init__(self, profile: str, available: Sequence[str]) -> None:
        self.profile = profile
        self.available = tuple(sorted(available))
        message = (
            f"Chunking profile '{profile}' not found. Available: {', '.join(self.available) or 'none'}"
        )
        super().__init__(message)


class TokenizerMismatchError(ChunkingError):
    """Raised when the configured tokenizer is incompatible with the embedding model."""

    def __init__(self, tokenizer: str, model: str) -> None:
        self.tokenizer = tokenizer
        self.model = model or "unknown"
        message = (
            f"Tokenizer '{self.tokenizer}' incompatible with embedding model '{self.model}'"
        )
        super().__init__(message)


class ChunkingFailedError(ChunkingError):
    """Raised when the chunking pipeline fails unexpectedly."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail


class MineruOutOfMemoryError(ChunkingError):
    """Raised when MinerU exhausts GPU memory during processing."""

    def __init__(self) -> None:
        super().__init__("GPU out of memory. Retry later or reduce PDF size.")


class MineruGpuUnavailableError(ChunkingError):
    """Raised when MinerU cannot detect a healthy GPU backend."""

    def __init__(self) -> None:
        super().__init__("GPU required for PDF processing. Check GPU availability.")


__all__ = [
    "ChunkingError",
    "ChunkerConfigurationError",
    "ChunkerRegistryError",
    "InvalidDocumentError",
    "ChunkingUnavailableError",
    "ProfileNotFoundError",
    "TokenizerMismatchError",
    "ChunkingFailedError",
    "MineruOutOfMemoryError",
    "MineruGpuUnavailableError",
]
