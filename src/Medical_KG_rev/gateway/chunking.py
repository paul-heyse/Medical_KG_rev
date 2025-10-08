"""Gateway-facing helpers for chunking interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from Medical_KG_rev.chunking.exceptions import (
    ChunkerConfigurationError,
    ChunkingFailedError,
    ChunkingUnavailableError,
    InvalidDocumentError,
    MineruGpuUnavailableError,
    MineruOutOfMemoryError,
    ProfileNotFoundError,
    TokenizerMismatchError,
)
from Medical_KG_rev.observability.metrics import record_chunking_failure
from Medical_KG_rev.services.retrieval.chunking_command import ChunkCommand

from .models import ProblemDetail

ChunkingSeverity = Literal["client", "server", "transient"]


@dataclass(slots=True)
class TranslatedChunkingError:
    detail: ProblemDetail
    failure_type: str
    severity: ChunkingSeverity


class ChunkingErrorTranslator:
    """Maps domain errors to protocol friendly responses and observability hooks."""

    def __init__(
        self,
        *,
        available_strategies: Callable[[], Sequence[str]] | None = None,
        failure_recorder: Callable[[str | None, str], None] = record_chunking_failure,
    ) -> None:
        self._available_strategies = available_strategies
        self._failure_recorder = failure_recorder

    def translate(
        self,
        exc: Exception,
        *,
        command: ChunkCommand | None,
    ) -> TranslatedChunkingError:
        profile = None
        correlation_id = None
        document_id = None
        if command is not None:
            metadata_profile = command.metadata.get("profile") if command.metadata else None
            profile = command.profile or metadata_profile
            correlation_id = command.correlation_id
            document_id = command.document_id
        failure_type = exc.__class__.__name__
        self._failure_recorder(profile or "unknown", failure_type)

        if isinstance(exc, ProfileNotFoundError):
            extensions = {
                "available_profiles": list(getattr(exc, "available", ())),
            }
            if correlation_id:
                extensions["correlation_id"] = correlation_id
            detail = ProblemDetail(
                title="Chunking profile not found",
                status=400,
                type="https://medical-kg/errors/chunking-profile-not-found",
                detail=str(exc),
                extensions=extensions,
            )
            return TranslatedChunkingError(detail, failure_type, "client")
        if isinstance(exc, TokenizerMismatchError):
            extensions = {}
            if correlation_id:
                extensions["correlation_id"] = correlation_id
            detail = ProblemDetail(
                title="Tokenizer mismatch",
                status=500,
                type="https://medical-kg/errors/tokenizer-mismatch",
                detail=str(exc),
                extensions=extensions,
            )
            return TranslatedChunkingError(detail, failure_type, "server")
        if isinstance(exc, ChunkingFailedError):
            message = exc.detail or str(exc) or "Chunking process failed"
            extensions = {}
            if correlation_id:
                extensions["correlation_id"] = correlation_id
            detail = ProblemDetail(
                title="Chunking failed",
                status=500,
                type="https://medical-kg/errors/chunking-failed",
                detail=message,
                extensions=extensions,
            )
            return TranslatedChunkingError(detail, failure_type, "server")
        if isinstance(exc, InvalidDocumentError):
            extensions = {}
            if correlation_id:
                extensions["correlation_id"] = correlation_id
            instance = f"/v1/chunk/{document_id}" if document_id else None
            detail = ProblemDetail(
                title="Invalid document payload",
                status=400,
                type="https://httpstatuses.com/400",
                detail=str(exc),
                instance=instance,
                extensions=extensions,
            )
            return TranslatedChunkingError(detail, failure_type, "client")
        if isinstance(exc, ChunkerConfigurationError):
            strategies = list(self._available_strategies() or []) if self._available_strategies else []
            extensions = {"valid_strategies": strategies}
            if correlation_id:
                extensions["correlation_id"] = correlation_id
            detail = ProblemDetail(
                title="Chunker configuration invalid",
                status=422,
                type="https://httpstatuses.com/422",
                detail=str(exc),
                extensions=extensions,
            )
            return TranslatedChunkingError(detail, failure_type, "client")
        if isinstance(exc, ChunkingUnavailableError):
            retry_after = max(1, int(round(exc.retry_after)))
            extensions = {"retry_after": retry_after}
            if correlation_id:
                extensions["correlation_id"] = correlation_id
            detail = ProblemDetail(
                title="Chunking temporarily unavailable",
                status=503,
                type="https://httpstatuses.com/503",
                detail=str(exc),
                extensions=extensions,
            )
            return TranslatedChunkingError(detail, failure_type, "transient")
        if isinstance(exc, MineruOutOfMemoryError):
            extensions = {"reason": "gpu_out_of_memory"}
            if correlation_id:
                extensions["correlation_id"] = correlation_id
            detail = ProblemDetail(
                title="MinerU out of memory",
                status=503,
                type="https://medical-kg/errors/mineru-oom",
                detail=str(exc),
                extensions=extensions,
            )
            return TranslatedChunkingError(detail, failure_type, "transient")
        if isinstance(exc, MineruGpuUnavailableError):
            extensions = {"reason": "gpu_unavailable"}
            if correlation_id:
                extensions["correlation_id"] = correlation_id
            detail = ProblemDetail(
                title="MinerU GPU unavailable",
                status=503,
                type="https://medical-kg/errors/mineru-gpu-unavailable",
                detail=str(exc),
                extensions=extensions,
            )
            return TranslatedChunkingError(detail, failure_type, "transient")
        if isinstance(exc, MemoryError):
            extensions = {"retry_after": 60}
            if correlation_id:
                extensions["correlation_id"] = correlation_id
            detail = ProblemDetail(
                title="Chunking resources exhausted",
                status=503,
                type="https://httpstatuses.com/503",
                detail=str(exc) or "Chunking operation exhausted available memory",
                extensions=extensions,
            )
            return TranslatedChunkingError(detail, failure_type, "transient")
        if isinstance(exc, TimeoutError):
            extensions = {"retry_after": 30}
            if correlation_id:
                extensions["correlation_id"] = correlation_id
            detail = ProblemDetail(
                title="Chunking resources exhausted",
                status=503,
                type="https://httpstatuses.com/503",
                detail=str(exc) or "Chunking operation timed out",
                extensions=extensions,
            )
            return TranslatedChunkingError(detail, failure_type, "transient")
        if isinstance(exc, RuntimeError) and "GPU semantic checks" in str(exc):
            extensions = {"reason": "gpu_unavailable"}
            if correlation_id:
                extensions["correlation_id"] = correlation_id
            detail = ProblemDetail(
                title="GPU unavailable for semantic chunking",
                status=503,
                type="https://httpstatuses.com/503",
                detail=str(exc),
                extensions=extensions,
            )
            return TranslatedChunkingError(detail, failure_type, "transient")
        raise exc


__all__ = ["ChunkingErrorTranslator", "TranslatedChunkingError", "ChunkingSeverity"]

