"""Utilities for translating chunking errors into API-facing payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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
from Medical_KG_rev.gateway.models import ProblemDetail
from Medical_KG_rev.services.retrieval.chunking import ChunkCommand


@dataclass(slots=True)
class ChunkingErrorReport:
    """Structured view of a translated chunking error."""

    problem: ProblemDetail
    severity: str
    metric: str | None = None
    job_id: str | None = None


class ChunkingErrorTranslator:
    """Map domain-specific chunking failures to `ProblemDetail` payloads."""

    def __init__(self, *, strategies: Sequence[str], base_path: str = "/v1/chunk") -> None:
        self._strategies = tuple(strategies)
        self._base_path = base_path.rstrip("/")

    def translate(
        self,
        exc: Exception,
        *,
        command: ChunkCommand,
        job_id: str | None = None,
    ) -> ChunkingErrorReport | None:
        profile = self._profile(command)
        instance = f"{self._base_path}/{command.document_id}"

        if isinstance(exc, ProfileNotFoundError):
            detail = self._problem(
                title="Chunking profile not found",
                status=400,
                detail=str(exc) or "Requested chunking profile is unavailable",
                instance=instance,
                extensions={"profile": profile} if profile else None,
            )
            return ChunkingErrorReport(detail, "client", "ProfileNotFoundError", job_id)

        if isinstance(exc, TokenizerMismatchError):
            detail = self._problem(
                title="Tokenizer mismatch",
                status=500,
                detail=str(exc) or "Tokenizer mismatch between profile and text",
                instance=instance,
            )
            return ChunkingErrorReport(detail, "fatal", "TokenizerMismatchError", job_id)

        if isinstance(exc, ChunkingFailedError):
            message = exc.detail or str(exc) or "Chunking process failed"
            detail = self._problem(
                title="Chunking failed",
                status=500,
                detail=message,
                instance=instance,
            )
            return ChunkingErrorReport(detail, "fatal", "ChunkingFailedError", job_id)

        if isinstance(exc, InvalidDocumentError):
            detail = self._problem(
                title="Invalid document payload",
                status=400,
                detail=str(exc) or "Chunking requests must include text",
                instance=instance,
            )
            return ChunkingErrorReport(detail, "client", "InvalidDocumentError", job_id)

        if isinstance(exc, ChunkerConfigurationError):
            detail = self._problem(
                title="Chunker configuration invalid",
                status=422,
                detail=str(exc) or "Chunking configuration is invalid",
                instance=instance,
                extensions={"valid_strategies": list(self._strategies)},
            )
            return ChunkingErrorReport(detail, "client", "ChunkerConfigurationError", job_id)

        if isinstance(exc, ChunkingUnavailableError):
            retry_after = max(1, int(round(exc.retry_after)))
            detail = self._problem(
                title="Chunking temporarily unavailable",
                status=503,
                detail=str(exc) or "Chunking temporarily unavailable",
                instance=instance,
                extensions={"retry_after": retry_after},
            )
            return ChunkingErrorReport(detail, "retryable", "ChunkingUnavailableError", job_id)

        if isinstance(exc, MineruOutOfMemoryError):
            detail = self._problem(
                title="MinerU out of memory",
                status=503,
                detail=str(exc) or "MinerU exhausted GPU memory",
                instance=instance,
                type_="https://medical-kg/errors/mineru-oom",
                extensions={"reason": "gpu_out_of_memory"},
            )
            return ChunkingErrorReport(detail, "retryable", "MineruOutOfMemoryError", job_id)

        if isinstance(exc, MineruGpuUnavailableError):
            detail = self._problem(
                title="MinerU GPU unavailable",
                status=503,
                detail=str(exc) or "MinerU GPU unavailable",
                instance=instance,
                type_="https://medical-kg/errors/mineru-gpu-unavailable",
                extensions={"reason": "gpu_unavailable"},
            )
            return ChunkingErrorReport(detail, "retryable", "MineruGpuUnavailableError", job_id)

        if isinstance(exc, MemoryError):
            detail = self._problem(
                title="Chunking resources exhausted",
                status=503,
                detail=str(exc) or "Chunking operation exhausted available memory",
                instance=instance,
                extensions={"retry_after": 60},
            )
            return ChunkingErrorReport(detail, "retryable", "MemoryError", job_id)

        if isinstance(exc, TimeoutError):
            detail = self._problem(
                title="Chunking operation timed out",
                status=503,
                detail=str(exc) or "Chunking operation timed out",
                instance=instance,
                extensions={"retry_after": 30},
            )
            return ChunkingErrorReport(detail, "retryable", "TimeoutError", job_id)

        if isinstance(exc, RuntimeError) and "GPU semantic checks" in str(exc):
            detail = self._problem(
                title="GPU unavailable for semantic chunking",
                status=503,
                detail=str(exc),
                instance=instance,
                extensions={"reason": "gpu_unavailable"},
            )
            return ChunkingErrorReport(detail, "retryable", "GpuUnavailable", job_id)

        return None

    def from_context(self, context: Mapping[str, Any]) -> ProblemDetail | None:
        problem = context.get("problem")
        if isinstance(problem, ProblemDetail):
            return problem
        return None

    def _problem(
        self,
        *,
        title: str,
        status: int,
        detail: str,
        instance: str,
        type_: str | None = None,
        extensions: Mapping[str, Any] | None = None,
    ) -> ProblemDetail:
        payload = {
            "type": type_ or "https://medical-kg/errors/chunking",
            "title": title,
            "status": status,
            "detail": detail,
            "instance": instance,
            "extensions": dict(extensions or {}),
        }
        return ProblemDetail.model_validate(payload)

    @staticmethod
    def _profile(command: ChunkCommand) -> str | None:
        profile = command.options.get("profile")
        if isinstance(profile, str) and profile:
            return profile
        return None


__all__ = ["ChunkingErrorReport", "ChunkingErrorTranslator"]
