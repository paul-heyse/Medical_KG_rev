"""Utilities for translating chunking errors into API-facing payloads.

This module provides error translation services for the chunking pipeline,
converting domain-specific chunking exceptions into standardized ProblemDetail
payloads suitable for API responses. It handles various error types including
configuration errors, resource unavailability, and processing failures.

The module defines:
- ChunkingErrorReport: Structured representation of translated errors
- ChunkingErrorTranslator: Main translation service for chunking errors

Architecture:
- Error translation follows a pattern-based approach
- Each exception type maps to specific HTTP status codes and error categories
- Extensions provide additional context for error handling
- Severity levels help with monitoring and alerting

Thread Safety:
- Translator instances are thread-safe and stateless
- Error reports are immutable data structures

Performance:
- Translation is lightweight and fast
- No external dependencies or I/O operations
- Minimal memory allocation

Examples
--------
    translator = ChunkingErrorTranslator(strategies=["semantic", "fixed"])
    report = translator.translate(exception, command=chunk_command, job_id="job-123")
    if report:
        return report.problem

"""

from __future__ import annotations

# IMPORTS
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from Medical_KG_rev.chunking.exceptions import HttpClient
from Medical_KG_rev.gateway.models import ProblemDetail
from Medical_KG_rev.services.retrieval.chunking import ChunkCommand



# DATA MODELS
@dataclass(slots=True)
class ChunkingErrorReport:
    """Structured view of a translated chunking error.

    This dataclass represents the result of error translation, providing
    a structured view of chunking errors suitable for API responses and
    monitoring systems.

    Attributes
    ----------
        problem: Standardized problem detail for API response
        severity: Error severity level (client, fatal, retryable)
        metric: Optional metric name for monitoring
        job_id: Optional job identifier for correlation

    Thread Safety:
        Immutable dataclass, thread-safe.

    Examples
    --------
        report = ChunkingErrorReport(
            problem=problem_detail,
            severity="client",
            metric="ProfileNotFoundError",
            job_id="job-123"
        )

    """

    problem: ProblemDetail
    severity: str
    metric: str | None = None
    job_id: str | None = None


# ERROR TRANSLATION SERVICE
class ChunkingErrorTranslator:
    """Map domain-specific chunking failures to `ProblemDetail` payloads.

    This class provides comprehensive error translation for chunking operations,
    converting various exception types into standardized ProblemDetail objects
    suitable for API responses. It handles configuration errors, resource
    unavailability, processing failures, and system errors.

    Attributes
    ----------
        _strategies: Available chunking strategies for validation
        _base_path: Base path for error instance URLs

    Thread Safety:
        Thread-safe and stateless. Safe for concurrent use.

    Performance:
        Lightweight translation with minimal overhead.
        No external dependencies or I/O operations.

    Examples
    --------
        translator = ChunkingErrorTranslator(
            strategies=["semantic", "fixed"],
            base_path="/v1/chunk"
        )
        report = translator.translate(exception, command=chunk_command)

    """

    def __init__(self, *, strategies: Sequence[str], base_path: str = "/v1/chunk") -> None:
        """Initialize the chunking error translator.

        Args:
        ----
            strategies: Available chunking strategies for validation
            base_path: Base path for error instance URLs

        Raises:
        ------
            None: Initialization always succeeds.

        """
        self._strategies = tuple(strategies)
        self._base_path = base_path.rstrip("/")

    def translate(
        self,
        exc: Exception,
        *,
        command: ChunkCommand,
        job_id: str | None = None,
    ) -> ChunkingErrorReport | None:
        """Translate a chunking exception into a structured error report.

        Analyzes the exception type and converts it into a ChunkingErrorReport
        with appropriate HTTP status codes, severity levels, and error details.
        Returns None for unrecognized exception types.

        Args:
        ----
            exc: The exception to translate
            command: Chunk command providing context
            job_id: Optional job identifier for correlation

        Returns:
        -------
            Structured error report or None if exception is not recognized

        Raises:
        ------
            None: This method never raises exceptions.

        """
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
        """Extract a ProblemDetail from a context mapping.

        Args:
        ----
            context: Context mapping potentially containing a problem

        Returns:
        -------
            ProblemDetail if found, None otherwise

        Raises:
        ------
            None: This method never raises exceptions.

        """
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
        """Create a ProblemDetail object with the given parameters.

        Args:
        ----
            title: Error title
            status: HTTP status code
            detail: Error detail message
            instance: Error instance URI
            type_: Optional error type URI
            extensions: Optional additional error context

        Returns:
        -------
            Validated ProblemDetail object

        Raises:
        ------
            ValidationError: If the payload is invalid

        """
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
        """Extract the profile name from a chunk command.

        Args:
        ----
            command: Chunk command to extract profile from

        Returns:
        -------
            Profile name if present and valid, None otherwise

        Raises:
        ------
            None: This method never raises exceptions.

        """
        profile = command.options.get("profile")
        if isinstance(profile, str) and profile:
            return profile
        return None


# EXPORTS
__all__ = ["ChunkingErrorReport", "ChunkingErrorTranslator"]
