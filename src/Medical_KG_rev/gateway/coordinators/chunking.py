"""Chunking coordinator for synchronous document chunking operations.

This module provides the ChunkingCoordinator class that coordinates synchronous
chunking operations by managing job lifecycle, delegating to ChunkingService,
and translating errors to coordinator-friendly responses.

Key Responsibilities:
    - Job lifecycle management (create, track, complete/fail jobs)
    - Request validation and text extraction
    - Error translation from chunking exceptions to coordinator errors
    - Metrics emission for chunking operations
    - Integration with ChunkingErrorTranslator for consistent error handling

Collaborators:
    - Upstream: Gateway service layer (calls execute method)
    - Downstream: ChunkingService (performs actual chunking), JobLifecycleManager (tracks jobs), ChunkingErrorTranslator (translates errors)

Side Effects:
    - Creates job entries in job lifecycle manager
    - Emits metrics via record_chunking_failure
    - Logs errors and operations
    - Updates job status (completed/failed)

Thread Safety:
    - Not thread-safe: Designed for single-threaded use per coordinator instance
    - Multiple coordinator instances can run concurrently

Performance Characteristics:
    - O(n) time complexity where n is document length
    - Memory usage scales with chunk count and size
    - Synchronous operation blocks until chunking completes

Example:
    >>> from Medical_KG_rev.gateway.coordinators import ChunkingCoordinator
    >>> coordinator = ChunkingCoordinator(
    ...     lifecycle=JobLifecycleManager(),
    ...     chunker=ChunkingService(),
    ...     config=CoordinatorConfig(name="chunking")
    ... )
    >>> result = coordinator.execute(ChunkingRequest(
    ...     document_id="doc1",
    ...     text="Sample document text for chunking.",
    ...     strategy="section"
    ... ))
    >>> print(f"Processed {len(result.chunks)} chunks")
"""
from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from Medical_KG_rev.chunking.exceptions import InvalidDocumentError
from Medical_KG_rev.gateway.models import DocumentChunk
from Medical_KG_rev.observability.metrics import record_chunking_failure
from Medical_KG_rev.services.retrieval.chunking import (
    ChunkCommand,
    ChunkingService,
)

from ..chunking_errors import ChunkingErrorReport, ChunkingErrorTranslator
from .base import (
    BaseCoordinator,
    CoordinatorConfig,
    CoordinatorError,
    CoordinatorRequest,
    CoordinatorResult,
)
from .job_lifecycle import JobLifecycleManager

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class ChunkingRequest(CoordinatorRequest):
    """Request for synchronous document chunking operations.

    Attributes:
        document_id: Unique identifier for the document being chunked.
        text: Optional document text content. If None, text will be extracted
              from options["text"] or retrieved from document storage.
        strategy: Chunking strategy name (e.g., "section", "semantic").
                  Defaults to profile setting if not specified.
        chunk_size: Maximum number of tokens per chunk. Defaults to profile
                    setting if not specified.
        overlap: Number of tokens to overlap between consecutive chunks.
                 Defaults to profile setting if not specified.
        options: Additional metadata and configuration options for chunking.
                 May contain text content, profile overrides, or custom settings.
    """

    def __init__(
        self,
        tenant_id: str,
        document_id: str,
        *,
        text: str | None = None,
        strategy: str | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        options: Mapping[str, Any] | None = None,
        correlation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize chunking request.

        Args:
            tenant_id: Tenant identifier for multi-tenancy.
            document_id: Unique identifier for the document being chunked.
            text: Optional document text content.
            strategy: Chunking strategy name.
            chunk_size: Maximum number of tokens per chunk.
            overlap: Number of tokens to overlap between consecutive chunks.
            options: Additional metadata and configuration options.
            correlation_id: Optional correlation ID for request tracking.
            metadata: Optional metadata for request context.
        """
        super().__init__(
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            metadata=metadata,
        )
        self.document_id = document_id
        self.text = text
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.options = options


@dataclass
class ChunkingResult(CoordinatorResult):
    """Result of synchronous document chunking operations.

    Attributes:
        chunks: Sequence of DocumentChunk objects containing the chunked content
                and associated metadata. Each chunk includes text content,
                position information, and chunk-specific metadata.
    """
    chunks: Sequence[DocumentChunk] = ()


# ============================================================================
# COORDINATOR IMPLEMENTATION
# ============================================================================


class ChunkingCoordinator(BaseCoordinator[ChunkingRequest, ChunkingResult]):
    """Coordinates synchronous document chunking operations.

    This class implements the coordinator pattern for document chunking, managing
    the complete lifecycle of chunking jobs from request validation through
    error handling and result assembly.

    The coordinator coordinates between the gateway service layer and the domain
    chunking service, providing a clean abstraction for synchronous chunking
    operations with comprehensive error handling and metrics.

    Attributes:
        _lifecycle: JobLifecycleManager for tracking job state and metadata.
        _chunker: ChunkingService for performing actual document chunking.
        _errors: ChunkingErrorTranslator for translating chunking exceptions
                 to coordinator-friendly errors.

    Invariants:
        - self._lifecycle is never None after __init__
        - self._chunker is never None after __init__
        - self._errors is never None after __init__
        - All public methods maintain job lifecycle consistency

    Thread Safety:
        - Not thread-safe: Designed for single-threaded use per coordinator instance
        - Multiple coordinator instances can run concurrently

    Lifecycle:
        - Created with injected dependencies (lifecycle, chunker, config, errors)
        - Used for processing chunking requests via execute method
        - No explicit cleanup required (stateless operations)

    Example:
        >>> coordinator = ChunkingCoordinator(
        ...     lifecycle=JobLifecycleManager(),
        ...     chunker=ChunkingService(),
        ...     config=CoordinatorConfig(name="chunking"),
        ...     errors=ChunkingErrorTranslator()
        ... )
        >>> result = coordinator.execute(ChunkingRequest(
        ...     document_id="doc1",
        ...     text="Sample document text for chunking.",
        ...     strategy="section"
        ... ))
        >>> print(f"Processed {len(result.chunks)} chunks")
    """

    def __init__(
        self,
        lifecycle: JobLifecycleManager,
        chunker: ChunkingService,
        config: CoordinatorConfig,
        *,
        errors: ChunkingErrorTranslator | None = None,
    ) -> None:
        """Initialize the chunking coordinator.

        Args:
            lifecycle: JobLifecycleManager for tracking job state and metadata.
                       Must be initialized and ready to manage jobs.
            chunker: ChunkingService for performing actual document chunking.
                     Must be configured with available chunking strategies.
            config: CoordinatorConfig with coordinator name and settings.
                    Used for metrics and configuration.
            errors: Optional ChunkingErrorTranslator for error translation.
                   If None, a new translator will be created with available
                   strategies from the chunker.

        Raises:
            ValueError: If any required dependency is None or invalid.
            ConfigurationError: If coordinator configuration is invalid.
        """
        super().__init__(config=config, metrics=self._metrics(config))
        self._lifecycle = lifecycle
        self._chunker = chunker
        strategies = chunker.available_strategies()
        self._errors = errors or ChunkingErrorTranslator(strategies=strategies)

    @staticmethod
    def _metrics(config: CoordinatorConfig) -> Any:
        from .base import CoordinatorMetrics

        return CoordinatorMetrics.create(config.name)

    def _execute(self, request: ChunkingRequest, /, **_: Any) -> ChunkingResult:
        """Execute synchronous document chunking operation.

        This method coordinates the complete chunking workflow:
        1. Create job entry for tracking
        2. Extract document text from request
        3. Create ChunkCommand with chunking parameters
        4. Delegate to ChunkingService for actual chunking
        5. Handle any exceptions via error translator
        6. Assemble chunks into result
        7. Mark job as completed
        8. Return chunking result

        Args:
            request: ChunkingRequest containing document ID, text, strategy,
                    chunk size, overlap, and additional options.
            **_: Additional keyword arguments (ignored for compatibility).

        Returns:
            ChunkingResult containing sequence of DocumentChunk objects
            with chunked content and metadata.

        Raises:
            CoordinatorError: For all handled errors after translation from
                             chunking exceptions. Includes appropriate HTTP
                             status codes and problem details.

        Note:
            Emits metrics for chunking failures via record_chunking_failure.
            Updates job lifecycle state throughout the operation.
            All exceptions are translated to CoordinatorError for consistent
            error handling across the gateway layer.

        Example:
            >>> result = coordinator._execute(ChunkingRequest(
            ...     document_id="doc1",
            ...     text="Sample document text for chunking.",
            ...     strategy="section"
            ... ))
            >>> assert len(result.chunks) > 0
        """
        job_id = self._lifecycle.create_job(request.tenant_id, "chunk")
        text = self._extract_text(job_id, request)
        command = ChunkCommand(
            tenant_id=request.tenant_id,
            document_id=request.document_id,
            text=text,
            strategy=request.strategy or "section",
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            metadata=dict(request.options or {}),
        )
        started = time.perf_counter()
        try:
            raw_chunks = self._chunker.chunk(command)
        except Exception as exc:
            raise self._translate_error(job_id, command, exc) from exc

        chunks: list[DocumentChunk] = []
        for index, chunk in enumerate(raw_chunks):
            meta = dict(chunk.meta)
            meta.setdefault("granularity", chunk.granularity)
            meta.setdefault("chunker", chunk.chunker)
            chunks.append(
                DocumentChunk(
                    document_id=request.document_id,
                    chunk_index=index,
                    content=chunk.body,
                    metadata=meta,
                    token_count=meta.get("token_count", 0),
                )
            )
        duration = time.perf_counter() - started
        payload = {"chunks": len(chunks), "strategy": command.strategy}
        payload = {"chunks": len(chunks)}
        self._lifecycle.update_metadata(job_id, payload)
        self._lifecycle.mark_completed(job_id, payload=payload)
        return ChunkingResult(
            job_id=job_id,
            duration_s=duration,
            chunks=tuple(chunks),
            metadata=payload,
        )

    def _extract_text(self, job_id: str, request: ChunkingRequest) -> str:
        """Extract document text from chunking request.

        This method extracts document text from the request, checking multiple
        sources in order of preference:
        1. request.text (primary source)
        2. request.options["text"] (fallback source)

        Args:
            job_id: Job identifier for error reporting and logging.
            request: ChunkingRequest containing text in text field or options.

        Returns:
            str: Non-empty document text ready for chunking.

        Raises:
            InvalidDocumentError: If no valid text is found in either source.
                                 Error message indicates which fields were checked.

        Note:
            Text can be provided in request.text or request.options["text"].
            Check request.text first for backwards compatibility, then fall back to options.
            All text is stripped of whitespace before validation.

        Example:
            >>> text = coordinator._extract_text("job123", ChunkingRequest(
            ...     document_id="doc1",
            ...     text="Sample document text for chunking."
            ... ))
            >>> assert len(text) > 0
        """
        candidate = request.text
        if isinstance(candidate, str) and candidate.strip():
            return candidate
        payload = request.options or {}
        raw_text = payload.get("text") if isinstance(payload, Mapping) else None
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise InvalidDocumentError(
                "Chunking requests must include a non-empty 'text' field"
            )
        return raw_text

    def _translate_error(
        self,
        job_id: str,
        command: ChunkCommand,
        exc: Exception,
    ) -> CoordinatorError:
        """Translate chunking exceptions to coordinator errors.

        This method uses the ChunkingErrorTranslator to convert chunking-specific
        exceptions into CoordinatorError instances with appropriate HTTP status
        codes, problem details, and retry strategies.

        Args:
            job_id: Job identifier for error reporting and lifecycle updates.
            command: ChunkCommand providing context for error translation.
                    Used to extract profile information and chunking parameters.
            exc: Exception raised during chunking operation.
                 Can be any chunking-specific exception or generic Exception.

        Returns:
            CoordinatorError: Translated error with HTTP status code, problem
                             details, and appropriate error context.

        Raises:
            Exception: Re-raises the original exception if translation fails
                      or returns None. This ensures unhandled exceptions
                      propagate correctly.

        Note:
            Calls _record_failure internally to update metrics and job lifecycle.
            If translation returns None, marks job as failed and re-raises
            the original exception.

        Example:
            >>> try:
            ...     chunks = self._chunker.chunk(command)
            ... except Exception as exc:
            ...     error = coordinator._translate_error("job123", command, exc)
            ...     raise error
        """
        report = self._errors.translate(exc, command=command, job_id=job_id)
        if report is None:
            self._lifecycle.mark_failed(
                job_id,
                reason=str(exc) or exc.__class__.__name__,
                stage="chunk",
            )
            raise exc
        self._record_failure(job_id, command, report)
        return CoordinatorError(
            report.problem.title,
            context={
                "problem": report.problem,
                "job_id": job_id,
                "severity": report.severity,
                "metric": report.metric,
            },
        )


# ============================================================================
# ERROR TRANSLATION
# ============================================================================


    @staticmethod
    def _metadata_without_text(options: Mapping[str, Any] | None) -> Mapping[str, Any]:
        """Extract metadata from options excluding text content.

        This helper method filters out the "text" key from options to create
        metadata that can be safely passed to chunking operations without
        duplicating text content.

        Args:
            options: Optional mapping containing various options including text.
                    If None or not a mapping, returns empty dict.

        Returns:
            Mapping[str, Any]: Filtered options with "text" key removed.
                              Returns empty dict if options is None or invalid.

        Example:
            >>> metadata = ChunkingCoordinator._metadata_without_text({
            ...     "text": "document content",
            ...     "profile": "section",
            ...     "custom": "value"
            ... })
            >>> assert "text" not in metadata
            >>> assert metadata["profile"] == "section"
        """
        if not isinstance(options, Mapping):
            return {}
        return {key: value for key, value in options.items() if key != "text"}

    def _record_failure(
        self,
        job_id: str,
        command: ChunkCommand,
        report: ChunkingErrorReport,
    ) -> None:
        """Record chunking failure by emitting metrics and updating job lifecycle.

        This method handles the side effects of chunking failures by:
        1. Extracting profile information from command options
        2. Emitting failure metrics via record_chunking_failure
        3. Updating job lifecycle state to failed

        Args:
            job_id: Job identifier for lifecycle updates and metric context.
            command: ChunkCommand containing profile information in options.
                    Used to extract profile name for metric labeling.
            report: ChunkingErrorReport containing error details and metric name.
                   Provides the specific error metric to record.

        Returns:
            None: This method performs side effects only.

        Note:
            Side effects include metric emission and lifecycle update.
            If profile is not found in command options, uses None as profile.
            Metric name defaults to "unknown_error" if not specified in report.

        Example:
            >>> coordinator._record_failure(
            ...     "job123",
            ...     ChunkCommand(tenant_id="tenant", document_id="doc1", text="text", strategy="section"),
            ...     ChunkingErrorReport(problem=ProblemDetail(...), metric="ProfileNotFoundError")
            ... )
        """
        profile = command.metadata.get("profile")
        if isinstance(profile, str) and profile:
            record_chunking_failure(profile, report.metric or "unknown_error")
        else:
            record_chunking_failure(None, report.metric or "unknown_error")
        self._lifecycle.mark_failed(
            job_id,
            reason=report.problem.detail or report.problem.title,
            stage="chunk",
        )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "ChunkingCoordinator",
    "ChunkingRequest",
    "ChunkingResult",
]
