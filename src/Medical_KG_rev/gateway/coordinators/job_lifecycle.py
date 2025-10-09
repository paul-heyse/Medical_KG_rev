"""Job lifecycle utilities shared by gateway coordinators.

This module provides the JobLifecycleManager class that encapsulates job
ledger interactions and SSE event publishing for gateway coordinators.
It provides a high-level interface for managing job state transitions
with automatic retry logic and event streaming.

Key Responsibilities:
    - Job creation and initialization with unique IDs
    - Job state transitions (processing, completed, failed, cancelled)
    - Metadata updates and tracking
    - SSE event publishing for real-time job status updates
    - Retry logic for ledger operations
    - Idempotent job creation for resilience

Architecture:
    - JobLifecycleManager coordinates between JobLedger and EventStreamManager
    - All ledger operations are wrapped with retry logic
    - Job state changes trigger corresponding SSE events
    - Pipeline information is embedded in job metadata

Thread Safety:
    - JobLifecycleManager is thread-safe for concurrent operations
    - Ledger operations are atomic and thread-safe
    - SSE event publishing is thread-safe

Performance Characteristics:
    - O(1) job creation and state updates
    - Retry logic adds exponential backoff delays
    - SSE events are published asynchronously

Example:
    >>> from Medical_KG_rev.gateway.coordinators.job_lifecycle import JobLifecycleManager
    >>> manager = JobLifecycleManager(
    ...     ledger=JobLedger(),
    ...     events=EventStreamManager()
    ... )
    >>> job_id = manager.create_job("tenant1", "embed", metadata={"model": "bert"})
    >>> manager.mark_completed(job_id, {"embeddings": 100})
"""
from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from tenacity import RetryError, Retrying, stop_after_attempt, wait_exponential

import structlog

from ...orchestration import JobLedger, JobLedgerEntry
from ..models import JobEvent
from ..sse.manager import EventStreamManager

logger = structlog.get_logger(__name__)


# ============================================================================
# JOB STATE DATA MODEL
# ============================================================================


# ============================================================================
# LIFECYCLE MANAGER
# ============================================================================


@dataclass(slots=True)
class JobLifecycleManager:
    """Manages job lifecycle operations with ledger persistence and SSE events.

    This class provides a high-level interface for managing job state transitions
    in the gateway coordinator system. It coordinates between the job ledger
    for persistence and the event stream manager for real-time updates.

    The manager handles job creation, state transitions (processing, completed,
    failed, cancelled), metadata updates, and automatic retry logic for
    ledger operations. All state changes trigger corresponding SSE events
    for real-time monitoring and client updates.

    Attributes:
        ledger: JobLedger instance for job persistence and state management.
        events: EventStreamManager instance for SSE event publishing.
        pipeline_name: Name of the pipeline processing jobs (default: "gateway-direct").
        pipeline_version: Version of the pipeline (default: "v1").
        retry_attempts: Maximum number of retry attempts for ledger operations.
        retry_wait_base: Base wait time in seconds for exponential backoff.
        retry_wait_max: Maximum wait time in seconds for exponential backoff.
        _retrying: Retrying instance for ledger operation retry logic.

    Invariants:
        - ledger is never None
        - events is never None
        - pipeline_name is never None or empty
        - pipeline_version is never None or empty
        - retry_attempts is positive
        - retry_wait_base is non-negative
        - retry_wait_max is greater than retry_wait_base
        - _retrying is initialized in __post_init__

    Thread Safety:
        - Thread-safe for concurrent operations
        - Ledger operations are atomic
        - SSE event publishing is thread-safe

    Lifecycle:
        - Created with ledger and events dependencies
        - __post_init__ initializes retry logic
        - Used by coordinators for job management
        - No explicit cleanup required

    Example:
        >>> manager = JobLifecycleManager(
        ...     ledger=JobLedger(),
        ...     events=EventStreamManager(),
        ...     pipeline_name="embedding-pipeline",
        ...     pipeline_version="v2"
        ... )
        >>> job_id = manager.create_job("tenant1", "embed", metadata={"model": "bert"})
        >>> manager.mark_completed(job_id, {"embeddings": 100})
    """

    ledger: JobLedger
    events: EventStreamManager
    pipeline_name: str = "gateway-direct"
    pipeline_version: str = "v1"
    retry_attempts: int = 3
    retry_wait_base: float = 0.1
    retry_wait_max: float = 1.0
    _retrying: Retrying = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize retry logic after dataclass construction.

        This method is called automatically after dataclass construction to
        set up the retry logic for ledger operations using exponential
        backoff with configurable parameters.

        Example:
            >>> manager = JobLifecycleManager(ledger=ledger, events=events)
            >>> # __post_init__ is called automatically
            >>> assert manager._retrying is not None
        """
        self._retrying = Retrying(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(
                multiplier=self.retry_wait_base,
                max=self.retry_wait_max,
            ),
            reraise=True,
        )

    # ------------------------------------------------------------------
    # Job lifecycle primitives
    # ------------------------------------------------------------------
    def create_job(
        self,
        tenant_id: str,
        operation: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        job_id: str | None = None,
    ) -> str:
        """Create a new job entry and emit the initial SSE event.

        This method creates a new job in the ledger with a unique ID,
        marks it as processing, and publishes an SSE event to notify
        clients of the job start. The job ID can be provided or
        auto-generated if not specified.

        Args:
            tenant_id: Unique identifier for the tenant creating the job.
            operation: The operation type being performed (e.g., "embed", "chunk").
            metadata: Optional metadata to attach to the job.
            job_id: Optional custom job ID (auto-generated if not provided).

        Returns:
            The job ID (provided or generated) for the created job.

        Raises:
            RuntimeError: If ledger operations fail after retry attempts.

        Example:
            >>> job_id = manager.create_job(
            ...     tenant_id="tenant1",
            ...     operation="embed",
            ...     metadata={"model": "bert", "texts": 5}
            ... )
            >>> print(f"Created job: {job_id}")
        """

        job_id = job_id or f"job-{uuid.uuid4().hex[:12]}"
        doc_key = f"{operation}:{job_id}"
        payload = {
            "operation": operation,
            "pipeline": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
        }
        if metadata:
            payload.update(dict(metadata))

        logger.info(
            "gateway.job.create",
            tenant_id=tenant_id,
            job_id=job_id,
            operation=operation,
        )
        self._call_ledger(
            self.ledger.create,
            job_id=job_id,
            doc_key=doc_key,
            tenant_id=tenant_id,
            pipeline=operation,
            metadata=payload,
        )
        self._call_ledger(self.ledger.mark_processing, job_id, stage=operation)
        self.events.publish(
            JobEvent(job_id=job_id, type="jobs.started", payload={"operation": operation})
        )
        return job_id

    def idempotent_create(
        self,
        *,
        job_id: str,
        doc_key: str,
        tenant_id: str,
        pipeline: str,
        metadata: Mapping[str, Any],
    ) -> JobLedgerEntry:
        """Create a job entry idempotently, returning existing entry if present.

        This method creates a job entry in the ledger if it doesn't exist,
        or returns the existing entry if it does. This provides resilience
        against duplicate job creation attempts and supports job recovery
        scenarios.

        Args:
            job_id: Unique identifier for the job.
            doc_key: Document key for the job entry.
            tenant_id: Unique identifier for the tenant.
            pipeline: Pipeline name processing the job.
            metadata: Metadata to attach to the job.

        Returns:
            JobLedgerEntry representing the created or existing job.

        Raises:
            RuntimeError: If ledger operations fail after retry attempts.

        Example:
            >>> entry = manager.idempotent_create(
            ...     job_id="job-123",
            ...     doc_key="embed:job-123",
            ...     tenant_id="tenant1",
            ...     pipeline="embedding",
            ...     metadata={"model": "bert"}
            ... )
            >>> print(f"Job status: {entry.status}")
        """
        logger.info(
            "gateway.job.idempotent_create",
            job_id=job_id,
            doc_key=doc_key,
            pipeline=pipeline,
        )
        return self._call_ledger(
            self.ledger.idempotent_create,
            job_id=job_id,
            doc_key=doc_key,
            tenant_id=tenant_id,
            pipeline=pipeline,
            metadata=dict(metadata),
        )

    def mark_completed(self, job_id: str, payload: Mapping[str, Any] | None = None) -> None:
        """Mark a job as completed and emit completion SSE event.

        This method updates the job status to completed in the ledger
        and publishes an SSE event to notify clients of the completion.
        Optional payload data can be included in the completion event.

        Args:
            job_id: Unique identifier for the job to mark as completed.
            payload: Optional payload data to include in the completion event.

        Raises:
            RuntimeError: If ledger operations fail after retry attempts.

        Example:
            >>> manager.mark_completed("job-123", {"embeddings": 100, "duration": 5.2})
            >>> # Job is marked completed and SSE event is published
        """
        metadata = dict(payload or {})
        logger.info("gateway.job.complete", job_id=job_id, metadata=metadata)
        self._call_ledger(self.ledger.mark_completed, job_id, metadata=metadata)
        self.events.publish(
            JobEvent(job_id=job_id, type="jobs.completed", payload=metadata)
        )

    def update_metadata(self, job_id: str, metadata: Mapping[str, Any]) -> None:
        """Update job metadata in the ledger.

        This method updates the metadata associated with a job in the ledger.
        This is useful for tracking progress, intermediate results, or other
        job-specific information during processing.

        Args:
            job_id: Unique identifier for the job to update.
            metadata: New metadata to associate with the job.

        Raises:
            RuntimeError: If ledger operations fail after retry attempts.

        Example:
            >>> manager.update_metadata("job-123", {"progress": 50, "processed": 25})
            >>> # Job metadata is updated in the ledger
        """
        logger.debug("gateway.job.metadata", job_id=job_id, metadata=dict(metadata))
        self._call_ledger(self.ledger.update_metadata, job_id, dict(metadata))

    def mark_failed(
        self,
        job_id: str,
        *,
        reason: str,
        stage: str = "error",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Mark a job as failed and emit failure SSE event.

        This method updates the job status to failed in the ledger
        and publishes an SSE event to notify clients of the failure.
        The failure reason and optional metadata are included in
        both the ledger entry and the SSE event.

        Args:
            job_id: Unique identifier for the job to mark as failed.
            reason: Human-readable reason for the failure.
            stage: Stage where the failure occurred (default: "error").
            metadata: Optional metadata to include with the failure.

        Raises:
            RuntimeError: If ledger operations fail after retry attempts.

        Example:
            >>> manager.mark_failed(
            ...     "job-123",
            ...     reason="Model loading failed",
            ...     stage="initialization",
            ...     metadata={"error_code": "MODEL_NOT_FOUND"}
            ... )
            >>> # Job is marked failed and SSE event is published
        """
        payload = {"reason": reason, **dict(metadata or {})}
        logger.warning(
            "gateway.job.fail",
            job_id=job_id,
            reason=reason,
            stage=stage,
            metadata=payload,
        )
        self._call_ledger(
            self.ledger.mark_failed,
            job_id,
            stage=stage,
            reason=reason,
            metadata=dict(metadata or {}),
        )
        self.events.publish(JobEvent(job_id=job_id, type="jobs.failed", payload=payload))

    def cancel(
        self,
        job_id: str,
        *,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Cancel a job and emit cancellation SSE event.

        This method marks a job as cancelled in the ledger
        and publishes an SSE event to notify clients of the cancellation.
        The cancellation reason and optional metadata are included
        in both the ledger entry and the SSE event.

        Args:
            job_id: Unique identifier for the job to cancel.
            reason: Human-readable reason for the cancellation.
            metadata: Optional metadata to include with the cancellation.

        Raises:
            RuntimeError: If ledger operations fail after retry attempts.

        Example:
            >>> manager.cancel(
            ...     "job-123",
            ...     reason="User requested cancellation",
            ...     metadata={"cancelled_by": "user"}
            ... )
            >>> # Job is marked cancelled and SSE event is published
        """
        payload = {"reason": reason, **dict(metadata or {})}
        logger.info("gateway.job.cancel", job_id=job_id, payload=payload)
        self._call_ledger(self.ledger.mark_cancelled, job_id, reason=reason)
        self.events.publish(JobEvent(job_id=job_id, type="jobs.cancelled", payload=payload))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _call_ledger(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute ledger operation with retry logic.

        This internal method wraps ledger operations with retry logic
        to handle transient failures. It uses the configured retry
        parameters to attempt the operation multiple times with
        exponential backoff.

        Args:
            func: The ledger function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            Result from the ledger operation.

        Raises:
            RuntimeError: If the operation fails after all retry attempts.

        Example:
            >>> result = manager._call_ledger(ledger.create, job_id="123", tenant_id="t1")
            >>> # Operation is retried automatically on failure
        """
        try:
            return self._retrying.call(func, *args, **kwargs)
        except RetryError as exc:
            last = exc.last_attempt
            raise RuntimeError("job ledger operation failed") from last.exception()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["JobLifecycleManager"]
