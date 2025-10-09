"""Job lifecycle utilities shared by gateway coordinators."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

import structlog
from tenacity import RetryError, Retrying, stop_after_attempt, wait_exponential

from ...orchestration import JobLedger, JobLedgerEntry
from ..models import JobEvent
from ..sse.manager import EventStreamManager

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class JobLifecycleManager:
    """Encapsulates job ledger interactions and SSE event publishing."""

    ledger: JobLedger
    events: EventStreamManager
    pipeline_name: str = "gateway-direct"
    pipeline_version: str = "v1"
    retry_attempts: int = 3
    retry_wait_base: float = 0.1
    retry_wait_max: float = 1.0
    _retrying: Retrying = field(init=False, repr=False)

    def __post_init__(self) -> None:
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
        """Create a job entry and emit the initial SSE event."""

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
        metadata = dict(payload or {})
        logger.info("gateway.job.complete", job_id=job_id, metadata=metadata)
        self._call_ledger(self.ledger.mark_completed, job_id, metadata=metadata)
        self.events.publish(JobEvent(job_id=job_id, type="jobs.completed", payload=metadata))

    def update_metadata(self, job_id: str, metadata: Mapping[str, Any]) -> None:
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
        payload = {"reason": reason, **dict(metadata or {})}
        logger.info("gateway.job.cancel", job_id=job_id, payload=payload)
        self._call_ledger(self.ledger.mark_cancelled, job_id, reason=reason)
        self.events.publish(JobEvent(job_id=job_id, type="jobs.cancelled", payload=payload))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _call_ledger(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._retrying.call(func, *args, **kwargs)
        except RetryError as exc:
            last = exc.last_attempt
            raise RuntimeError("job ledger operation failed") from last.exception()


__all__ = ["JobLifecycleManager"]
