"""Helpers for keeping the job ledger in sync with orchestration state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Protocol

import structlog

from Medical_KG_rev.orchestration.ledger import (
    JobLedger,
    JobLedgerEntry,
    JobLedgerError,
)

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class StageAttempt:
    """Details about a stage attempt returned from the ledger."""

    attempt: int
    entry: JobLedgerEntry | None = None


class SupportsModelDump(Protocol):
    """Protocol for request objects used when recording metadata."""

    def model_dump(self) -> Mapping[str, Any]:
        ...


class LedgerStateManager:
    """Facade that centralises job ledger interactions for orchestration state."""

    def __init__(
        self,
        ledger: JobLedger,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._ledger = ledger
        self._clock = clock or datetime.utcnow

    # ------------------------------------------------------------------
    # Run lifecycle helpers
    # ------------------------------------------------------------------
    def record_job_attempt(self, job_id: str | None) -> int:
        """Increment the overall job attempt counter, defaulting to 1."""

        if not job_id:
            return 1
        try:
            return self._ledger.record_attempt(job_id)
        except JobLedgerError:
            logger.debug("ledger.state.job_attempt_missing", job_id=job_id)
            return 1

    def prepare_run(
        self,
        context_metadata: Mapping[str, Any],
        *,
        job_id: str | None,
        pipeline: str,
        pipeline_version: str,
        adapter_request: SupportsModelDump | Mapping[str, Any] | None,
        payload: Mapping[str, Any] | None,
    ) -> JobLedgerEntry | None:
        """Update the ledger with metadata before a Dagster run executes."""

        if not job_id:
            return None
        metadata: dict[str, Any] = {
            "pipeline_version": pipeline_version,
            "context": dict(context_metadata),
        }
        if adapter_request is not None:
            if hasattr(adapter_request, "model_dump"):
                metadata["adapter_request"] = dict(adapter_request.model_dump())
            elif isinstance(adapter_request, Mapping):
                metadata["adapter_request"] = dict(adapter_request)
        if payload is not None:
            metadata["payload"] = dict(payload)
        try:
            self._ledger.assign_pipeline(job_id, pipeline)
            entry = self._ledger.update_metadata(job_id, metadata)
            self._ledger.mark_processing(job_id, stage="bootstrap")
            return entry
        except JobLedgerError:
            logger.debug(
                "ledger.state.prepare_failed",
                job_id=job_id,
                pipeline=pipeline,
            )
            return None

    def complete_run(self, job_id: str | None) -> JobLedgerEntry | None:
        """Mark a job as completed if the ledger entry is available."""

        if not job_id:
            return None
        try:
            return self._ledger.mark_completed(job_id)
        except JobLedgerError:
            logger.debug("ledger.state.complete_missing", job_id=job_id)
            try:
                return self._ledger.get(job_id)
            except JobLedgerError:  # pragma: no cover - defensive guard
                return None

    def fetch_entry(self, job_id: str | None) -> JobLedgerEntry | None:
        if not job_id:
            return None
        return self._ledger.get(job_id)

    # ------------------------------------------------------------------
    # Stage lifecycle helpers
    # ------------------------------------------------------------------
    def stage_started(self, job_id: str | None, stage: str) -> StageAttempt:
        if not job_id:
            return StageAttempt(attempt=1)
        try:
            entry = self._ledger.mark_stage_started(job_id, stage)
        except JobLedgerError:
            logger.debug(
                "ledger.state.stage_start_missing",
                job_id=job_id,
                stage=stage,
            )
            return StageAttempt(attempt=1)
        attempt = entry.retry_count_per_stage.get(stage, 0) + 1
        return StageAttempt(attempt=attempt, entry=entry)

    def record_retry(self, job_id: str | None, stage: str) -> None:
        if not job_id:
            return
        try:
            self._ledger.increment_retry(job_id, stage)
        except JobLedgerError:
            logger.debug(
                "ledger.state.retry_missing",
                job_id=job_id,
                stage=stage,
            )

    def stage_succeeded(
        self,
        job_id: str | None,
        stage: str,
        *,
        attempts: int,
        output_count: int,
        duration_ms: int,
    ) -> None:
        if not job_id:
            return
        metadata = {
            f"stage.{stage}.attempts": attempts,
            f"stage.{stage}.output_count": output_count,
            f"stage.{stage}.duration_ms": duration_ms,
            f"stage.{stage}.completed_at": self._clock().isoformat(),
        }
        try:
            self._ledger.update_metadata(job_id, metadata)
        except JobLedgerError:
            logger.debug(
                "ledger.state.stage_success_missing",
                job_id=job_id,
                stage=stage,
            )

    def stage_failed(
        self,
        job_id: str | None,
        stage: str,
        *,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not job_id:
            return
        try:
            self._ledger.mark_failed(job_id, stage=stage, reason=reason, metadata=dict(metadata or {}))
        except JobLedgerError:
            logger.debug(
                "ledger.state.stage_failed_missing",
                job_id=job_id,
                stage=stage,
            )


__all__ = ["LedgerStateManager", "StageAttempt"]

