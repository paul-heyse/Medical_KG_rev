"""Job ledger tracking orchestration state."""

from __future__ import annotations

import builtins
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
ALLOWED_TRANSITIONS = {
    "queued": {"processing", "cancelled"},
    "processing": {"processing", "completed", "failed", "cancelled"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}


_UNSET: Any = object()


@dataclass
class JobTransition:
    from_status: str
    to_status: str
    stage: str
    reason: str | None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PdfHistoryEvent:
    timestamp: datetime
    stage: str
    status: str
    detail: str | None = None


@dataclass
class JobLedgerEntry:
    job_id: str
    doc_key: str
    tenant_id: str
    status: str = "queued"
    stage: str = "pending"
    current_stage: str = "pending"
    pipeline: str | None = None
    pipeline_name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    attempts: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    history: list[JobTransition] = field(default_factory=list)
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    error_reason: str | None = None
    retry_count: int = 0
    retry_count_per_stage: dict[str, int] = field(default_factory=dict)
    pdf_downloaded: bool = False
    pdf_ir_ready: bool = False
    pdf_url: str | None = None
    pdf_storage_key: str | None = None
    pdf_size: int | None = None
    pdf_content_type: str | None = None
    pdf_checksum: str | None = None
    pdf_downloaded_at: datetime | None = None
    pdf_ir_ready_at: datetime | None = None
    pdf_error: str | None = None
    pdf_failure_code: str | None = None
    pdf_history: list[PdfHistoryEvent] = field(default_factory=list)

    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def snapshot(self) -> JobLedgerEntry:
        """Return a copy suitable for external consumption."""

        return JobLedgerEntry(
            job_id=self.job_id,
            doc_key=self.doc_key,
            tenant_id=self.tenant_id,
            status=self.status,
            stage=self.stage,
            current_stage=self.current_stage,
            pipeline=self.pipeline,
            pipeline_name=self.pipeline_name,
            metadata=dict(self.metadata),
            attempts=self.attempts,
            created_at=self.created_at,
            updated_at=self.updated_at,
            history=list(self.history),
            completed_at=self.completed_at,
            duration_seconds=self.duration_seconds,
            error_reason=self.error_reason,
            retry_count=self.retry_count,
            retry_count_per_stage=dict(self.retry_count_per_stage),
            pdf_downloaded=self.pdf_downloaded,
            pdf_ir_ready=self.pdf_ir_ready,
            pdf_url=self.pdf_url,
            pdf_storage_key=self.pdf_storage_key,
            pdf_size=self.pdf_size,
            pdf_content_type=self.pdf_content_type,
            pdf_checksum=self.pdf_checksum,
            pdf_downloaded_at=self.pdf_downloaded_at,
            pdf_ir_ready_at=self.pdf_ir_ready_at,
            pdf_error=self.pdf_error,
            pdf_failure_code=self.pdf_failure_code,
            pdf_history=list(self.pdf_history),
        )


class JobLedgerError(RuntimeError):
    pass


class JobLedger:
    """In-memory ledger implementation with idempotency helpers."""

    def __init__(self) -> None:
        self._entries: dict[str, JobLedgerEntry] = {}
        self._doc_index: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Creation & idempotency
    # ------------------------------------------------------------------
    def create(
        self,
        *,
        job_id: str,
        doc_key: str,
        tenant_id: str,
        pipeline: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> JobLedgerEntry:
        if job_id in self._entries:
            raise JobLedgerError(f"Job {job_id} already exists")
        if doc_key in self._doc_index:
            raise JobLedgerError(f"Document key {doc_key} already registered")
        entry = JobLedgerEntry(
            job_id=job_id,
            doc_key=doc_key,
            tenant_id=tenant_id,
            pipeline=pipeline,
            pipeline_name=pipeline,
            metadata=metadata or {},
        )
        self._entries[job_id] = entry
        self._doc_index[doc_key] = job_id
        self._refresh_metrics()
        return entry

    def idempotent_create(
        self,
        *,
        job_id: str,
        doc_key: str,
        tenant_id: str,
        pipeline: str | None,
        metadata: dict[str, object] | None = None,
    ) -> JobLedgerEntry:
        existing_id = self._doc_index.get(doc_key)
        if existing_id:
            return self._entries[existing_id].snapshot()
        created = self.create(
            job_id=job_id,
            doc_key=doc_key,
            tenant_id=tenant_id,
            pipeline=pipeline,
            metadata=metadata,
        )
        self._refresh_metrics()
        return created

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def _update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        metadata: dict[str, object] | None = None,
        reason: str | None = None,
        current_stage: str | None = None,
        pipeline_name: str | None = None,
        pdf_downloaded: bool | None = None,
        pdf_ir_ready: bool | None = None,
        pdf_url: str | None = None,
        pdf_storage_key: str | None = None,
        pdf_size: int | None = None,
        pdf_content_type: str | None = None,
        pdf_checksum: str | None = None,
        pdf_downloaded_at: datetime | None = None,
        pdf_ir_ready_at: datetime | None = None,
        pdf_error: str | None = None,
        pdf_failure_code: str | None | Any = _UNSET,
        pdf_history_event: PdfHistoryEvent | None = None,
    ) -> JobLedgerEntry:
        if job_id not in self._entries:
            raise JobLedgerError(f"Job {job_id} not found")
        entry = self._entries[job_id]
        next_status = status or entry.status
        if next_status not in ALLOWED_TRANSITIONS:
            raise JobLedgerError(f"Unsupported status {next_status}")
        if status and next_status not in ALLOWED_TRANSITIONS[entry.status]:
            raise JobLedgerError(
                f"Invalid transition {entry.status} -> {next_status} for job {job_id}"
            )
        if status and status != entry.status:
            entry.history.append(
                JobTransition(
                    from_status=entry.status,
                    to_status=next_status,
                    stage=stage or entry.stage,
                    reason=reason,
                )
            )
            entry.status = next_status
        if stage:
            entry.stage = stage
        if current_stage:
            entry.current_stage = current_stage
        elif stage:
            entry.current_stage = stage
        if pipeline_name:
            entry.pipeline = pipeline_name
            entry.pipeline_name = pipeline_name
        if metadata:
            entry.metadata.update(metadata)
        if pdf_downloaded is not None:
            entry.pdf_downloaded = pdf_downloaded
        if pdf_url is not None:
            entry.pdf_url = pdf_url
        if pdf_storage_key is not None:
            entry.pdf_storage_key = pdf_storage_key
        if pdf_checksum is not None:
            entry.pdf_checksum = pdf_checksum
        if pdf_size_bytes is not None:
            entry.pdf_size_bytes = pdf_size_bytes
        if pdf_content_type is not None:
            entry.pdf_content_type = pdf_content_type
        if pdf_downloaded_at is not None:
            entry.pdf_downloaded_at = pdf_downloaded_at
        if pdf_ir_ready is not None:
            entry.pdf_ir_ready = pdf_ir_ready
        if pdf_url is not None:
            entry.pdf_url = pdf_url
        if pdf_storage_key is not None:
            entry.pdf_storage_key = pdf_storage_key
        if pdf_size is not None:
            entry.pdf_size = pdf_size
        if pdf_content_type is not None:
            entry.pdf_content_type = pdf_content_type
        if pdf_checksum is not None:
            entry.pdf_checksum = pdf_checksum
        if pdf_downloaded_at is not None:
            entry.pdf_downloaded_at = pdf_downloaded_at
        if pdf_ir_ready_at is not None:
            entry.pdf_ir_ready_at = pdf_ir_ready_at
        if pdf_error is not None:
            entry.pdf_error = pdf_error
        if pdf_failure_code is not _UNSET:
            entry.pdf_failure_code = cast(str | None, pdf_failure_code)
        if pdf_history_event is not None:
            entry.pdf_history.append(pdf_history_event)
        entry.updated_at = datetime.utcnow()
        self._refresh_metrics()
        return entry

    def update_metadata(self, job_id: str, metadata: dict[str, object]) -> JobLedgerEntry:
        return self._update(job_id, metadata=metadata)

    def update_pdf_state(
        self,
        job_id: str,
        *,
        url: str | None = None,
        storage_key: str | None = None,
        size: int | None = None,
        content_type: str | None = None,
        checksum: str | None = None,
        downloaded: bool | None = None,
        downloaded_at: datetime | None = None,
        ir_ready: bool | None = None,
        ir_ready_at: datetime | None = None,
        error: str | None = None,
        failure_code: str | None | Any = _UNSET,
        history_status: str | None = None,
        history_stage: str = "pdf",
        detail: str | None = None,
    ) -> JobLedgerEntry:
        event = None
        if history_status:
            event = PdfHistoryEvent(
                timestamp=datetime.utcnow(),
                stage=history_stage,
                status=history_status,
                detail=detail,
            )
        return self._update(
            job_id,
            pdf_url=url,
            pdf_storage_key=storage_key,
            pdf_size=size,
            pdf_content_type=content_type,
            pdf_checksum=checksum,
            pdf_downloaded=downloaded,
            pdf_ir_ready=ir_ready,
            pdf_downloaded_at=downloaded_at,
            pdf_ir_ready_at=ir_ready_at,
            pdf_error=error,
            pdf_failure_code=failure_code,
            pdf_history_event=event,
        )

    def mark_processing(self, job_id: str, stage: str) -> JobLedgerEntry:
        entry = self._update(
            job_id,
            status="processing",
            stage=stage,
            current_stage=stage,
        )
        entry.retry_count_per_stage.setdefault(stage, entry.retry_count_per_stage.get(stage, 0))
        return entry

    def assign_pipeline(self, job_id: str, pipeline_name: str | None) -> JobLedgerEntry:
        return self._update(job_id, pipeline_name=pipeline_name)

    def mark_stage_started(self, job_id: str, stage: str) -> JobLedgerEntry:
        entry = self.mark_processing(job_id, stage)
        entry.retry_count_per_stage.setdefault(stage, 0)
        return entry

    def mark_completed(
        self, job_id: str, *, metadata: dict[str, object] | None = None
    ) -> JobLedgerEntry:
        entry = self._update(
            job_id,
            status="completed",
            stage="completed",
            current_stage="completed",
            metadata=metadata,
        )
        entry.completed_at = datetime.utcnow()
        entry.duration_seconds = (entry.completed_at - entry.created_at).total_seconds()
        return entry

    def mark_failed(
        self,
        job_id: str,
        *,
        stage: str,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> JobLedgerEntry:
        entry = self._update(
            job_id,
            status="failed",
            stage=stage,
            current_stage=stage,
            metadata=metadata,
            reason=reason,
        )
        entry.error_reason = reason
        entry.completed_at = datetime.utcnow()
        entry.duration_seconds = (entry.completed_at - entry.created_at).total_seconds()
        return entry

    def mark_cancelled(self, job_id: str, *, reason: str | None = None) -> JobLedgerEntry:
        return self._update(
            job_id,
            status="cancelled",
            stage="cancelled",
            current_stage="cancelled",
            reason=reason,
        )

    def increment_retry(self, job_id: str, stage: str) -> JobLedgerEntry:
        if job_id not in self._entries:
            raise JobLedgerError(f"Job {job_id} not found")
        entry = self._entries[job_id]
        entry.retry_count += 1
        entry.retry_count_per_stage[stage] = entry.retry_count_per_stage.get(stage, 0) + 1
        entry.attempts = max(entry.attempts, entry.retry_count_per_stage[stage] + 1)
        entry.current_stage = stage
        entry.stage = stage
        entry.updated_at = datetime.utcnow()
        self._refresh_metrics()
        return entry

    def set_pdf_downloaded(
        self,
        job_id: str,
        value: bool = True,
        *,
        url: str | None = None,
        storage_key: str | None = None,
        size: int | None = None,
        content_type: str | None = None,
        checksum: str | None = None,
    ) -> JobLedgerEntry:
        timestamp = datetime.utcnow() if value else None
        status = "downloaded" if value else "reset"
        return self.update_pdf_state(
            job_id,
            url=url,
            storage_key=storage_key,
            size=size,
            content_type=content_type,
            checksum=checksum,
            downloaded=value,
            downloaded_at=timestamp,
            failure_code=None,
            history_status=status,
            detail="PDF downloaded" if value else "PDF download state reset",
        )

    def set_pdf_ir_ready(
        self,
        job_id: str,
        value: bool = True,
        *,
        checksum: str | None = None,
    ) -> JobLedgerEntry:
        timestamp = datetime.utcnow() if value else None
        status = "ir-ready" if value else "ir-reset"
        return self.update_pdf_state(
            job_id,
            checksum=checksum,
            ir_ready=value,
            ir_ready_at=timestamp,
            failure_code=None,
            history_status=status,
            history_stage="mineru",
            detail="MinerU processing completed" if value else "MinerU state reset",
        )

    def record_pdf_failure(
        self,
        job_id: str,
        *,
        stage: str,
        reason: str,
        retryable: bool = False,
        code: str | None = None,
    ) -> JobLedgerEntry:
        status = "retryable-error" if retryable else "error"
        entry = self.update_pdf_state(
            job_id,
            error=reason,
            failure_code=code,
            history_status=status,
            history_stage=stage,
            detail=reason,
        )
        return entry

    def record_pdf_partial(
        self,
        job_id: str,
        *,
        stage: str,
        detail: str,
        retryable: bool = True,
    ) -> JobLedgerEntry:
        status = "partial-retryable" if retryable else "partial"
        return self.update_pdf_state(
            job_id,
            history_status=status,
            history_stage=stage,
            detail=detail,
        )

    def rollback_pdf_state(self, job_id: str, *, reason: str | None = None) -> JobLedgerEntry:
        return self.update_pdf_state(
            job_id,
            storage_key=None,
            size=None,
            content_type=None,
            checksum=None,
            downloaded=False,
            downloaded_at=None,
            ir_ready=False,
            ir_ready_at=None,
            error=reason,
            failure_code=None,
            history_status="rollback",
            detail=reason or "PDF state rolled back",
        )

    def clear_pdf_error(self, job_id: str) -> JobLedgerEntry:
        entry = self.update_pdf_state(
            job_id,
            error=None,
            failure_code=None,
            history_status="cleared",
        )
        return entry

    def record_attempt(self, job_id: str) -> int:
        if job_id not in self._entries:
            raise JobLedgerError(f"Job {job_id} not found")
        entry = self._entries[job_id]
        entry.attempts += 1
        entry.retry_count = entry.attempts
        entry.updated_at = datetime.utcnow()
        return entry.attempts

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get(self, job_id: str) -> JobLedgerEntry | None:
        entry = self._entries.get(job_id)
        return entry.snapshot() if entry else None

    def list(self, *, status: str | None = None) -> builtins.list[JobLedgerEntry]:
        items = (
            entry.snapshot()
            for entry in self._entries.values()
            if status is None or entry.status == status
        )
        return sorted(items, key=lambda item: item.created_at)

    def by_doc_key(self, doc_key: str) -> JobLedgerEntry | None:
        job_id = self._doc_index.get(doc_key)
        return self.get(job_id) if job_id else None

    def all(self) -> Iterator[JobLedgerEntry]:
        for entry in self._entries.values():
            yield entry.snapshot()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _refresh_metrics(self) -> None:
        counts = Counter(entry.status for entry in self._entries.values())
        update_job_status_metrics(counts)


__all__ = ["JobLedger", "JobLedgerEntry", "JobLedgerError", "JobTransition"]
from Medical_KG_rev.observability.metrics import update_job_status_metrics

