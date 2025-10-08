"""Job ledger tracking orchestration state."""

from __future__ import annotations

import builtins
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
ALLOWED_TRANSITIONS = {
    "queued": {"processing", "cancelled"},
    "processing": {"processing", "completed", "failed", "cancelled"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}


@dataclass
class JobTransition:
    from_status: str
    to_status: str
    stage: str
    reason: str | None
    timestamp: datetime = field(default_factory=datetime.utcnow)


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
    phase: str = "phase-1"
    gate_state: dict[str, dict[str, object]] = field(default_factory=dict)

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
            phase=self.phase,
            gate_state={name: dict(state) for name, state in self.gate_state.items()},
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
        phase: str | None = None,
        gate_state: dict[str, dict[str, object]] | None = None,
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
        if pdf_ir_ready is not None:
            entry.pdf_ir_ready = pdf_ir_ready
        if phase:
            entry.phase = phase
        if gate_state:
            for gate, values in gate_state.items():
                existing = entry.gate_state.get(gate, {})
                existing.update(values)
                entry.gate_state[gate] = existing
        entry.updated_at = datetime.utcnow()
        self._refresh_metrics()
        return entry

    def update_metadata(self, job_id: str, metadata: dict[str, object]) -> JobLedgerEntry:
        return self._update(job_id, metadata=metadata)

    def mark_processing(self, job_id: str, stage: str) -> JobLedgerEntry:
        entry = self._update(
            job_id,
            status="processing",
            stage=stage,
            current_stage=stage,
        )
        entry.retry_count_per_stage.setdefault(stage, entry.retry_count_per_stage.get(stage, 0))
        return entry

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

    def set_pdf_downloaded(self, job_id: str, value: bool = True) -> JobLedgerEntry:
        return self._update(job_id, pdf_downloaded=value)

    def set_pdf_ir_ready(self, job_id: str, value: bool = True) -> JobLedgerEntry:
        return self._update(job_id, pdf_ir_ready=value)

    def set_phase(self, job_id: str, phase: str) -> JobLedgerEntry:
        return self._update(job_id, phase=phase)

    def record_gate_state(
        self,
        job_id: str,
        gate_name: str,
        *,
        status: str,
        reason: str | None = None,
        attempts: int | None = None,
        elapsed_seconds: float | None = None,
        extra: dict[str, object] | None = None,
    ) -> JobLedgerEntry:
        state: dict[str, object] = {"status": status, "updated_at": datetime.utcnow().isoformat()}
        if reason:
            state["reason"] = reason
        if attempts is not None:
            state["attempts"] = attempts
        if elapsed_seconds is not None:
            state["elapsed_seconds"] = elapsed_seconds
        if extra:
            state.update(extra)
        return self._update(job_id, gate_state={gate_name: state})

    def get_gate_state(self, job_id: str, gate_name: str) -> dict[str, object] | None:
        entry = self._entries.get(job_id)
        if not entry:
            return None
        state = entry.gate_state.get(gate_name)
        return dict(state) if state else None

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

