"""Job ledger tracking orchestration state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Optional


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
    reason: Optional[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class JobLedgerEntry:
    job_id: str
    doc_key: str
    tenant_id: str
    status: str = "queued"
    stage: str = "pending"
    pipeline: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    attempts: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    history: List[JobTransition] = field(default_factory=list)

    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def snapshot(self) -> "JobLedgerEntry":
        """Return a copy suitable for external consumption."""

        return JobLedgerEntry(
            job_id=self.job_id,
            doc_key=self.doc_key,
            tenant_id=self.tenant_id,
            status=self.status,
            stage=self.stage,
            pipeline=self.pipeline,
            metadata=dict(self.metadata),
            attempts=self.attempts,
            created_at=self.created_at,
            updated_at=self.updated_at,
            history=list(self.history),
        )


class JobLedgerError(RuntimeError):
    pass


class JobLedger:
    """In-memory ledger implementation with idempotency helpers."""

    def __init__(self) -> None:
        self._entries: Dict[str, JobLedgerEntry] = {}
        self._doc_index: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Creation & idempotency
    # ------------------------------------------------------------------
    def create(
        self,
        *,
        job_id: str,
        doc_key: str,
        tenant_id: str,
        pipeline: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
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
            metadata=metadata or {},
        )
        self._entries[job_id] = entry
        self._doc_index[doc_key] = job_id
        return entry

    def idempotent_create(
        self,
        *,
        job_id: str,
        doc_key: str,
        tenant_id: str,
        pipeline: Optional[str],
        metadata: Optional[Dict[str, object]] = None,
    ) -> JobLedgerEntry:
        existing_id = self._doc_index.get(doc_key)
        if existing_id:
            return self._entries[existing_id].snapshot()
        return self.create(
            job_id=job_id,
            doc_key=doc_key,
            tenant_id=tenant_id,
            pipeline=pipeline,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def _update(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
        reason: Optional[str] = None,
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
        if metadata:
            entry.metadata.update(metadata)
        entry.updated_at = datetime.utcnow()
        return entry

    def update_metadata(self, job_id: str, metadata: Dict[str, object]) -> JobLedgerEntry:
        return self._update(job_id, metadata=metadata)

    def mark_processing(self, job_id: str, stage: str) -> JobLedgerEntry:
        return self._update(job_id, status="processing", stage=stage)

    def mark_completed(self, job_id: str, *, metadata: Optional[Dict[str, object]] = None) -> JobLedgerEntry:
        return self._update(job_id, status="completed", stage="completed", metadata=metadata)

    def mark_failed(
        self,
        job_id: str,
        *,
        stage: str,
        reason: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> JobLedgerEntry:
        return self._update(
            job_id,
            status="failed",
            stage=stage,
            metadata=metadata,
            reason=reason,
        )

    def mark_cancelled(self, job_id: str, *, reason: Optional[str] = None) -> JobLedgerEntry:
        return self._update(job_id, status="cancelled", stage="cancelled", reason=reason)

    def record_attempt(self, job_id: str) -> int:
        if job_id not in self._entries:
            raise JobLedgerError(f"Job {job_id} not found")
        entry = self._entries[job_id]
        entry.attempts += 1
        entry.updated_at = datetime.utcnow()
        return entry.attempts

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get(self, job_id: str) -> Optional[JobLedgerEntry]:
        entry = self._entries.get(job_id)
        return entry.snapshot() if entry else None

    def list(self, *, status: Optional[str] = None) -> List[JobLedgerEntry]:
        items = (
            entry.snapshot()
            for entry in self._entries.values()
            if status is None or entry.status == status
        )
        return sorted(items, key=lambda item: item.created_at)

    def by_doc_key(self, doc_key: str) -> Optional[JobLedgerEntry]:
        job_id = self._doc_index.get(doc_key)
        return self.get(job_id) if job_id else None

    def all(self) -> Iterator[JobLedgerEntry]:
        for entry in self._entries.values():
            yield entry.snapshot()


__all__ = ["JobLedger", "JobLedgerEntry", "JobTransition", "JobLedgerError"]
