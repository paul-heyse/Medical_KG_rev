from __future__ import annotations

import pytest

from Medical_KG_rev.orchestration.ledger import JobLedger, JobLedgerError


def test_create_and_retrieve_entry() -> None:
    ledger = JobLedger()
    entry = ledger.create(job_id="job-1", doc_key="doc-1", tenant_id="tenant", pipeline="auto")
    assert entry.job_id == "job-1"

    retrieved = ledger.get("job-1")
    assert retrieved is not None
    assert retrieved.doc_key == "doc-1"


def test_idempotent_create_returns_existing() -> None:
    ledger = JobLedger()
    ledger.create(job_id="job-1", doc_key="doc-1", tenant_id="tenant", pipeline="auto")

    duplicate = ledger.idempotent_create(
        job_id="job-2", doc_key="doc-1", tenant_id="tenant", pipeline="auto"
    )
    assert duplicate.job_id == "job-1"


def test_invalid_transition_raises() -> None:
    ledger = JobLedger()
    ledger.create(job_id="job-1", doc_key="doc-1", tenant_id="tenant", pipeline="auto")
    ledger.mark_processing("job-1", stage="ingest")
    ledger.mark_completed("job-1")

    with pytest.raises(JobLedgerError):
        ledger.mark_processing("job-1", stage="again")


def test_update_metadata_and_attempts() -> None:
    ledger = JobLedger()
    ledger.create(job_id="job-1", doc_key="doc-1", tenant_id="tenant", pipeline="auto")
    ledger.update_metadata("job-1", {"chunks": 4})

    entry = ledger.get("job-1")
    assert entry is not None
    assert entry.metadata["chunks"] == 4

    attempts = ledger.record_attempt("job-1")
    assert attempts == 1


def test_mark_failed_records_history() -> None:
    ledger = JobLedger()
    ledger.create(job_id="job-1", doc_key="doc-1", tenant_id="tenant", pipeline="auto")
    ledger.mark_processing("job-1", stage="ingest")
    ledger.mark_failed("job-1", stage="ingest", reason="boom")

    entry = ledger.get("job-1")
    assert entry is not None
    assert entry.status == "failed"
    assert entry.history[-1].to_status == "failed"
