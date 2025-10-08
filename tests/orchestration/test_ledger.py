from __future__ import annotations

import pytest

from Medical_KG_rev.orchestration.ledger import JobLedger, JobLedgerError


def test_create_and_retrieve_entry() -> None:
    ledger = JobLedger()
    entry = ledger.create(job_id="job-1", doc_key="doc-1", tenant_id="tenant", pipeline="auto")
    assert entry.job_id == "job-1"
    assert entry.pipeline_name == "auto"
    assert entry.current_stage == "pending"
    assert not entry.pdf_downloaded
    assert not entry.pdf_ir_ready

    retrieved = ledger.get("job-1")
    assert retrieved is not None
    assert retrieved.doc_key == "doc-1"
    assert retrieved.pipeline_name == "auto"


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

    ledger.set_pdf_downloaded("job-1")
    ledger.set_pdf_ir_ready("job-1")
    updated = ledger.get("job-1")
    assert updated is not None
    assert updated.pdf_downloaded is True
    assert updated.pdf_ir_ready is True
    assert updated.pdf_history[-2].status == "downloaded"
    assert updated.pdf_history[-1].status == "ir-ready"


def test_pdf_state_history_and_errors() -> None:
    ledger = JobLedger()
    ledger.create(job_id="job-1", doc_key="doc-1", tenant_id="tenant", pipeline="auto")

    ledger.update_pdf_state(
        "job-1",
        url="https://example.com/file.pdf",
        storage_key="tenant/doc/file.pdf",
        size=1024,
        content_type="application/pdf",
        history_status="scheduled",
        detail="Queued for download",
    )

    entry = ledger.get("job-1")
    assert entry is not None
    assert entry.pdf_url == "https://example.com/file.pdf"
    assert entry.pdf_storage_key == "tenant/doc/file.pdf"
    assert entry.pdf_history[-1].status == "scheduled"

    ledger.record_pdf_failure(
        "job-1", stage="download", reason="timeout", retryable=True, code="timeout"
    )
    failed = ledger.get("job-1")
    assert failed is not None
    assert failed.pdf_error == "timeout"
    assert failed.pdf_history[-1].status == "retryable-error"
    assert failed.pdf_failure_code == "timeout"

    ledger.clear_pdf_error("job-1")
    cleared = ledger.get("job-1")
    assert cleared is not None
    assert cleared.pdf_error is None
    assert cleared.pdf_history[-1].status == "cleared"
    assert cleared.pdf_failure_code is None


def test_pdf_partial_and_rollback_state() -> None:
    ledger = JobLedger()
    ledger.create(job_id="job-2", doc_key="doc-2", tenant_id="tenant", pipeline="auto")

    ledger.record_pdf_partial("job-2", stage="mineru", detail="timeout", retryable=True)
    entry = ledger.get("job-2")
    assert entry is not None
    assert entry.pdf_history[-1].status == "partial-retryable"

    ledger.rollback_pdf_state("job-2", reason="cleanup")
    rolled_back = ledger.get("job-2")
    assert rolled_back is not None
    assert rolled_back.pdf_downloaded is False
    assert rolled_back.pdf_ir_ready is False
    assert rolled_back.pdf_failure_code is None
    assert rolled_back.pdf_history[-1].status == "rollback"


def test_mark_failed_records_history() -> None:
    ledger = JobLedger()
    ledger.create(job_id="job-1", doc_key="doc-1", tenant_id="tenant", pipeline="auto")
    ledger.mark_stage_started("job-1", stage="ingest")
    ledger.increment_retry("job-1", stage="ingest")
    ledger.mark_failed("job-1", stage="ingest", reason="boom")

    entry = ledger.get("job-1")
    assert entry is not None
    assert entry.status == "failed"
    assert entry.history[-1].to_status == "failed"
    assert entry.retry_count_per_stage["ingest"] == 1
