from __future__ import annotations

from typing import Any, Mapping

from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.state_manager import LedgerStateManager


class _StubRequest:
    def __init__(self, **payload: Any) -> None:
        self._payload = payload

    def model_dump(self) -> Mapping[str, Any]:
        return dict(self._payload)


def test_prepare_run_updates_metadata_and_stage() -> None:
    ledger = JobLedger()
    entry = ledger.create(
        job_id="job-1",
        doc_key="doc-1",
        tenant_id="tenant-1",
        pipeline="stub",
        metadata={},
    )
    assert entry.status == "queued"

    manager = LedgerStateManager(ledger)
    manager.prepare_run(
        {"source": "tests"},
        job_id="job-1",
        pipeline="stub",
        pipeline_version="v1",
        adapter_request=_StubRequest(adapter="clinical-trials"),
        payload={"seed": 42},
    )

    updated = ledger.get("job-1")
    assert updated is not None
    assert updated.status == "processing"
    assert updated.stage == "bootstrap"
    assert updated.metadata["pipeline_version"] == "v1"
    assert updated.metadata["context"] == {"source": "tests"}
    assert updated.metadata["payload"] == {"seed": 42}
    assert updated.metadata["adapter_request"]["adapter"] == "clinical-trials"


def test_stage_lifecycle_records_metrics() -> None:
    ledger = JobLedger()
    ledger.create(
        job_id="job-2",
        doc_key="doc-2",
        tenant_id="tenant-1",
        pipeline="stub",
        metadata={},
    )

    manager = LedgerStateManager(ledger)
    attempt = manager.stage_started("job-2", "chunk")
    assert attempt.attempt == 1

    manager.record_retry("job-2", "chunk")
    retry_entry = ledger.get("job-2")
    assert retry_entry is not None
    assert retry_entry.retry_count_per_stage.get("chunk") == 1

    manager.stage_succeeded("job-2", "chunk", attempts=2, output_count=3, duration_ms=1200)
    metadata = ledger.get("job-2").metadata  # type: ignore[union-attr]
    assert metadata["stage.chunk.attempts"] == 2
    assert metadata["stage.chunk.output_count"] == 3
    assert metadata["stage.chunk.duration_ms"] == 1200
    assert "stage.chunk.completed_at" in metadata


def test_failure_operations_are_noops_for_missing_job() -> None:
    ledger = JobLedger()
    manager = LedgerStateManager(ledger)

    # Should not raise
    manager.stage_failed("unknown-job", "embed", reason="boom")
    manager.stage_succeeded("unknown-job", "embed", attempts=1, output_count=0, duration_ms=10)
    manager.record_retry("unknown-job", "embed")
    manager.prepare_run({}, job_id=None, pipeline="p", pipeline_version="v", adapter_request=None, payload=None)
    assert ledger.get("unknown-job") is None
