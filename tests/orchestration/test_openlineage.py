from Medical_KG_rev.orchestration.ledger import JobLedgerEntry
from Medical_KG_rev.orchestration.openlineage import OpenLineageEmitter


def test_openlineage_emitter_builds_run_events() -> None:
    emitter = OpenLineageEmitter(enabled=True)
    entry = JobLedgerEntry(
        job_id="job-ol-1",
        doc_key="doc-ol-1",
        tenant_id="tenant",
        pipeline="auto",
        pipeline_name="auto",
        retry_count_per_stage={"chunk": 1},
    )
    run_metadata = {
        "metrics": {"gpu_memory_mb": 1024, "gpu_utilization_percent": 58.0},
        "models": {"embed": {"name": "qwen", "version": "1.5"}},
    }

    payload = emitter.emit_run_completed(
        "auto",
        run_id="run-1",
        context={},
        attempt=2,
        ledger_entry=entry,
        run_metadata=run_metadata,
        duration_ms=1500,
    )

    assert payload["eventType"] == "COMPLETE"
    assert payload["job"]["name"] == "auto"
    run_facets = payload["run"]["facets"]
    assert run_facets["retryAttempts"]["attempts"]["chunk"] == 1
    job_facets = payload["jobFacets"]
    assert "gpuUtilization" in job_facets
    assert "modelVersion" in job_facets


def test_openlineage_failure_event_records_error() -> None:
    emitter = OpenLineageEmitter(enabled=True)
    error_payload = emitter.emit_run_failed(
        "auto",
        run_id="run-err",
        context={},
        attempt=1,
        ledger_entry=None,
        run_metadata=None,
        error="boom",
    )

    assert error_payload["eventType"] == "FAIL"
    assert error_payload["message"] == "boom"
