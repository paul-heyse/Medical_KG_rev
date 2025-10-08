from __future__ import annotations

from dagster import RunRequest, SkipReason
from dagster._core.test_utils import build_sensor_context

from Medical_KG_rev.orchestration.dagster.runtime import pdf_ir_ready_sensor
from Medical_KG_rev.orchestration.ledger import JobLedger


def test_pdf_ir_sensor_skips_when_no_jobs() -> None:
    ledger = JobLedger()
    context = build_sensor_context(resources={"job_ledger": ledger})

    results = list(pdf_ir_ready_sensor(context))
    assert len(results) == 1
    assert isinstance(results[0], SkipReason)


def test_pdf_ir_sensor_emits_run_request() -> None:
    ledger = JobLedger()
    job_id = "job-sensor-1"
    ledger.create(
        job_id=job_id,
        doc_key="doc-sensor",
        tenant_id="tenant-1",
        pipeline="pdf-two-phase",
        metadata={
            "pipeline_version": "2025-01-01",
            "correlation_id": "corr-sensor",
            "adapter_request": {
                "tenant_id": "tenant-1",
                "correlation_id": "corr-sensor",
                "domain": "biomedical",
                "parameters": {"dataset": "pmc"},
            },
            "payload": {"dataset": "pmc", "item": {"id": "doc-sensor"}},
        },
    )
    ledger.mark_processing(job_id, stage="gate_pdf_ir_ready")
    ledger.set_pdf_ir_ready(job_id)

    context = build_sensor_context(resources={"job_ledger": ledger})
    results = list(pdf_ir_ready_sensor(context))

    assert results
    assert all(isinstance(item, RunRequest) for item in results)
    request = results[0]
    assert request.run_key == "job-sensor-1-resume"
    ctx_config = request.run_config["ops"]["bootstrap"]["config"]["context"]
    assert ctx_config["job_id"] == job_id
    assert ctx_config["pipeline_name"] == "pdf-two-phase"
