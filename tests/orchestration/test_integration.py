from __future__ import annotations

import time

from fastapi.testclient import TestClient

from Medical_KG_rev.gateway.app import create_app
from Medical_KG_rev.gateway.services import get_gateway_service
from Medical_KG_rev.orchestration.orchestrator import INGEST_RESULTS_TOPIC, PipelineStage


def drain_results() -> None:
    service = get_gateway_service()
    list(service.orchestrator.kafka.consume(INGEST_RESULTS_TOPIC))


def test_ingestion_job_lifecycle(api_key: str) -> None:
    app = create_app()
    client = TestClient(app)
    service = get_gateway_service()

    payload = {
        "tenant_id": "tenant",
        "items": [{"id": "demo-1", "document_type": "pdf"}],
        "metadata": {"requested_by": "test"},
    }

    response = client.post(
        "/v1/ingest/clinicaltrials", json=payload, headers={"X-API-Key": api_key}
    )
    assert response.status_code == 207
    data = response.json()["data"]
    job_id = data[0]["job_id"]

    for worker in service.workers:
        worker.run_once()
    drain_results()

    job_response = client.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": api_key})
    assert job_response.status_code == 200
    job_data = job_response.json()["data"]
    assert job_data["status"] == "completed"
    assert job_data["pipeline"] == "two-phase"

    list_response = client.get("/v1/jobs", headers={"X-API-Key": api_key})
    assert list_response.status_code == 200
    job_ids = [item["job_id"] for item in list_response.json()["data"]]
    assert job_id in job_ids


def test_cancel_job_before_processing(api_key: str) -> None:
    app = create_app()
    client = TestClient(app)

    payload = {
        "tenant_id": "tenant",
        "items": [{"id": "demo-2"}],
    }

    response = client.post(
        "/v1/ingest/clinicaltrials", json=payload, headers={"X-API-Key": api_key}
    )
    job_id = response.json()["data"][0]["job_id"]

    cancel_response = client.post(f"/v1/jobs/{job_id}/cancel", headers={"X-API-Key": api_key})
    assert cancel_response.status_code == 202
    cancel_data = cancel_response.json()["data"]
    assert cancel_data["status"] == "cancelled"


def test_concurrent_job_processing() -> None:
    service = get_gateway_service()
    job_ids = []
    for index in range(5):
        entry = service.orchestrator.submit_job(
            tenant_id="tenant",
            dataset="bulk",
            item={"id": f"bulk-{index}"},
            priority="high" if index % 2 == 0 else "normal",
        )
        job_ids.append(entry.job_id)

    for _ in range(3):
        for worker in service.workers:
            worker.run_once()
    drain_results()

    statuses = [service.get_job(job_id, tenant_id="tenant") for job_id in job_ids]
    assert all(status and status.status == "completed" for status in statuses)


def test_retry_logic_with_transient_failure() -> None:
    service = get_gateway_service()
    orchestrator = service.orchestrator
    original_stage = orchestrator.pipelines["auto"].stages[1]
    state = {"failed": False}

    def flaky_stage(entry, context):
        if not state["failed"]:
            state["failed"] = True
            raise RuntimeError("transient failure")
        return original_stage.handler(entry, context)

    orchestrator.pipelines["auto"].stages[1] = PipelineStage("chunk", flaky_stage)
    entry = orchestrator.submit_job(tenant_id="tenant", dataset="chaos", item={"id": "chaos"})

    service.workers[0].run_once()
    time.sleep(1.1)
    service.workers[0].run_once()
    service.workers[1].run_once()
    drain_results()

    status = service.get_job(entry.job_id, tenant_id="tenant")
    assert status is not None
    assert status.status == "completed"
    assert status.attempts >= 1

    orchestrator.pipelines["auto"].stages[1] = original_stage
