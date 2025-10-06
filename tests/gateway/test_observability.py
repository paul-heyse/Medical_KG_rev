from __future__ import annotations

from fastapi.testclient import TestClient

from Medical_KG_rev.gateway.app import create_app


def test_metrics_endpoint_records_requests() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/docs/openapi")
    assert response.status_code == 200

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    body = metrics.text
    assert "api_requests_total" in body
    assert 'path="/docs/openapi"' in body


def test_business_metrics_increment_for_retrieve() -> None:
    app = create_app()
    client = TestClient(app)

    payload = {"tenant_id": "tenant", "query": "cancer", "top_k": 1}
    result = client.post("/v1/retrieve", json=payload)
    assert result.status_code == 200

    metrics = client.get("/metrics")
    assert 'business_events_total{event="retrieval_requests"}' in metrics.text
