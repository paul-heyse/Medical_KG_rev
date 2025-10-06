from fastapi.testclient import TestClient

from Medical_KG_rev.gateway.app import create_app


def test_health_endpoints(api_key: str):
    app = create_app()
    client = TestClient(app)

    response = client.get("/health", headers={"X-API-Key": api_key})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "uptime_seconds" in payload

    ready = client.get("/ready", headers={"X-API-Key": api_key})
    assert ready.status_code == 200
    checks = ready.json()["checks"]
    assert "kafka" in checks


def test_caching_etag_flow(api_key: str):
    app = create_app()
    client = TestClient(app)

    headers = {"X-API-Key": api_key}
    first = client.get("/v1/search", params={"query": "hypertension"}, headers=headers)
    assert first.status_code == 200
    etag = first.headers.get("etag")
    assert etag
    assert first.headers["cache-control"].startswith("private")

    cached = client.get(
        "/v1/search",
        params={"query": "hypertension"},
        headers={"If-None-Match": etag, "X-API-Key": api_key},
    )
    assert cached.status_code == 304


def test_post_requests_are_not_cached(api_key: str):
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/v1/retrieve",
        json={"tenant_id": "tenant", "query": "test"},
        headers={"X-API-Key": api_key},
    )
    assert response.headers["cache-control"] == "no-store"
