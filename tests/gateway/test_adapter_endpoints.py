from fastapi.testclient import TestClient

from Medical_KG_rev.gateway.app import create_app


def _client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_list_adapters_returns_metadata(api_key: str) -> None:
    client = _client()
    response = client.get("/v1/adapters")
    assert response.status_code == 200
    payload = response.json()
    assert "data" in payload
    assert payload["meta"]["total"] >= 1
    names = [item["name"] for item in payload["data"]]
    assert "clinicaltrials" in names


def test_adapter_metadata_endpoint(api_key: str) -> None:
    client = _client()
    listing = client.get("/v1/adapters").json()
    name = listing["data"][0]["name"]
    response = client.get(f"/v1/adapters/{name}/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["name"] == name


def test_adapter_health_endpoint(api_key: str) -> None:
    client = _client()
    listing = client.get("/v1/adapters").json()
    name = listing["data"][0]["name"]
    response = client.get(f"/v1/adapters/{name}/health")
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["healthy"] is True


def test_adapter_config_schema_endpoint(api_key: str) -> None:
    client = _client()
    listing = client.get("/v1/adapters").json()
    name = listing["data"][0]["name"]
    response = client.get(f"/v1/adapters/{name}/config-schema")
    assert response.status_code == 200
    body = response.json()
    assert "schema" in body["data"]


def test_unknown_adapter_returns_404(api_key: str) -> None:
    client = _client()
    response = client.get("/v1/adapters/unknown/metadata")
    assert response.status_code == 404
