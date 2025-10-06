from fastapi.testclient import TestClient

from Medical_KG_rev.gateway.app import create_app


def test_tenant_mismatch_returns_403(api_key: str) -> None:
    app = create_app()
    client = TestClient(app)

    payload = {
        "tenant_id": "other",
        "items": [{"id": "item-1"}],
    }

    response = client.post(
        "/v1/ingest/clinicaltrials",
        json=payload,
        headers={"X-API-Key": api_key},
    )

    assert response.status_code == 403
    body = response.json()
    assert body["title"] == "Tenant mismatch"
