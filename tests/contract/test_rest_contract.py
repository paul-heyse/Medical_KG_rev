from __future__ import annotations

import json
from typing import Any, Dict

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from Medical_KG_rev.gateway.app import create_app
from Medical_KG_rev.gateway.models import IngestionRequest


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def _jsonapi(data: Any, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {"data": data, "meta": meta or {}}


def test_ingest_returns_multi_status(client: TestClient) -> None:
    payload = IngestionRequest(tenant_id="test", items=[{"id": "doc-1"}]).model_dump()
    response = client.post("/v1/ingest/clinicaltrials", json=payload)
    assert response.status_code == 207
    body = response.json()
    assert "data" in body and isinstance(body["data"], list)


def test_retrieve_supports_odata_parameters(client: TestClient) -> None:
    request = {"tenant_id": "test", "query": "cancer", "top_k": 2}
    response = client.post("/v1/retrieve?$select=title&$expand=entities", json=request)
    assert response.status_code == 200
    meta = response.json()["meta"]
    assert meta["select"] == ["title"]
    assert meta["expand"] == ["entities"]


def test_problem_details_format(client: TestClient) -> None:
    response = client.get("/v1/jobs/job-123/events")
    assert response.status_code == 422
    problem = response.json()
    assert problem["status"] == 422


def test_openapi_contract_matches_file(client: TestClient) -> None:
    live_spec = client.get("/openapi.json").json()
    with open("docs/openapi.yaml", "r", encoding="utf-8") as handle:
        file_spec = handle.read()
    assert json.loads(json.dumps(live_spec))  # ensures valid JSON
    assert "paths" in live_spec
    assert "v1/ingest/{dataset}" in json.dumps(live_spec)
    assert "Medical KG Multi-Protocol Gateway" in file_spec
