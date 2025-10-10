from __future__ import annotations

import pytest

from Medical_KG_rev.gateway.app import create_app
from Medical_KG_rev.gateway.models import EmbeddingMetadata, EmbeddingResponse, EmbeddingVector
from Medical_KG_rev.gateway.services import get_gateway_service

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


def _prepare_stub_response() -> EmbeddingResponse:
    vector = EmbeddingVector(
        id="job-123:chunk:0",
        model="stub-model",
        namespace="single_vector.qwen3.4096.v1",
        kind="single_vector",
        dimension=3,
        vector=[0.1, 0.2, 0.3],
        metadata={"tenant_id": "tenant", "storage": {"faiss_index": "/data/faiss/tenant/ns.index"}},
    )
    metadata = EmbeddingMetadata(provider="vllm", dimension=3, duration_ms=10.0, model="stub-model")
    return EmbeddingResponse(namespace=vector.namespace, embeddings=[vector], metadata=metadata)


def test_graphql_embed_uses_authenticated_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app()
    service = get_gateway_service()
    monkeypatch.setattr(service, "embed", lambda request: _prepare_stub_response())
    client = TestClient(app)
    mutation = """
    mutation EmbedTexts($texts: [String!]!, $namespace: String!) {
      embed(input: {texts: $texts, namespace: $namespace}) {
        namespace
        embeddings { id model metadata }
      }
    }
    """
    variables = {"texts": ["alpha"], "namespace": "single_vector.qwen3.4096.v1"}
    response = client.post("/graphql", json={"query": mutation, "variables": variables})
    assert response.status_code == 200
    payload = response.json()["data"]["embed"]
    assert payload["namespace"] == "single_vector.qwen3.4096.v1"
    metadata = payload["embeddings"][0]["metadata"]
    assert metadata["tenant_id"] == "tenant"


def test_graphql_namespaces_defaults_to_context_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app()
    service = get_gateway_service()
    # Ensure namespace listing returns deterministic data
    namespaces = service.list_namespaces(tenant_id="tenant")
    monkeypatch.setattr(service, "list_namespaces", lambda tenant_id, scope: namespaces)
    client = TestClient(app)
    query = """
    query { namespaces { id provider allowedTenants } }
    """
    response = client.post("/graphql", json={"query": query})
    assert response.status_code == 200
    payload = response.json()["data"]["namespaces"]
    assert payload
    assert any(entry["id"] == namespaces[0].id for entry in payload)
