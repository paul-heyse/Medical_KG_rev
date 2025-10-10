"""Contract tests for the Docling PDF processing REST API."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from Medical_KG_rev.gateway.app import create_app
from Medical_KG_rev.gateway.models import DoclingProcessingResponse, DoclingProcessingPayload


class StubGatewayService:
    def __init__(self) -> None:
        self.docling_service = MagicMock()
        self.process_docling_pdf = MagicMock(
            return_value=DoclingProcessingResponse(
                result=DoclingProcessingPayload(
                    document_id="doc-1",
                    text="hello",
                    tables=[],
                    figures=[],
                    metadata={"provenance": {"model_name": "stub"}},
                ),
                model_name="stub",
                processing_time_seconds=0.5,
                gpu_memory_fraction=0.5,
            )
        )
        self.orchestrator = MagicMock()
        self.orchestrator.kafka.health.return_value = {"broker": True}
        self.ledger = MagicMock()


class StubDoclingService:
    def health(self) -> dict[str, str]:  # pragma: no cover - compatibility
        return {"status": "ok"}


def _fake_secure_endpoint(*_, **__):
    async def dependency():
        context = MagicMock()
        context.tenant_id = "tenant"
        return context

    return dependency


def test_docling_rest_endpoint_returns_payload(monkeypatch):
    stub_service = StubGatewayService()
    stub_service.docling_service = StubDoclingService()

    monkeypatch.setattr(
        "Medical_KG_rev.gateway.services.get_gateway_service",
        lambda: stub_service,
    )
    monkeypatch.setattr(
        "Medical_KG_rev.gateway.rest.router.get_gateway_service",
        lambda: stub_service,
    )
    monkeypatch.setattr(
        "Medical_KG_rev.gateway.rest.router.secure_endpoint",
        _fake_secure_endpoint,
    )

    client = TestClient(create_app())
    response = client.post(
        "/v1/pdf/docling/process",
        json={"document_id": "doc-1", "pdf_path": "/tmp/file.pdf"},
        headers={"Authorization": "Bearer token"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["document_id"] == "doc-1"
    assert payload["model_name"] == "stub"
