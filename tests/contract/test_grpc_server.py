from __future__ import annotations

from unittest import mock

import asyncio
import types

import pytest

grpc = pytest.importorskip("grpc")

from Medical_KG_rev.gateway.grpc.server import EmbeddingService, GatewayGrpcServer
from Medical_KG_rev.gateway.services import get_gateway_service
from Medical_KG_rev.gateway.models import EmbeddingMetadata, EmbeddingResponse, EmbeddingVector


def test_grpc_server_start_registers_services(monkeypatch) -> None:
    fake_server = mock.AsyncMock()
    monkeypatch.setattr("grpc.aio.server", lambda *args, **kwargs: fake_server)
    server = GatewayGrpcServer(service=get_gateway_service())
    # Even if protobuf stubs are missing, start should not raise.
    monkeypatch.setattr("Medical_KG_rev.gateway.grpc.server.mineru_pb2_grpc", None)
    monkeypatch.setattr("Medical_KG_rev.gateway.grpc.server.embedding_pb2_grpc", None)
    monkeypatch.setattr("Medical_KG_rev.gateway.grpc.server.extraction_pb2_grpc", None)
    monkeypatch.setattr("Medical_KG_rev.gateway.grpc.server.ingestion_pb2_grpc", None)
    mock_health_servicer = mock.Mock()
    monkeypatch.setattr(
        "Medical_KG_rev.gateway.grpc.server.health.HealthServicer",
        mock.Mock(return_value=mock_health_servicer),
    )
    monkeypatch.setattr(
        "Medical_KG_rev.gateway.grpc.server.health_pb2_grpc.add_HealthServicer_to_server",
        mock.Mock(),
    )
    monkeypatch.setattr(
        "Medical_KG_rev.gateway.grpc.server.health_pb2.HealthCheckResponse",
        mock.Mock(SERVING="SERVING"),
    )

    async def _start():
        await server.start()

    # Ensure the coroutine executes without raising
    import asyncio

    asyncio.run(_start())
    fake_server.add_insecure_port.assert_called()


@pytest.mark.asyncio()
async def test_embedding_service_serializes_response(monkeypatch: pytest.MonkeyPatch) -> None:
    service = get_gateway_service()

    stub_response = EmbeddingResponse(
        namespace="single_vector.qwen3.4096.v1",
        embeddings=[
            EmbeddingVector(
                id="vec-1",
                model="stub-model",
                namespace="single_vector.qwen3.4096.v1",
                kind="single_vector",
                dimension=3,
                vector=[0.1, 0.2, 0.3],
                metadata={"tenant_id": "tenant"},
            )
        ],
        metadata=EmbeddingMetadata(provider="vllm", dimension=3, duration_ms=5.0, model="stub-model"),
    )
    monkeypatch.setattr(service, "embed", lambda request: stub_response)

    class _EmbedVector:
        def __init__(self) -> None:
            self.id = ""
            self.model = ""
            self.namespace = ""
            self.kind = ""
            self.dimension = 0
            self.values: list[float] = []
            self.terms: dict[str, float] = {}
            self.metadata: dict[str, str] = {}

    class _Embeddings(list):
        def add(self):
            vector = _EmbedVector()
            self.append(vector)
            return vector

    class _Metadata:
        def __init__(self) -> None:
            self.provider = ""
            self.dimension = 0
            self.duration_ms = 0.0
            self.model = ""

    class _EmbedResponse:
        def __init__(self) -> None:
            self.namespace = ""
            self.embeddings = _Embeddings()
            self.metadata = _Metadata()

    stub_module = types.SimpleNamespace(EmbedResponse=_EmbedResponse)
    monkeypatch.setattr(
        "Medical_KG_rev.gateway.grpc.server.embedding_pb2",
        stub_module,
    )

    request = types.SimpleNamespace(
        tenant_id="tenant",
        inputs=["alpha"],
        namespace="single_vector.qwen3.4096.v1",
        normalize=False,
    )

    response = await EmbeddingService(service).Embed(request, None)
    assert response.namespace == stub_response.namespace
    assert response.metadata.model == "stub-model"
    assert response.embeddings[0].metadata["tenant_id"] == "tenant"
