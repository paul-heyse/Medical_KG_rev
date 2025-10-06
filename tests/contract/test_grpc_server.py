from __future__ import annotations

from unittest import mock

import pytest

grpc = pytest.importorskip("grpc")

from Medical_KG_rev.gateway.grpc.server import GatewayGrpcServer
from Medical_KG_rev.gateway.services import get_gateway_service


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
    monkeypatch.setattr("Medical_KG_rev.gateway.grpc.server.health.HealthServicer", mock.Mock(return_value=mock_health_servicer))
    monkeypatch.setattr("Medical_KG_rev.gateway.grpc.server.health_pb2_grpc.add_HealthServicer_to_server", mock.Mock())
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
