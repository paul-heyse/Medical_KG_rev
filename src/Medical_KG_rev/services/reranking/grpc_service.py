"""Placeholder gRPC servicer for the reranking microservice."""

from __future__ import annotations

import logging
from typing import Iterable

import grpc
from google.protobuf import empty_pb2

from Medical_KG_rev.services.reranking import fusion

logger = logging.getLogger(__name__)


class RerankingService(grpc.ServiceRpcHandler):  # pragma: no cover - shim only
    """Minimal implementation that reports the service as unavailable."""

    def service_name(self) -> str:
        return "Medical_KG_rev.reranking.RerankingService"

    def unary_unary_handlers(self) -> Iterable[tuple[str, grpc.RpcMethodHandler]]:
        # Returning empty iterable keeps the gRPC server alive but disables RPCs.
        return []


def add_RerankingServiceServicer_to_server(*_args, **_kwargs) -> None:  # noqa: N802
    """Compatibility helper used by gRPC generated code."""
    logger.warning("Reranking gRPC service is not available in this build")


__all__ = ["RerankingService", "add_RerankingServiceServicer_to_server", "empty_pb2", "fusion"]
