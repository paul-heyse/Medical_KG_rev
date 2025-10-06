"""Launch all GPU-backed gRPC microservices."""

from __future__ import annotations

import asyncio
import signal
from typing import Sequence

import structlog

try:  # pragma: no cover - gRPC optional during unit testing
    import grpc
    from grpc_health.v1 import health_pb2_grpc
except Exception:  # pragma: no cover
    grpc = None  # type: ignore

from Medical_KG_rev.services.embedding import (
    EmbeddingGrpcService,
    EmbeddingModelRegistry,
    EmbeddingWorker,
)
from Medical_KG_rev.services.extraction import ExtractionGrpcService, ExtractionService
from Medical_KG_rev.services.gpu import GpuManager
from Medical_KG_rev.services.grpc import (
    GrpcServiceState,
    UnaryUnaryLoggingInterceptor,
    UnaryUnaryTracingInterceptor,
)
from Medical_KG_rev.services.mineru import MineruGrpcService, MineruProcessor

logger = structlog.get_logger(__name__)


async def _start_server(
    service_name: str,
    servicer,
    add_to_server,
    *,
    port: int,
    interceptors: Sequence[object],
    state: GrpcServiceState,
) -> grpc.aio.Server:
    if grpc is None:
        raise RuntimeError("grpcio must be installed to run GPU microservices")
    server = grpc.aio.server(interceptors=list(interceptors))
    add_to_server(servicer, server)
    if state.health_servicer is not None:
        health_pb2_grpc.add_HealthServicer_to_server(state.health_servicer, server)
    server.add_insecure_port(f"0.0.0.0:{port}")
    await server.start()
    state.set_ready()
    logger.info("gpu.service.started", service=service_name, port=port)
    return server


async def serve_async() -> None:
    manager = GpuManager()
    mineru_processor = MineruProcessor(manager)
    embedding_registry = EmbeddingModelRegistry(manager)
    embedding_worker = EmbeddingWorker(embedding_registry)
    extraction_service = ExtractionService(manager)

    if grpc is None:
        raise RuntimeError("grpcio must be installed to run GPU microservices")

    from Medical_KG_rev.proto.gen import (
        embedding_pb2_grpc,
        extraction_pb2_grpc,
        mineru_pb2_grpc,
    )  # type: ignore import-error

    interceptors = [
        UnaryUnaryTracingInterceptor("gpu"),
        UnaryUnaryLoggingInterceptor("gpu"),
    ]

    mineru_state = GrpcServiceState("MineruService")
    embedding_state = GrpcServiceState("EmbeddingService")
    extraction_state = GrpcServiceState("ExtractionService")

    servers = await asyncio.gather(
        _start_server(
            "mineru",
            MineruGrpcService(mineru_processor),
            mineru_pb2_grpc.add_MineruServiceServicer_to_server,
            port=7000,
            interceptors=interceptors,
            state=mineru_state,
        ),
        _start_server(
            "embedding",
            EmbeddingGrpcService(embedding_worker),
            embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server,
            port=7001,
            interceptors=interceptors,
            state=embedding_state,
        ),
        _start_server(
            "extraction",
            ExtractionGrpcService(extraction_service),
            extraction_pb2_grpc.add_ExtractionServiceServicer_to_server,
            port=7002,
            interceptors=interceptors,
            state=extraction_state,
        ),
    )

    stop_event = asyncio.Event()

    def _signal_handler(*_):  # pragma: no cover - depends on runtime signals
        logger.info("gpu.service.shutdown_requested")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    await stop_event.wait()

    await asyncio.gather(*(server.stop(grace=None) for server in servers))
    logger.info("gpu.services.stopped")


def main() -> None:
    asyncio.run(serve_async())


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
