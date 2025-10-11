#!/usr/bin/env python3
"""gRPC server for GPU services with torch ecosystem."""

import asyncio
import logging
import signal
import sys
from pathlib import Path

import grpc
from grpc_health.v1 import health, health_pb2_grpc
from grpc_health.v1.health_pb2 import HealthCheckResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent


def _load_gpu_components():
    """Load GPU service modules after configuring sys.path."""
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    from proto import gpu_pb2_grpc  # type: ignore import-not-found
    from services.gpu.grpc_service import GPUServiceServicer  # type: ignore import-not-found

    return gpu_pb2_grpc, GPUServiceServicer

class HealthServicer(health.HealthServicer):
    """Health check servicer for gRPC health protocol."""

    def __init__(self):
        self._status_map = {}

    def Check(self, request, context):  # noqa: N802  # gRPC method signature defined by protocol
        """Check service health."""
        service = request.service
        status = self._status_map.get(service, HealthCheckResponse.UNKNOWN)
        return HealthCheckResponse(status=status)

    def set_status(self, service, status):
        """Set service health status."""
        self._status_map[service] = status


async def serve():
    """Start the gRPC server."""
    gpu_pb2_grpc, GPUServiceServicer = _load_gpu_components()

    # Create gRPC server
    server = grpc.aio.server()

    # Add GPU service
    gpu_servicer = GPUServiceServicer()
    gpu_pb2_grpc.add_GPUServiceServicer_to_server(gpu_servicer, server)

    # Add health check service
    health_servicer = HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # Set health status
    health_servicer.set_status("", HealthCheckResponse.SERVING)
    health_servicer.set_status("medicalkg.gateway.v1.GPUService", HealthCheckResponse.SERVING)

    # Listen on port 50051
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)

    logger.info("Starting GPU services gRPC server on %s", listen_addr)

    # Start server
    await server.start()

    # Wait for termination
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        await server.stop(grace=5.0)


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("Received signal %d, shutting down", signum)
    sys.exit(0)


if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(serve())
    except Exception as e:
        logger.error("Server failed to start: %s", e)
        sys.exit(1)
