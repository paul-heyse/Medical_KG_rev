"""gRPC server helpers shared across GPU microservices."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import structlog

try:  # pragma: no cover - gRPC optional in unit tests
    import grpc
    from grpc.aio import HandlerCallDetails, ServerInterceptor
    from grpc_health.v1 import health, health_pb2
except Exception:  # pragma: no cover
    grpc = None  # type: ignore
    HandlerCallDetails = Any  # type: ignore
    ServerInterceptor = object  # type: ignore
    health = None  # type: ignore
    health_pb2 = None  # type: ignore

try:  # pragma: no cover - tracing optional in unit tests
    from opentelemetry import trace
except Exception:  # pragma: no cover
    trace = None  # type: ignore

logger = structlog.get_logger(__name__)


@dataclass
class GrpcServiceState:
    """Tracks readiness and exposes health reporting."""

    service_name: str
    health_servicer: Any = field(default=None)
    ready: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        if health is not None:
            self.health_servicer = health.HealthServicer()
            self.health_servicer.set(
                self.service_name,
                health_pb2.HealthCheckResponse.NOT_SERVING,  # type: ignore[attr-defined]
            )

    def set_ready(self) -> None:
        self.ready.set()
        if self.health_servicer is not None:
            self.health_servicer.set(
                self.service_name,
                health_pb2.HealthCheckResponse.SERVING,  # type: ignore[attr-defined]
            )
        logger.info("grpc.service.ready", service=self.service_name)

    def set_not_ready(self, reason: str | None = None) -> None:
        self.ready.clear()
        if self.health_servicer is not None:
            self.health_servicer.set(
                self.service_name,
                health_pb2.HealthCheckResponse.NOT_SERVING,  # type: ignore[attr-defined]
            )
        logger.warning("grpc.service.not_ready", service=self.service_name, reason=reason)

    async def wait_until_ready(self, timeout: float | None = None) -> None:
        if timeout is None:
            await self.ready.wait()
            return
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self.ready.is_set():
                return
            await asyncio.sleep(0.05)
        raise TimeoutError(f"Service {self.service_name} not ready within {timeout} seconds")


class UnaryUnaryTracingInterceptor(ServerInterceptor):
    """Adds OpenTelemetry spans for unary-unary RPCs."""

    def __init__(self, service_name: str) -> None:
        self.service_name = service_name

    async def intercept_service(
        self,
        continuation: Callable[[HandlerCallDetails], Awaitable[Any]],
        handler_call_details: HandlerCallDetails,
    ) -> Any:
        if trace is None:
            return await continuation(handler_call_details)
        tracer = trace.get_tracer(__name__)
        method = getattr(handler_call_details, "method", "unknown")
        with tracer.start_as_current_span(method) as span:  # type: ignore[attr-defined]
            span.set_attribute("rpc.system", "grpc")
            span.set_attribute("rpc.service", self.service_name)
            return await continuation(handler_call_details)


class UnaryUnaryLoggingInterceptor(ServerInterceptor):
    """Structured logging interceptor for unary-unary RPCs."""

    def __init__(self, service_name: str) -> None:
        self.service_name = service_name

    async def intercept_service(
        self,
        continuation: Callable[[HandlerCallDetails], Awaitable[Any]],
        handler_call_details: HandlerCallDetails,
    ) -> Any:
        method = getattr(handler_call_details, "method", "unknown")
        logger.info("grpc.request", service=self.service_name, method=method)
        try:
            response = await continuation(handler_call_details)
            logger.info("grpc.response", service=self.service_name, method=method)
            return response
        except Exception as exc:
            logger.exception("grpc.error", service=self.service_name, method=method, error=str(exc))
            raise


__all__ = [
    "GrpcServiceState",
    "UnaryUnaryTracingInterceptor",
    "UnaryUnaryLoggingInterceptor",
]
