"""OpenTelemetry instrumentation helpers."""

from __future__ import annotations

from types import ModuleType
from typing import Any, Awaitable, Callable, TYPE_CHECKING, cast

from opentelemetry import trace

otel_fastapi: ModuleType | None
try:
    import opentelemetry.instrumentation.fastapi as otel_fastapi
except ImportError:  # pragma: no cover - optional dependency
    otel_fastapi = None

otel_grpc: ModuleType | None
try:
    import opentelemetry.instrumentation.grpc as otel_grpc
except (ImportError, AttributeError):  # pragma: no cover - optional dependency
    otel_grpc = None

from Medical_KG_rev.config.settings import AppSettings

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor as FastAPIInstrumentorType
    from opentelemetry.instrumentation.grpc import (
        GrpcInstrumentorClient as GrpcInstrumentorClientType,
        GrpcInstrumentorServer as GrpcInstrumentorServerType,
    )
else:
    FastAPIInstrumentorType = Any
    GrpcInstrumentorClientType = Any
    GrpcInstrumentorServerType = Any

fastapi_instrumentor: FastAPIInstrumentorType | None = (
    cast(FastAPIInstrumentorType, otel_fastapi.FastAPIInstrumentor) if otel_fastapi else None
)
grpc_instrumentor_client: GrpcInstrumentorClientType | None = (
    cast(GrpcInstrumentorClientType, otel_grpc.GrpcInstrumentorClient) if otel_grpc else None
)
grpc_instrumentor_server: GrpcInstrumentorServerType | None = (
    cast(GrpcInstrumentorServerType, otel_grpc.GrpcInstrumentorServer) if otel_grpc else None
)

_GRPC_INSTRUMENTATION_AVAILABLE = grpc_instrumentor_client is not None and grpc_instrumentor_server is not None
_FASTAPI_INSTRUMENTED = False
_GRPC_INSTRUMENTED = False


def configure_tracing(service_name: str, telemetry: Any) -> None:
    """Configure OpenTelemetry tracing.

    The full tracing setup is handled by :func:`utils.logging.configure_tracing`.
    This helper exists for backwards compatibility and may be expanded in the
    future when additional telemetry settings are introduced.
    """


def instrument_application(app: Any, settings: AppSettings) -> None:
    """Instrument FastAPI and gRPC services when instrumentation is available."""

    global _FASTAPI_INSTRUMENTED, _GRPC_INSTRUMENTED

    if fastapi_instrumentor is not None and not _FASTAPI_INSTRUMENTED:
        fastapi_instrumentor.instrument_app(app, excluded_urls=settings.observability.metrics.path)
        _FASTAPI_INSTRUMENTED = True

    if _GRPC_INSTRUMENTATION_AVAILABLE and not _GRPC_INSTRUMENTED:
        server = cast(Callable[[], Any], grpc_instrumentor_server)
        client = cast(Callable[[], Any], grpc_instrumentor_client)
        server().instrument()
        client().instrument()
        _GRPC_INSTRUMENTED = True

    async def tracing_context_middleware(
        request: Any, call_next: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        response = await call_next(request)
        span = trace.get_current_span()
        correlation_id = getattr(request.state, "correlation_id", None)
        if span is not None and correlation_id:
            span.set_attribute("correlation_id", correlation_id)
        return response

    middleware = getattr(app, "middleware")
    middleware("http")(tracing_context_middleware)
