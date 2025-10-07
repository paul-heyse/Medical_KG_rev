"""OpenTelemetry instrumentation helpers."""

from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore
from opentelemetry.instrumentation.grpc import (  # type: ignore
    GrpcInstrumentorClient,
    GrpcInstrumentorServer,
)

from Medical_KG_rev.config.settings import AppSettings

_FASTAPI_INSTRUMENTED = False
_GRPC_INSTRUMENTED = False


def configure_tracing(service_name: str, telemetry: Any) -> None:
    """Configure OpenTelemetry tracing."""
    # Configure tracing based on telemetry settings
    pass


def instrument_application(app, settings: AppSettings) -> None:
    global _FASTAPI_INSTRUMENTED, _GRPC_INSTRUMENTED

    if not _FASTAPI_INSTRUMENTED:
        FastAPIInstrumentor.instrument_app(app, excluded_urls=settings.observability.metrics.path)
        _FASTAPI_INSTRUMENTED = True

    if not _GRPC_INSTRUMENTED:
        GrpcInstrumentorServer().instrument()
        GrpcInstrumentorClient().instrument()
        _GRPC_INSTRUMENTED = True

    @app.middleware("http")
    async def tracing_context_middleware(request, call_next):
        response = await call_next(request)
        span = trace.get_current_span()
        correlation_id = getattr(request.state, "correlation_id", None)
        if span is not None and correlation_id:
            span.set_attribute("correlation_id", correlation_id)
        return response
