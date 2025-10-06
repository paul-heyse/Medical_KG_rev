"""Logging helpers with OpenTelemetry integration."""
from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

try:  # Optional OTLP exporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except Exception:  # pragma: no cover - optional dependency
    OTLPSpanExporter = None  # type: ignore

from Medical_KG_rev.config import TelemetrySettings


class JsonFormatter(logging.Formatter):
    """Formats log records as single line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        payload.update(getattr(record, "extra", {}))
        return json.dumps(payload, sort_keys=True)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure global logging for the application."""

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler])


def configure_tracing(service_name: str, telemetry: TelemetrySettings) -> None:
    """Configure OpenTelemetry tracing provider."""

    resource = Resource(attributes={"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = ConsoleSpanExporter()
    if telemetry.exporter == "otlp" and OTLPSpanExporter is not None:
        exporter = OTLPSpanExporter(endpoint=telemetry.endpoint)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)


def get_logger(name: str) -> logging.Logger:
    """Helper to fetch configured logger."""

    return logging.getLogger(name)
