"""Logging helpers with OpenTelemetry integration."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterable
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING
from urllib.parse import urlparse

try:  # Optional structlog dependency
    import structlog
except Exception:  # pragma: no cover - structlog may not be installed in lightweight envs
    structlog = None  # type: ignore

try:  # Optional OpenTelemetry dependency
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
except Exception:  # pragma: no cover - tracing optional in lightweight environments
    trace = None  # type: ignore
    Resource = None  # type: ignore
    TracerProvider = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
    ConsoleSpanExporter = None  # type: ignore
    TraceIdRatioBased = None  # type: ignore

try:  # Optional Jaeger exporter dependency
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
except Exception:  # pragma: no cover - Jaeger exporter optional
    JaegerExporter = None  # type: ignore

try:  # Optional OTLP exporter dependency
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except Exception:  # pragma: no cover - OTLP exporter optional
    OTLPSpanExporter = None  # type: ignore

try:  # pragma: no cover - optional settings dependency
    from Medical_KG_rev.config import TelemetrySettings
except Exception:  # pragma: no cover - fallback for lightweight environments

    class TelemetrySettings:  # type: ignore[override]
        """Fallback telemetry settings used when the config package is minimal."""

        sample_ratio: float = 1.0
        exporter: str = "console"
        endpoint: str | None = None


if TYPE_CHECKING:  # pragma: no cover - typing only
    from Medical_KG_rev.config.settings import LoggingSettings


_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


class JsonFormatter(logging.Formatter):
    """Formats log records as single line JSON objects."""

    def __init__(self, *, scrub_fields: Iterable[str] | None = None) -> None:
        super().__init__(datefmt="%Y-%m-%dT%H:%M:%S%z")
        self._scrub_fields = {field.lower() for field in scrub_fields or ()}

    def _scrub(self, value: object) -> object:
        if isinstance(value, dict):
            return {
                k: self._scrub(v) if k.lower() not in self._scrub_fields else "***"
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [self._scrub(item) for item in value]
        return value

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }

        correlation_id = _correlation_id.get()
        if correlation_id:
            payload["correlation_id"] = correlation_id

        for key, value in record.__dict__.items():
            if key in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
            }:
                continue
            if key.lower() in self._scrub_fields:
                payload[key] = "***"
            else:
                payload[key] = self._scrub(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, sort_keys=True)


def _structlog_scrubber(scrub_fields: Iterable[str] | None):
    lower_fields = {field.lower() for field in scrub_fields or ()}

    def processor(_, __, event_dict):  # type: ignore[override]
        correlation_id = _correlation_id.get()
        if correlation_id:
            event_dict.setdefault("correlation_id", correlation_id)
        for key in list(event_dict.keys()):
            if key.lower() in lower_fields:
                event_dict[key] = "***"
        return event_dict

    return processor


def configure_logging(
    level: int | str | None = None,
    *,
    settings: LoggingSettings | None = None,
) -> None:
    """Configure global logging for the application."""
    scrub_fields: Iterable[str] | None = None
    if settings is not None:
        level = settings.level
        scrub_fields = settings.scrub_fields

    if isinstance(level, str):
        level_value = getattr(logging, level.upper(), logging.INFO)
    elif isinstance(level, int):
        level_value = level
    else:
        level_value = logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    json_formatter = JsonFormatter(scrub_fields=scrub_fields)
    handler.setFormatter(json_formatter)

    root_logger = logging.getLogger()
    preserved_handlers: list[logging.Handler] = []
    for existing in root_logger.handlers:
        module = getattr(existing.__class__, "__module__", "")
        if module.startswith("_pytest."):
            existing.setFormatter(JsonFormatter(scrub_fields=scrub_fields))
            preserved_handlers.append(existing)

    logging.basicConfig(
        level=level_value,
        handlers=[*preserved_handlers, handler],
        force=True,
    )

    if structlog:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _structlog_scrubber(scrub_fields),
            structlog.processors.JSONRenderer(sort_keys=True),
        ]
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(level_value),
            logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
            cache_logger_on_first_use=True,
        )


def configure_tracing(service_name: str, telemetry: TelemetrySettings) -> None:
    """Configure OpenTelemetry tracing provider."""
    if trace is None or TracerProvider is None or Resource is None or TraceIdRatioBased is None:
        logging.getLogger(__name__).warning(
            "telemetry.opentelemetry.unavailable",
            message="OpenTelemetry not installed; tracing disabled",
        )
        return

    resource = Resource(attributes={"service.name": service_name})
    sampler = TraceIdRatioBased(telemetry.sample_ratio)
    provider = TracerProvider(resource=resource, sampler=sampler)

    exporter = ConsoleSpanExporter() if ConsoleSpanExporter is not None else None
    if telemetry.exporter.lower() == "jaeger" and JaegerExporter is not None:
        host = "localhost"
        port = 6831
        if telemetry.endpoint:
            parsed = urlparse(telemetry.endpoint)
            host = parsed.hostname or telemetry.endpoint
            port = parsed.port or port
        exporter = JaegerExporter(agent_host_name=host, agent_port=port)
    elif telemetry.exporter.lower() == "otlp" and OTLPSpanExporter is not None:
        if telemetry.endpoint:
            exporter = OTLPSpanExporter(endpoint=telemetry.endpoint)
        else:  # pragma: no cover - default constructor handles env based config
            exporter = OTLPSpanExporter()

    if exporter is not None and BatchSpanProcessor is not None:
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)


def bind_correlation_id(value: str) -> Token:
    """Bind a correlation identifier to the current execution context."""
    token = _correlation_id.set(value)
    if structlog:
        structlog.contextvars.bind_contextvars(correlation_id=value)
    return token


def reset_correlation_id(token: Token | None) -> None:
    """Reset the correlation identifier context."""
    if token is not None:
        _correlation_id.reset(token)
    if structlog:
        try:
            structlog.contextvars.unbind_contextvars("correlation_id")
        except LookupError:  # pragma: no cover - defensive
            pass


def get_correlation_id() -> str | None:
    """Return the currently bound correlation identifier, if any."""
    return _correlation_id.get()


def get_logger(name: str) -> logging.Logger:
    """Helper to fetch configured logger."""
    return logging.getLogger(name)
