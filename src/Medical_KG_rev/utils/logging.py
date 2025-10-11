"""Logging configuration helpers with OpenTelemetry and Structlog integration.

Key Responsibilities:
    - Configure standard library logging with JSON formatting and field scrubbing
    - Integrate Structlog, OpenTelemetry tracing, and correlation IDs
    - Expose helpers for binding request correlation identifiers

Collaborators:
    - Upstream: Gateway and service entry-points call configuration helpers
    - Downstream: Relies on ``logging``, ``structlog``, and OpenTelemetry SDKs

Side Effects:
    - Configures global logging handlers and tracing providers
    - Binds correlation IDs via context variables

Thread Safety:
    - Logging configuration should be invoked once during process startup
    - Correlation ID helpers rely on ``contextvars`` and are safe for async use

Performance Characteristics:
    - JSON formatting introduces minimal serialization overhead
    - Tracing exporters may add network latency depending on configuration
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterable
from contextvars import ContextVar, Token
from typing import Any, Callable
from urllib.parse import urlparse

import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
)
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from Medical_KG_rev.config import TelemetrySettings
from Medical_KG_rev.config.settings import LoggingSettings

# ==============================================================================
# CONTEXT VARIABLES
# ==============================================================================

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)

# ==============================================================================
# FORMATTERS
# ==============================================================================


class JsonFormatter(logging.Formatter):
    """Formats log records as single line JSON objects."""

    def __init__(self, *, scrub_fields: Iterable[str] | None = None) -> None:
        """Initialise formatter with optional sensitive field scrubbing.

        Args:
            scrub_fields: Iterable of field names (case-insensitive) whose values
                should be replaced with ``***`` in log output.
        """
        super().__init__(datefmt="%Y-%m-%dT%H:%M:%S%z")
        self._scrub_fields = {field.lower() for field in scrub_fields or ()}

    def _scrub(self, value: object) -> object:
        """Recursively scrub values in dictionaries and lists."""
        if isinstance(value, dict):
            return {
                k: self._scrub(v) if k.lower() not in self._scrub_fields else "***"
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [self._scrub(item) for item in value]
        return value

    def format(self, record: logging.LogRecord) -> str:
        """Serialise a log record into a JSON string."""
        payload: dict[str, Any] = {
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


# ==============================================================================
# STRUCTLOG PROCESSORS
# ==============================================================================


def _structlog_scrubber(
    scrub_fields: Iterable[str] | None,
) -> Callable[[Any, str, dict[str, Any]], dict[str, Any]]:
    """Create a Structlog processor that scrubs sensitive fields.

    Args:
        scrub_fields: Iterable of field names to obfuscate.

    Returns:
        Structlog processor that replaces configured fields with ``***`` and
        injects the correlation ID when present.
    """
    lower_fields = {field.lower() for field in scrub_fields or ()}

    def processor(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
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
    """Configure global logging for the application.

    Args:
        level: Optional logging level or level name. When ``settings`` is
            provided this argument is ignored.
        settings: Optional logging settings object providing level and scrub
            configuration.

    Note:
        Calling this function reconfigures the root logger and should therefore
        happen once during application startup.
    """
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
        module_attr = getattr(existing.__class__, "__module__", "")
        module: str = module_attr if isinstance(module_attr, str) else ""
        if module.startswith("_pytest."):
            existing.setFormatter(JsonFormatter(scrub_fields=scrub_fields))
            preserved_handlers.append(existing)

    logging.basicConfig(
        level=level_value,
        handlers=[*preserved_handlers, handler],
        force=True,
    )

    if structlog:
        processors: list[Any] = [
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


# ==============================================================================
# CONFIGURATION HELPERS
# ==============================================================================


def configure_tracing(service_name: str, telemetry: TelemetrySettings) -> None:
    """Configure OpenTelemetry tracing provider.

    Args:
        service_name: Name reported to tracing backends.
        telemetry: Telemetry settings describing exporter type, endpoint, and
            sampling configuration.
    """
    resource = Resource(attributes={"service.name": service_name})
    sampler = TraceIdRatioBased(telemetry.sample_ratio)
    provider = TracerProvider(resource=resource, sampler=sampler)

    exporter: SpanExporter | None = None
    target = telemetry.exporter.lower()
    if target == "jaeger" and JaegerExporter is not None:
        host = "localhost"
        port = 6831
        if telemetry.endpoint:
            parsed = urlparse(telemetry.endpoint)
            host = parsed.hostname or telemetry.endpoint
            port = parsed.port or port
        exporter = JaegerExporter(agent_host_name=host, agent_port=port)
    elif target == "otlp" and OTLPSpanExporter is not None:
        exporter = OTLPSpanExporter(endpoint=telemetry.endpoint) if telemetry.endpoint else OTLPSpanExporter()
    else:
        exporter = ConsoleSpanExporter()

    if exporter is not None:
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


# ==============================================================================
# CORRELATION ID HELPERS
# ==============================================================================


def bind_correlation_id(value: str) -> Token[str | None]:
    """Bind a correlation identifier to the current execution context.

    Args:
        value: Correlation identifier to associate with the current context.

    Returns:
        Context variable token that can be used to restore the previous value.
    """
    token = _correlation_id.set(value)
    if structlog:
        structlog.contextvars.bind_contextvars(correlation_id=value)
    return token


def reset_correlation_id(token: Token[str | None] | None) -> None:
    """Reset the correlation identifier context.

    Args:
        token: Token returned by :func:`bind_correlation_id` or ``None``.
    """
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
    """Return a configured logger with the given name."""
    return logging.getLogger(name)
