import logging

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from Medical_KG_rev.config import LoggingSettings, TelemetrySettings
from Medical_KG_rev.utils.logging import (
    bind_correlation_id,
    configure_logging,
    configure_tracing,
    get_logger,
    reset_correlation_id,
)


def test_configure_logging_sets_handler(caplog):
    configure_logging(level=logging.DEBUG)
    logger = get_logger("test")
    logger.info("hello", extra={"extra": {"key": "value"}})
    assert logger.name == "test"


def test_configure_tracing_console_exporter():
    telemetry = TelemetrySettings(exporter="console")
    configure_tracing("service", telemetry)
    assert isinstance(trace.get_tracer_provider(), TracerProvider)


def test_structured_logging_includes_correlation_id(caplog):
    settings = LoggingSettings(scrub_fields=["token"])
    configure_logging(settings=settings)
    token = bind_correlation_id("corr-123")
    logger = get_logger("observability")
    logger.info("processed", extra={"token": "super-secret", "detail": "ok"})
    reset_correlation_id(token)
    assert '"correlation_id": "corr-123"' in caplog.text
    assert '"token": "***"' in caplog.text
