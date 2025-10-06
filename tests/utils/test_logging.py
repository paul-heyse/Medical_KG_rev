import logging

from opentelemetry import trace

from Medical_KG_rev.config import TelemetrySettings
from Medical_KG_rev.utils.logging import configure_logging, configure_tracing, get_logger


def test_configure_logging_sets_handler(caplog):
    configure_logging(level=logging.DEBUG)
    logger = get_logger("test")
    logger.info("hello", extra={"extra": {"key": "value"}})
    assert logger.name == "test"


def test_configure_tracing_console_exporter():
    telemetry = TelemetrySettings(exporter="console")
    previous = trace.get_tracer_provider()
    configure_tracing("service", telemetry)
    assert trace.get_tracer_provider() is not previous
