"""Observability helpers for FastAPI and background services."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:  # pragma: no cover - import hints only
    from fastapi import FastAPI
    from Medical_KG_rev.config.settings import AppSettings

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from fastapi import FastAPI
else:
    FastAPI = Any  # type: ignore[misc,assignment]

__all__ = ["setup_observability"]

logger = structlog.get_logger(__name__)
_FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


def setup_observability(app: "FastAPI", settings: "AppSettings") -> None:
    """Configure logging, tracing, metrics, and error tracking for the app."""

    if not _FASTAPI_AVAILABLE:
        logger.warning(
            "observability.fastapi.unavailable",
            message="FastAPI is not installed; observability setup skipped",
        )
        return

    configure_logging(settings=settings.observability.logging)
    configure_tracing(settings.service_name, settings.telemetry)
    initialise_sentry(settings)
    instrument_application(app, settings)
    register_metrics(app, settings)
