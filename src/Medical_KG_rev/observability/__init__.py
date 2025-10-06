"""Observability helpers for FastAPI and background services."""

from __future__ import annotations

from fastapi import FastAPI

from Medical_KG_rev.config.settings import AppSettings
from Medical_KG_rev.utils.logging import configure_logging, configure_tracing

from .metrics import register_metrics
from .sentry import initialise_sentry
from .tracing import instrument_application

__all__ = ["setup_observability"]


def setup_observability(app: FastAPI, settings: AppSettings) -> None:
    """Configure logging, tracing, metrics, and error tracking for the app."""

    configure_logging(settings=settings.observability.logging)
    configure_tracing(settings.service_name, settings.telemetry)
    initialise_sentry(settings)
    instrument_application(app, settings)
    register_metrics(app, settings)
