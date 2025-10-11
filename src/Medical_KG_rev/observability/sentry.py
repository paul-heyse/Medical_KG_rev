"""Sentry error tracking integration."""

from __future__ import annotations

import logging

import sentry_sdk
from sentry_sdk.integrations import Integration
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

from Medical_KG_rev.config.settings import AppSettings

_SENTRY_INITIALISED = False


def initialise_sentry(settings: AppSettings) -> None:
    global _SENTRY_INITIALISED

    if _SENTRY_INITIALISED or sentry_sdk is None:
        return

    sentry_settings = settings.observability.sentry
    if not sentry_settings.dsn:
        return

    integrations: list[Integration] = [FastApiIntegration(transaction_style="endpoint")]
    integrations.append(LoggingIntegration(level=logging.INFO, event_level=logging.ERROR))

    sentry_sdk.init(
        dsn=sentry_settings.dsn,
        environment=sentry_settings.environment or settings.environment.value,
        traces_sample_rate=sentry_settings.traces_sample_rate,
        send_default_pii=sentry_settings.send_default_pii,
        integrations=integrations,
    )

    _SENTRY_INITIALISED = True
