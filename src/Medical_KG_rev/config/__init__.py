"""Lightweight configuration package exports."""

from __future__ import annotations

from .domains import DomainConfig, DomainRegistry

__all__ = ["DomainConfig", "DomainRegistry"]

try:  # pragma: no cover - optional settings dependency
    from .settings import (  # type: ignore[import-not-found]
        AppSettings,
        Environment,
        FeatureFlagSettings,
        LoggingSettings,
        ObservabilitySettings,
        RerankingSettings,
        SecretResolver,
        TelemetrySettings,
        get_settings,
        load_settings,
        migrate_reranking_config,
    )

    __all__.extend(
        [
            "AppSettings",
            "Environment",
            "FeatureFlagSettings",
            "LoggingSettings",
            "ObservabilitySettings",
            "RerankingSettings",
            "SecretResolver",
            "TelemetrySettings",
            "get_settings",
            "load_settings",
            "migrate_reranking_config",
        ]
    )
except ModuleNotFoundError:  # pragma: no cover - minimal environments
    # Settings module depends on pydantic. Skip optional exports when dependency missing.
    pass
