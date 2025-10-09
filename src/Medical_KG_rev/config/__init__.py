"""Lightweight configuration package exports."""

from __future__ import annotations

from .domains import DomainConfig, DomainRegistry
from .pyserini_config import (
    DEFAULT_PYSERINI_CONFIG,
    PyseriniConfig,
    PyseriniExpansionConfig,
    PyseriniExpansionSideConfig,
    PyseriniModelConfig,
    PyseriniOpenSearchConfig,
    PyseriniServiceConfig,
    load_pyserini_config,
)
from .vllm_config import (
    DEFAULT_VLLM_CONFIG,
    VLLMBatchingConfig,
    VLLMConfig,
    VLLMHealthCheckConfig,
    VLLMLoggingConfig,
    VLLMModelConfig,
    VLLMServiceConfig,
    load_vllm_config,
)

__all__ = [
    "DEFAULT_PYSERINI_CONFIG",
    "DEFAULT_VLLM_CONFIG",
    "DomainConfig",
    "DomainRegistry",
    "PyseriniConfig",
    "PyseriniExpansionConfig",
    "PyseriniExpansionSideConfig",
    "PyseriniModelConfig",
    "PyseriniOpenSearchConfig",
    "PyseriniServiceConfig",
    "VLLMBatchingConfig",
    "VLLMConfig",
    "VLLMHealthCheckConfig",
    "VLLMLoggingConfig",
    "VLLMModelConfig",
    "VLLMServiceConfig",
    "load_pyserini_config",
    "load_vllm_config",
]

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
