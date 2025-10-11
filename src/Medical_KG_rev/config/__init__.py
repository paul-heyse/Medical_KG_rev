"""Lightweight configuration package exports."""

from __future__ import annotations

from .docling_config import (
    DEFAULT_CONFIG_PATH as DEFAULT_DOCLING_CONFIG_PATH,
)
from .docling_config import (
    DEFAULT_MODEL_NAME as DEFAULT_DOCLING_MODEL_NAME,
)
from .docling_config import (
    DEFAULT_MODEL_PATH as DEFAULT_DOCLING_MODEL_PATH,
)
from .docling_config import (
    DoclingVLMConfig,
)
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
from .retrieval_config import (
    DEFAULT_RETRIEVAL_CONFIG_PATH,
    BM25Config,
    FusionConfig,
    Qwen3Config,
    RetrievalConfig,
    SPLADEConfig,
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
    "DEFAULT_DOCLING_CONFIG_PATH",
    "DEFAULT_DOCLING_MODEL_NAME",
    "DEFAULT_DOCLING_MODEL_PATH",
    "DEFAULT_PYSERINI_CONFIG",
    "DEFAULT_RETRIEVAL_CONFIG_PATH",
    "DEFAULT_VLLM_CONFIG",
    "BM25Config",
    "DoclingVLMConfig",
    "DomainConfig",
    "DomainRegistry",
    "FusionConfig",
    "PyseriniConfig",
    "PyseriniExpansionConfig",
    "PyseriniExpansionSideConfig",
    "PyseriniModelConfig",
    "PyseriniOpenSearchConfig",
    "PyseriniServiceConfig",
    "Qwen3Config",
    "RetrievalConfig",
    "SPLADEConfig",
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
