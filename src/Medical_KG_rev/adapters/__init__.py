"""Adapter SDK exports."""

from .base import AdapterContext, AdapterResult, BaseAdapter
from .biomedical import (
    ChEMBLAdapter,
    ClinicalTrialsAdapter,
    COREAdapter,
    CrossrefAdapter,
    ICD11Adapter,
    MeSHAdapter,
    OpenFDADeviceAdapter,
    OpenFDADrugEventAdapter,
    OpenFDADrugLabelAdapter,
    PMCAdapter,
    RxNormAdapter,
    SemanticScholarAdapter,
    UnpaywallAdapter,
)
from .interfaces import PdfAssetManifest, PdfCapableAdapter, PdfManifest
from .openalex import OpenAlexAdapter
from .plugins.base import BaseAdapterPlugin, ReadOnlyAdapterPlugin

# from .plugins.bootstrap import (
#     SettingsHotReloader,  # Not implemented
#     validate_on_startup,  # Not implemented
# )
from .plugins.config import (
    ConfigValidationResult,
    # AdapterConfig,  # Not implemented
    # AdapterSettings,  # Not implemented
    # ValidationOutcome,  # Not implemented
    # apply_env_overrides,  # Not implemented
    # migrate_yaml_to_env,  # Not implemented
)
from .plugins.domains.biomedical import register_biomedical_plugins

# from .plugins.domains.biomedical import BIOMEDICAL_PLUGINS  # Not implemented
from .plugins.domains.financial import FinancialNewsAdapterPlugin
from .plugins.domains.legal import LegalPrecedentAdapterPlugin
from .plugins.errors import AdapterPluginError
from .plugins.manager import AdapterHookSpec, AdapterPluginManager
from .plugins.models import (
    AdapterDomain,
    # AdapterExecutionContext,  # Not implemented
    # AdapterExecutionMetrics,  # Not implemented
    # AdapterExecutionState,  # Not implemented
    AdapterMetadata,
    # AdapterPipeline,  # Not implemented
    AdapterRequest,
    AdapterResponse,
    AdapterResponseEnvelope,
    # AdapterSettings,  # Not implemented
    # AdapterStage,  # Not implemented
    # BackoffStrategy,  # Not implemented
    BiomedicalPayload,
    # CircuitBreaker,  # Not implemented
    FinancialPayload,
    LegalPayload,
    Pagination,
    # ResilienceConfig,  # Not implemented
    # ResilientHTTPClient,  # Not implemented
    # StageResult,  # Not implemented
    # VaultSecretProvider,  # Not implemented
    # circuit_breaker,  # Not implemented
    # get_plugin_manager,  # Not implemented
    # list_adapters_by_domain,  # Not implemented
    # plugin_framework_enabled,  # Not implemented
    # rate_limit,  # Not implemented
    # retry_on_failure,  # Not implemented
)
from .plugins.pipeline import AdapterPipelineFactory
from .plugins.resilience import (
    CircuitBreaker,
    ResilienceConfig,
    # ResilientHTTPClient,  # Not implemented
    # circuit_breaker,  # Not implemented
    # retry_on_failure,  # Not implemented
)
from .plugins.runtime import AdapterExecutionPlan, AdapterInvocationResult
from .yaml_parser import load_adapter_config

# from .yaml_parser import AdapterConfig  # Not implemented
# from .yaml_parser import create_adapter_from_config  # Not implemented - YAMLConfiguredAdapter missing

__all__ = [
    # "BIOMEDICAL_PLUGINS",  # Not implemented
    # "AdapterConfig",  # Not implemented
    "AdapterContext",
    "AdapterDomain",
    "AdapterExecutionContext",
    "AdapterExecutionMetrics",
    "AdapterExecutionPlan",
    "AdapterExecutionState",
    "AdapterHookSpec",
    "AdapterInvocationResult",
    "AdapterMetadata",
    "AdapterPipeline",
    "AdapterPipelineFactory",
    "AdapterPluginError",
    "AdapterPluginManager",
    "AdapterRequest",
    "AdapterResponse",
    "AdapterResponseEnvelope",
    "AdapterResult",
    # "AdapterSettings",  # Not implemented
    "AdapterStage",
    "BackoffStrategy",
    "BaseAdapter",
    "BaseAdapterPlugin",
    "BiomedicalPayload",
    "COREAdapter",
    "ChEMBLAdapter",
    "CircuitBreaker",
    "ClinicalTrialsAdapter",
    "ConfigValidationResult",
    "CrossrefAdapter",
    "FinancialNewsAdapterPlugin",
    "FinancialPayload",
    "ICD11Adapter",
    "LegalPayload",
    "LegalPrecedentAdapterPlugin",
    "MeSHAdapter",
    "OpenAlexAdapter",
    "OpenFDADeviceAdapter",
    "OpenFDADrugEventAdapter",
    "OpenFDADrugLabelAdapter",
    "PMCAdapter",
    "Pagination",
    "PdfAssetManifest",
    "PdfCapableAdapter",
    "PdfManifest",
    "ReadOnlyAdapterPlugin",
    "ResilienceConfig",
    "ResilientHTTPClient",
    "RxNormAdapter",
    "SemanticScholarAdapter",
    # "SettingsHotReloader",  # Not implemented
    "StageResult",
    "UnpaywallAdapter",
    # "ValidationOutcome",  # Not implemented
    "VaultSecretProvider",
    # "apply_env_overrides",  # Not implemented
    "circuit_breaker",
    # "create_adapter_from_config",  # Not implemented
    "get_plugin_manager",
    "list_adapters_by_domain",
    "load_adapter_config",
    # "migrate_yaml_to_env",  # Not implemented
    "plugin_framework_enabled",
    "rate_limit",
    "register_biomedical_plugins",
    "retry_on_failure",
    # "validate_on_startup",  # Not implemented
]
