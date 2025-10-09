"""Adapter SDK exports."""

from .base import AdapterContext, AdapterResult, BaseAdapter
from .interfaces import PdfAssetManifest, PdfCapableAdapter, PdfManifest
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
from .openalex import OpenAlexAdapter
from .plugins.base import BaseAdapterPlugin, ReadOnlyAdapterPlugin
from .plugins.bootstrap import (
    get_plugin_manager,
    list_adapters_by_domain,
    plugin_framework_enabled,
)
from .plugins.config import (
    AdapterSettings,
    ConfigValidationResult,
    SettingsHotReloader,
    VaultSecretProvider,
    apply_env_overrides,
    migrate_yaml_to_env,
    validate_on_startup,
)
from .plugins.domains.biomedical import (
    BIOMEDICAL_PLUGINS,
    register_biomedical_plugins,
)
from .plugins.domains.financial import FinancialNewsAdapterPlugin
from .plugins.domains.legal import LegalPrecedentAdapterPlugin
from .plugins.errors import AdapterPluginError
from .plugins.manager import AdapterHookSpec, AdapterPluginManager
from .plugins.models import (
    AdapterCostEstimate,
    AdapterDomain,
    AdapterMetadata,
    AdapterRequest,
    AdapterResponse,
    AdapterResponseEnvelope,
    BiomedicalPayload,
    FinancialPayload,
    LegalPayload,
    Pagination,
    ValidationOutcome,
)
from .plugins.pipeline import (
    AdapterExecutionContext,
    AdapterExecutionMetrics,
    AdapterExecutionState,
    AdapterPipeline,
    AdapterPipelineFactory,
    AdapterStage,
    StageResult,
)
from .plugins.resilience import (
    BackoffStrategy,
    CircuitBreaker,
    ResilienceConfig,
    ResilientHTTPClient,
    circuit_breaker,
    rate_limit,
    retry_on_failure,
)
from .plugins.runtime import AdapterExecutionPlan, AdapterInvocationResult
from .pmc import PMCAdapter as PMCAdapterV2
from .semanticscholar import SemanticScholarAdapter as SemanticScholarAdapterV2
from .terminology import (
    ChEMBLAdapter as ChEMBLAdapterV2,
)
from .terminology import (
    ICD11Adapter as ICD11AdapterV2,
)
from .terminology import (
    MeSHAdapter as MeSHAdapterV2,
)
from .terminology import (
    RxNormAdapter as RxNormAdapterV2,
)
from .unpaywall import UnpaywallAdapter as UnpaywallAdapterV2
from .yaml_parser import AdapterConfig, create_adapter_from_config, load_adapter_config

__all__ = [
    "BIOMEDICAL_PLUGINS",
    "AdapterConfig",
    "AdapterContext",
    "AdapterCostEstimate",
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
    "AdapterSettings",
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
    "PdfAssetManifest",
    "PdfCapableAdapter",
    "PdfManifest",
    "PMCAdapter",
    "Pagination",
    "ReadOnlyAdapterPlugin",
    "ResilienceConfig",
    "ResilientHTTPClient",
    "RxNormAdapter",
    "SemanticScholarAdapter",
    "SettingsHotReloader",
    "StageResult",
    "UnpaywallAdapter",
    "ValidationOutcome",
    "VaultSecretProvider",
    "apply_env_overrides",
    "circuit_breaker",
    "create_adapter_from_config",
    "get_plugin_manager",
    "list_adapters_by_domain",
    "load_adapter_config",
    "migrate_yaml_to_env",
    "plugin_framework_enabled",
    "rate_limit",
    "register_biomedical_plugins",
    "retry_on_failure",
    "validate_on_startup",
]
