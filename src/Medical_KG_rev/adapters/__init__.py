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
from .clinicaltrials import ClinicalTrialsAdapter as ClinicalTrialsAdapterV2
from .core import COREAdapter as COREAdapterV2
from .crossref import CrossrefAdapter as CrossrefAdapterV2
from .openalex import OpenAlexAdapter
from .openfda import (
    OpenFDADeviceAdapter as OpenFDADeviceAdapterV2,
)
from .openfda import (
    OpenFDADrugEventAdapter as OpenFDADrugEventAdapterV2,
)
from .openfda import (
    OpenFDADrugLabelAdapter as OpenFDADrugLabelAdapterV2,
)
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
    "AdapterConfig",
    "AdapterContext",
    "AdapterResult",
    "BaseAdapter",
    "COREAdapter",
    "ChEMBLAdapter",
    "ClinicalTrialsAdapter",
    "CrossrefAdapter",
    "AdapterHookSpec",
    "AdapterPluginError",
    "AdapterPluginManager",
    "AdapterExecutionContext",
    "AdapterExecutionMetrics",
    "AdapterExecutionPlan",
    "AdapterExecutionState",
    "AdapterInvocationResult",
    "AdapterPipeline",
    "AdapterPipelineFactory",
    "AdapterStage",
    "StageResult",
    "AdapterCostEstimate",
    "AdapterDomain",
    "AdapterMetadata",
    "AdapterRequest",
    "AdapterResponse",
    "AdapterResponseEnvelope",
    "AdapterSettings",
    "BackoffStrategy",
    "BaseAdapterPlugin",
    "ICD11Adapter",
    "CircuitBreaker",
    "MeSHAdapter",
    "OpenAlexAdapter",
    "OpenFDADeviceAdapter",
    "OpenFDADrugEventAdapter",
    "OpenFDADrugLabelAdapter",
    "PMCAdapter",
    "Pagination",
    "RxNormAdapter",
    "ResilienceConfig",
    "ResilientHTTPClient",
    "SettingsHotReloader",
    "SemanticScholarAdapter",
    "UnpaywallAdapter",
    "create_adapter_from_config",
    "circuit_breaker",
    "load_adapter_config",
    "migrate_yaml_to_env",
    "rate_limit",
    "retry_on_failure",
    "validate_on_startup",
    "VaultSecretProvider",
    "apply_env_overrides",
    "BiomedicalPayload",
    "FinancialPayload",
    "FinancialNewsAdapterPlugin",
    "LegalPayload",
    "LegalPrecedentAdapterPlugin",
    "ValidationOutcome",
    "ConfigValidationResult",
    "ReadOnlyAdapterPlugin",
    "get_plugin_manager",
    "plugin_framework_enabled",
    "list_adapters_by_domain",
    "BIOMEDICAL_PLUGINS",
    "register_biomedical_plugins",
]
