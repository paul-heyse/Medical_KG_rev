# Design Document: Adapter Plugin Framework

## Context

The Medical_KG_rev system currently supports 11+ biomedical data sources through a custom adapter pattern. The architecture has evolved organically, leading to:

- **Registry Fragmentation**: `ADAPTERS` dictionary in `adapters/registry.py` manually maps source names to adapter classes
- **Configuration Inconsistency**: Mix of YAML files (`adapters/config/*.yaml`) and Python-defined configs
- **Resilience Duplication**: Each adapter implements its own retry logic (e.g., `ResilientHTTPAdapter` vs `BaseAdapter.http_client`)
- **Domain Coupling**: All adapters assume biomedical data structures (Clinical trials, literature, ontologies)

The system needs to scale to:

- **50+ data sources** across 3 domains (biomedical, financial, legal)
- **Multi-tenant deployments** with per-tenant adapter configurations
- **Third-party adapter development** by external contributors
- **Dynamic adapter loading** without code changes

## Goals / Non-Goals

### Goals

1. **Standardize Adapter Lifecycle**: Single contract for fetch → parse → validate → transform
2. **Enable Plugin Ecosystem**: Third-party adapters installable via pip with automatic discovery
3. **Centralize Resilience**: Retry, rate limiting, circuit breaking as framework concerns
4. **Support Multi-Domain**: Biomedical, financial, legal adapters share common infrastructure
5. **Improve Observability**: Adapter-level metrics, health checks, and cost estimation
6. **Simplify Configuration**: Environment-based settings with secret management

### Non-Goals

1. **Not changing ingestion pipeline**: Auto/two-phase pipeline stages remain unchanged
2. **Not modifying data models**: Document IR and domain overlays are out of scope
3. **Not replacing HTTP clients**: Keep httpx/aiohttp, only wrap with resilience
4. **Not implementing new adapters**: Focus on framework, example adapters only

## Decisions

### Decision 1: Pluggy vs importlib.metadata

**Choice**: Use **Pluggy** (<https://pluggy.readthedocs.io/>) version 1.3.0+ with `importlib.metadata.entry_points` for hybrid approach

**Rationale**:

- **Pluggy** is the industry-standard plugin framework used by:
  - **pytest** - Python's most popular testing framework
  - **tox** - Python testing automation tool
  - **devpi** - PyPI server and packaging ecosystem
- Provides hook specifications (`@hookspec`) and lifecycle management
- Built-in support for hook implementation markers (`@hookimpl`)
- Type-safe with full support for type hints and validation
- `importlib.metadata` handles entry point discovery from installed packages
- Hybrid approach allows internal adapters (always loaded) and external adapters (opt-in)
- Supports setuptools, Poetry, and Flit packaging workflows
- Battle-tested and actively maintained by the Python community

**Alternatives Considered**:

- **Pure importlib.metadata**: Too low-level, requires manual hook management, no plugin lifecycle
- **Stevedore**: Heavy dependency, OpenStack-specific patterns, less widely used
- **Yapsy**: Unmaintained since 2019, lacks type safety and modern Python support
- **SQLAlchemy**: Fundamentally wrong abstraction - it's an ORM for databases, not a plugin system

**Implementation**:

```python
# pyproject.toml
[project.entry-points."medical_kg.adapters"]
clinicaltrials = "Medical_KG_rev.adapters.biomedical.clinicaltrials:ClinicalTrialsAdapter"
openfda = "Medical_KG_rev.adapters.biomedical.openfda:OpenFDAAdapter"

# Plugin manager using Pluggy
import pluggy
from importlib.metadata import entry_points

# Create Pluggy hook markers for this project
hookspec = pluggy.HookspecMarker("medical_kg")
hookimpl = pluggy.HookimplMarker("medical_kg")

class AdapterHookSpec:
    """Pluggy hook specifications for adapter plugins."""

    @hookspec
    def get_metadata(self) -> AdapterMetadata:
        """Return adapter metadata."""

    @hookspec
    def fetch(self, request: AdapterRequest) -> AsyncIterator[RawPayload]:
        """Fetch data from source."""

    @hookspec
    def parse(self, payloads: Iterable[RawPayload]) -> list[Document]:
        """Parse raw data to Document IR."""

# Create Pluggy plugin manager
pm = pluggy.PluginManager("medical_kg")
pm.add_hookspecs(AdapterHookSpec)

# Discover and register adapters via entry points
for ep in entry_points(group="medical_kg.adapters"):
    adapter_cls = ep.load()
    pm.register(adapter_cls())
```

### Decision 2: Tenacity vs Custom Retry Logic

**Choice**: Replace custom retry logic with **Tenacity** (<https://tenacity.readthedocs.io/>) version 8.2.0+

**Rationale**:

- Declarative retry policies: `@retry(stop=stop_after_attempt(3))`
- Built-in backoff strategies: exponential, jitter, adaptive
- Retry statistics and callbacks for observability
- Less code to maintain (remove custom `ResilientHTTPAdapter`)
- Battle-tested library used in production by major projects
- Async/await support for modern Python

**Alternatives Considered**:

- **backoff library**: Simpler but lacks advanced features
- **Custom implementation**: High maintenance burden
- **aiohttp-retry**: HTTP-specific, not reusable for gRPC

**Implementation**:

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

class ResilienceConfig(BaseModel):
    max_attempts: int = 3
    max_wait_seconds: int = 60
    exponential_base: int = 2
    jitter: bool = True

def make_retry_decorator(config: ResilienceConfig):
    return retry(
        stop=stop_after_attempt(config.max_attempts),
        wait=wait_exponential(
            multiplier=1,
            min=1,
            max=config.max_wait_seconds,
            exp_base=config.exponential_base
        ),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
        reraise=True
    )

# Usage in adapter
@make_retry_decorator(ResilienceConfig(max_attempts=5))
async def fetch_with_retry(self, url: str) -> httpx.Response:
    async with httpx.AsyncClient() as client:
        return await client.get(url)
```

### Decision 3: Pydantic Settings vs YAML Config

**Choice**: Migrate to **pydantic-settings** (<https://docs.pydantic.dev/latest/concepts/pydantic_settings/>) version 2.0.0+ with environment variables

**Rationale**:

- Type-safe configuration with automatic validation via Pydantic v2
- 12-factor app compliance (config via environment)
- Secret management via Vault or env vars (no secrets in code)
- Hot-reload support via settings refresh mechanism
- Auto-generated documentation from Pydantic schemas
- JSON Schema export for API documentation

**Alternatives Considered**:

- **Keep YAML**: Requires custom parsing, no type safety
- **dynaconf**: Extra dependency, overlaps with Pydantic
- **python-decouple**: Less powerful than pydantic-settings

**Implementation**:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class AdapterSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MK_ADAPTER_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    # Common settings
    timeout_seconds: int = 30
    rate_limit_per_second: float = 5.0
    retry_max_attempts: int = 3

    # Source-specific settings
    clinicaltrials_api_key: SecretStr | None = None
    openfda_api_key: SecretStr | None = None

# Environment variables
# MK_ADAPTER_TIMEOUT_SECONDS=60
# MK_ADAPTER_CLINICALTRIALS_API_KEY=secret123
```

### Decision 4: Adapter Metadata Schema

**Choice**: Structured metadata using Pydantic with capability tags

**Rationale**:

- Self-documenting adapters (no external registry)
- Enables dynamic adapter selection by orchestrator
- Supports cost estimation for rate limit planning
- Facilitates testing and mocking

**Schema**:

```python
class AdapterCapability(str, Enum):
    FULL_TEXT = "full_text"
    METADATA_ONLY = "metadata_only"
    PDF_SUPPORT = "pdf_support"
    STREAMING = "streaming"
    BATCH = "batch"

class AdapterDomain(str, Enum):
    BIOMEDICAL = "biomedical"
    FINANCIAL = "financial"
    LEGAL = "legal"

class AdapterMetadata(BaseModel):
    name: str
    version: str
    domain: AdapterDomain
    capabilities: set[AdapterCapability]
    supported_identifiers: list[str]  # e.g., ["NCT", "DOI", "PMCID"]
    auth_required: bool
    rate_limit_default: float
    cost_per_request: float | None  # For quota tracking
    health_check_url: str | None

# Example
ClinicalTrialsAdapter.metadata = AdapterMetadata(
    name="clinicaltrials",
    version="2.0.0",
    domain=AdapterDomain.BIOMEDICAL,
    capabilities={AdapterCapability.FULL_TEXT, AdapterCapability.BATCH},
    supported_identifiers=["NCT"],
    auth_required=False,
    rate_limit_default=5.0,
    health_check_url="https://clinicaltrials.gov/api/v2/health"
)
```

## Data Flow

### Before (Current State)

```
User Request → REST Endpoint → Orchestrator → Manual Registry Lookup
    → Adapter.fetch() [custom retry] → Adapter.parse() [custom validation]
    → Pipeline Stage → Kafka Topic
```

### After (Plugin Framework)

```
User Request → REST Endpoint → Orchestrator → Plugin Manager.get_adapter(name, domain)
    → @retry_on_failure Adapter.fetch(AdapterRequest) → @rate_limit
    → Adapter.parse() → Pydantic Validation → AdapterResponse
    → Pipeline Stage → Kafka Topic
```

### Plugin Discovery Flow

```
Application Startup:
1. PluginManager scans entry_points("medical_kg.adapters")
2. Load adapter classes via importlib
3. Instantiate and call get_metadata()
4. Register hooks (fetch, parse, validate)
5. Store metadata in registry cache

Runtime:
1. Orchestrator queries: pm.get_adapter(name="clinicaltrials", domain="biomedical")
2. Plugin manager returns registered adapter instance
3. Call adapter.fetch(AdapterRequest(...)) with resilience decorators
4. Return AdapterResponse with validation outcomes
```

## Migration Strategy

### Phase 1: Foundation (Week 1-2)

- Install pluggy, tenacity, pydantic-settings
- Implement plugin manager and hook specifications
- Create canonical Pydantic models
- Add backward compatibility shims

### Phase 2: Adapter Migration (Week 3-4)

- Migrate ClinicalTrials adapter (pilot)
- Migrate OpenFDA, OpenAlex, Unpaywall
- Migrate remaining biomedical adapters
- Run dual systems in parallel (feature flag)

### Phase 3: Framework Integration (Week 5-6)

- Update orchestrator to use plugin discovery
- Add gateway API endpoints for adapter metadata
- Implement health checks and monitoring
- Add example financial/legal adapters

### Phase 4: Deprecation (Week 7+)

- Remove old adapter registry
- Remove backward compatibility layer
- Update all documentation
- Archive this change

## Backward Compatibility

**Compatibility Layer**:

```python
class LegacyAdapterWrapper(BaseAdapter):
    """Wraps old-style adapters to work with new plugin system."""

    def __init__(self, legacy_adapter):
        self.legacy = legacy_adapter

    @hookimpl
    def get_metadata(self) -> AdapterMetadata:
        # Infer metadata from adapter attributes
        return AdapterMetadata(
            name=self.legacy.config.source,
            version="1.0.0",
            domain=AdapterDomain.BIOMEDICAL,
            capabilities={AdapterCapability.BATCH},
            supported_identifiers=[],
            auth_required=False,
            rate_limit_default=self.legacy.config.rate_limit.requests_per_second
        )

    @hookimpl
    async def fetch(self, request: AdapterRequest) -> AsyncIterator[RawPayload]:
        # Convert new request format to old fetch() signature
        legacy_params = self._convert_request(request)
        results = await self.legacy.fetch(**legacy_params)
        for item in results:
            yield RawPayload(data=item, source=request.source)

# Auto-wrap old adapters during migration
for name, adapter_cls in OLD_ADAPTERS.items():
    wrapped = LegacyAdapterWrapper(adapter_cls(config))
    pm.register(wrapped, name=f"legacy-{name}")
```

## Risks / Trade-offs

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Plugin discovery overhead | Medium | Low | Cache metadata, lazy load adapters |
| Breaking existing integrations | High | Medium | Feature flag, 2-release deprecation period |
| Configuration migration errors | Medium | High | Automated migration script, validation |
| Third-party adapter quality | Low | Medium | Adapter validation tests, certification program |
| Backward compatibility bugs | High | Medium | Comprehensive test suite, gradual rollout |

**Trade-offs**:

- **Pro**: Extensibility, maintainability, cross-domain support
- **Con**: Migration complexity, temporary dual-system maintenance
- **Pro**: Centralized resilience, observability
- **Con**: Additional dependencies (pluggy, tenacity)
- **Pro**: Environment-based configuration
- **Con**: Loss of YAML-based config inspection

## Open Questions

1. **Q**: Should we support hot-reloading of adapter plugins without restart?
   **A**: Phase 2 feature - implement plugin refresh endpoint

2. **Q**: How do we handle adapter versioning and compatibility?
   **A**: Semantic versioning in metadata, orchestrator can specify version constraints

3. **Q**: Should adapters be allowed to define custom pipeline stages?
   **A**: The default `AdapterPipeline` enforces fetch → parse → validate, but advanced adapters may override `build_pipeline` to
   insert domain-specific stages while still returning an `AdapterExecutionContext`/`AdapterInvocationResult`. The framework keeps
   ownership of cross-cutting concerns (resilience, auditing) and validates custom pipelines at registration time.

4. **Q**: How do we test third-party adapters for certification?
   **A**: Provide adapter test harness, require >80% coverage for "certified" badge
