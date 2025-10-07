# Biomedical Adapters Specification Deltas

## MODIFIED Requirements

### Requirement: Adapter Registration

The system SHALL support dynamic adapter registration via setuptools entry points using the group `medical_kg.adapters` and the **Pluggy** plugin framework (https://pluggy.readthedocs.io/). Each adapter MUST provide a `get_metadata()` method returning structured metadata including name, version, domain, capabilities, supported identifiers, authentication requirements, and rate limits. The plugin manager SHALL use Pluggy's `HookspecMarker` to define hook specifications and `HookimplMarker` for plugin implementations.

#### Scenario: Third-party adapter installation

- **GIVEN** a third-party adapter package installed via `pip install medical-kg-adapter-pubmed`
- **WHEN** the application starts
- **THEN** the Pluggy plugin manager discovers the adapter via entry points
- **AND** the adapter is registered using Pluggy's hook implementation mechanism
- **AND** the adapter appears in `GET /v1/adapters` endpoint

#### Scenario: Adapter metadata query

- **GIVEN** multiple adapters are registered (clinicaltrials, openfda, openalex)
- **WHEN** a client requests `GET /v1/adapters?domain=biomedical`
- **THEN** the response includes adapter metadata for all biomedical adapters
- **AND** each adapter includes capabilities, rate limits, and supported identifiers

### Requirement: Adapter Contract

All adapters MUST implement the `BaseAdapter` abstract class with standardized `fetch()`, `parse()`, and `validate()` methods accepting and returning canonical Pydantic models (`AdapterRequest`, `AdapterResponse`, `ValidationOutcome`). The adapter lifecycle SHALL be: authentication → fetch → parse → validate → transform.

#### Scenario: Successful adapter execution

- **GIVEN** a registered adapter "clinicaltrials"
- **WHEN** the orchestrator calls `adapter.fetch(AdapterRequest(identifiers=["NCT04267848"]))`
- **THEN** the adapter returns `AsyncIterator[RawPayload]` with fetched data
- **AND** calling `adapter.parse(payloads)` returns `list[Document]` in IR format
- **AND** calling `adapter.validate(documents)` returns `ValidationOutcome(valid=True)`

#### Scenario: Adapter validation failure

- **GIVEN** a parsed document with missing required field "title"
- **WHEN** `adapter.validate(documents)` is called
- **THEN** the validation returns `ValidationOutcome(valid=False, errors=[...])`
- **AND** the error includes field path and validation message
- **AND** the document is not passed to downstream pipeline stages

## ADDED Requirements

### Requirement: Unified Resilience Layer

The system SHALL provide a centralized resilience layer using the `tenacity` library for retry policies and backoff strategies. All adapter HTTP/gRPC calls MUST be wrapped with `@retry_on_failure` decorator configured via `ResilienceConfig`. Circuit breaker pattern SHALL be implemented for external services with configurable failure thresholds.

#### Scenario: Automatic retry on transient failure

- **GIVEN** an adapter configured with `ResilienceConfig(max_attempts=3, exponential_base=2)`
- **WHEN** the external API returns HTTP 503 (Service Unavailable)
- **THEN** the adapter automatically retries after 1 second (first retry)
- **AND** retries after 2 seconds (second retry)
- **AND** retries after 4 seconds (third retry)
- **AND** raises exception after max attempts exhausted

#### Scenario: Circuit breaker opens on repeated failures

- **GIVEN** an adapter with circuit breaker threshold of 5 failures
- **WHEN** the external API fails 5 consecutive times within 60 seconds
- **THEN** the circuit breaker opens
- **AND** subsequent requests fail immediately without calling external API
- **AND** Prometheus metric `adapter_circuit_breaker_state{name="clinicaltrials"}` = "open"
- **AND** the circuit breaker attempts reset after 30 seconds

### Requirement: Configuration Management

Adapters SHALL use `pydantic-settings` for type-safe configuration loaded from environment variables with prefix `MK_ADAPTER_`. Sensitive configuration (API keys, tokens) MUST be stored as `SecretStr` and loaded from HashiCorp Vault or environment variables. Configuration validation SHALL occur at application startup with clear error messages for missing or invalid settings.

#### Scenario: Environment-based configuration

- **GIVEN** environment variables `MK_ADAPTER_CLINICALTRIALS_API_KEY=secret123` and `MK_ADAPTER_TIMEOUT_SECONDS=60`
- **WHEN** the ClinicalTrialsAdapter is initialized
- **THEN** the adapter loads settings via pydantic-settings
- **AND** `adapter.settings.api_key.get_secret_value()` returns "secret123"
- **AND** `adapter.settings.timeout_seconds` returns 60

#### Scenario: Configuration validation failure

- **GIVEN** a required setting `MK_ADAPTER_OPENFDA_API_KEY` is not provided
- **WHEN** the OpenFDAAdapter is initialized
- **THEN** a `ValidationError` is raised
- **AND** the error message indicates missing required field "api_key"
- **AND** the application startup fails with exit code 1

### Requirement: Domain-Specific Namespaces

Adapters SHALL be organized into domain-specific namespaces: `adapters.biomedical`, `adapters.financial`, `adapters.legal`. All adapters within a namespace MUST implement the same `BaseAdapter` contract but MAY extend with domain-specific payload models using Pydantic discriminated unions. The plugin manager SHALL support filtering adapters by domain.

#### Scenario: Cross-domain adapter query

- **GIVEN** adapters registered in biomedical, financial, and legal domains
- **WHEN** orchestrator queries `pm.get_adapters(domain=AdapterDomain.BIOMEDICAL)`
- **THEN** only biomedical adapters are returned (clinicaltrials, openfda, etc.)
- **AND** financial and legal adapters are excluded from results

#### Scenario: Domain-specific payload extension

- **GIVEN** a biomedical adapter with `MedicalPayload(base_fields + nct_id, pmcid)`
- **AND** a financial adapter with `FinancialPayload(base_fields + ticker, cusip)`
- **WHEN** each adapter processes its domain-specific request
- **THEN** the appropriate discriminated union variant is used
- **AND** Pydantic validates domain-specific fields correctly

### Requirement: Adapter Health Checks

Each adapter MUST implement a `health_check()` method that verifies connectivity to the external data source. Health checks SHALL be callable via `GET /v1/adapters/{name}/health` endpoint and SHALL return status (healthy/unhealthy), response time, and optional error message. The orchestrator SHALL poll adapter health checks on startup and periodically thereafter.

#### Scenario: Healthy adapter

- **GIVEN** the ClinicalTrialsAdapter with health check URL "<https://clinicaltrials.gov/api/v2/health>"
- **WHEN** `adapter.health_check()` is called
- **THEN** the method returns `HealthCheckResult(status="healthy", response_time_ms=120)`
- **AND** the result is cached for 60 seconds

#### Scenario: Unhealthy adapter

- **GIVEN** the OpenFDA API is down
- **WHEN** `adapter.health_check()` is called
- **THEN** the method returns `HealthCheckResult(status="unhealthy", error="Connection timeout")`
- **AND** the orchestrator skips this adapter for new ingestion jobs
- **AND** an alert is triggered via Prometheus Alertmanager

### Requirement: Adapter Cost Estimation

Adapters MUST declare cost-per-request in metadata for quota tracking. The orchestrator SHALL call `adapter.estimate_cost(request)` before executing fetch operations to verify tenant quota availability. Cost tracking SHALL be exposed via Prometheus metrics `adapter_cost_total{adapter="clinicaltrials", tenant="tenant-123"}`.

#### Scenario: Cost estimation before fetch

- **GIVEN** a ClinicalTrialsAdapter with `cost_per_request=0.01` (quota units)
- **WHEN** orchestrator receives request to fetch 100 studies
- **THEN** `adapter.estimate_cost(AdapterRequest(identifiers=["NCT..."] * 100))` returns 1.0
- **AND** orchestrator checks tenant quota (remaining >= 1.0)
- **AND** fetch proceeds if quota available, else returns 429 Too Many Requests

## REMOVED Requirements

### Requirement: Manual Adapter Registry

**Reason**: Replaced by pluggy-based plugin discovery via entry points
**Migration**: Existing adapters in `ADAPTERS` dictionary will be auto-migrated to entry points via migration script
