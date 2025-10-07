# Implementation Tasks

## 1. Plugin Framework Infrastructure

- [x] 1.1 Install and configure **Pluggy** dependency (pluggy>=1.3.0)
- [x] 1.2 Define `AdapterHookSpec` with Pluggy hook specifications using `@hookspec`
- [x] 1.3 Implement `PluginManager` for adapter discovery using Pluggy's `PluginManager` class
- [x] 1.4 Create adapter metadata model (`AdapterMetadata` Pydantic class)
- [x] 1.5 Add entry point specification in `pyproject.toml` template (group: `medical_kg.adapters`)
- [x] 1.6 Implement dynamic adapter loading from entry points using `importlib.metadata`
- [x] 1.7 Add adapter capability querying API via Pluggy hook calls
- [x] 1.8 Write unit tests for Pluggy plugin manager

## 2. Canonical Data Models

- [x] 2.1 Define `AdapterRequest` Pydantic model with tenant context
- [x] 2.2 Define `AdapterResponse` model with pagination support
- [x] 2.3 Define `ValidationOutcome` model with error details
- [x] 2.4 Define `AdapterConfig` base model with pydantic-settings
- [x] 2.5 Create domain-specific payload extensions (Medical, Financial, Legal)
- [x] 2.6 Add JSON Schema export for API documentation
- [x] 2.7 Implement model versioning strategy
- [x] 2.8 Write comprehensive model validation tests

## 3. Unified Resilience Layer

- [x] 3.1 Create `ResilienceConfig` Pydantic model
- [x] 3.2 Implement `@retry_on_failure` decorator using **Tenacity** (tenacity>=8.2.0)
- [x] 3.3 Implement `@rate_limit` decorator with token bucket algorithm
- [x] 3.4 Add circuit breaker pattern for external services using Tenacity
- [x] 3.5 Create `BackoffStrategy` enum (exponential, linear, jitter)
- [x] 3.6 Implement `ResilientHTTPClient` wrapper with Tenacity automatic retries
- [x] 3.7 Add Prometheus metrics for retry attempts and circuit breaker state
- [x] 3.8 Write resilience layer integration tests

## 4. Configuration & Secret Management

- [x] 4.1 Define `AdapterSettings` using **pydantic-settings** (pydantic-settings>=2.0.0)
- [x] 4.2 Implement Vault secret provider integration
- [x] 4.3 Add environment variable mapping with `MK_ADAPTER_` prefix
- [x] 4.4 Create configuration validation on startup
- [x] 4.5 Implement hot-reload for configuration changes
- [x] 4.6 Add configuration schema documentation generator
- [x] 4.7 Migrate existing YAML configs to environment-based settings
- [x] 4.8 Write configuration management tests

## 5. BaseAdapter Contract Rewrite

- [x] 5.1 Define new `BaseAdapter` abstract class with Pluggy `@hookimpl` decorators
- [x] 5.2 Implement `fetch()` method signature with `AdapterRequest`
- [x] 5.3 Implement `parse()` method signature with `AdapterResponse`
- [x] 5.4 Add `validate()` hook for schema validation
- [x] 5.5 Add `get_metadata()` method for capability declaration (Pluggy hook)
- [x] 5.6 Implement `health_check()` for adapter readiness (Pluggy hook)
- [x] 5.7 Add `estimate_cost()` for rate limit planning (Pluggy hook)
- [x] 5.8 Write BaseAdapter contract tests

## 6. Domain-Specific Adapter Namespaces

- [x] 6.1 Create `adapters.biomedical` package structure
- [x] 6.2 Create `adapters.financial` package structure (scaffold)
- [x] 6.3 Create `adapters.legal` package structure (scaffold)
- [x] 6.4 Define domain-specific metadata schemas
- [x] 6.5 Implement cross-domain adapter registry
- [x] 6.6 Add domain filtering in plugin discovery
- [x] 6.7 Create domain-specific test fixtures
- [x] 6.8 Write cross-domain integration tests

## 7. Migrate Biomedical Adapters

- [x] 7.1 Migrate `ClinicalTrialsAdapter` to new contract
- [x] 7.2 Migrate `OpenFDAAdapter` (drug labels, adverse events, devices)
- [x] 7.3 Migrate `OpenAlexAdapter` to new contract
- [x] 7.4 Migrate `UnpaywallAdapter` to new contract
- [x] 7.5 Migrate `CrossrefAdapter` to new contract
- [x] 7.6 Migrate `COREAdapter` to new contract
- [x] 7.7 Migrate `PMCAdapter` to new contract
- [x] 7.8 Migrate `RxNormAdapter` to new contract
- [x] 7.9 Migrate `ICD11Adapter` to new contract
- [x] 7.10 Migrate `MeSHAdapter` to new contract
- [x] 7.11 Migrate `ChEMBLAdapter` to new contract
- [x] 7.12 Add backward compatibility layer for old adapters

## 8. Orchestration Integration

- [x] 8.1 Update `Orchestrator` to use Pluggy plugin manager
- [x] 8.2 Replace manual adapter registry with Pluggy plugin discovery
- [x] 8.3 Implement adapter selection based on metadata from Pluggy hooks
- [x] 8.4 Add dynamic pipeline construction for multi-domain ingestion
- [x] 8.5 Update job ledger to track adapter plugin versions
- [x] 8.6 Add adapter health checks to orchestrator startup (via Pluggy hooks)
- [x] 8.7 Implement adapter fallback strategies
- [x] 8.8 Write orchestration integration tests

## 9. Gateway API Updates

- [x] 9.1 Add `GET /v1/adapters` endpoint for adapter listing
- [x] 9.2 Add `GET /v1/adapters/{name}/metadata` endpoint
- [x] 9.3 Add `GET /v1/adapters/{name}/health` endpoint
- [x] 9.4 Add `GET /v1/adapters/{name}/config-schema` endpoint
- [x] 9.5 Update OpenAPI specification with adapter endpoints
- [x] 9.6 Add GraphQL queries for adapter discovery
- [x] 9.7 Update REST API documentation
- [x] 9.8 Write API contract tests

## 10. Testing & Documentation

- [x] 10.1 Create mock adapter plugin for testing (using Pluggy hooks)
- [x] 10.2 Write unit tests for Pluggy plugin framework (>90% coverage)
- [x] 10.3 Write integration tests for adapter lifecycle
- [x] 10.4 Write performance tests for Pluggy adapter discovery overhead
- [x] 10.5 Update developer documentation for adapter authoring with Pluggy
- [x] 10.6 Create adapter migration guide (manual registry → Pluggy)
- [x] 10.7 Add example adapters for financial and legal domains
- [x] 10.8 Update `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md`

## 11. Migration & Deprecation

- [x] 11.1 Create automated migration script for entry points
- [x] 11.2 Add deprecation warnings to old adapter registry
- [x] 11.3 Implement feature flag for plugin framework toggle
- [x] 11.4 Update CI/CD to test both old and new systems
- [x] 11.5 Create rollback procedure documentation
- [x] 11.6 Schedule old registry removal for release v0.3.0
- [x] 11.7 Monitor adapter performance metrics post-migration
- [x] 11.8 Remove backward compatibility layer
