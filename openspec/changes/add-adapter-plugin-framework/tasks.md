# Implementation Tasks

## 1. Plugin Framework Infrastructure

- [ ] 1.1 Install and configure **Pluggy** dependency (pluggy>=1.3.0)
- [ ] 1.2 Define `AdapterHookSpec` with Pluggy hook specifications using `@hookspec`
- [ ] 1.3 Implement `PluginManager` for adapter discovery using Pluggy's `PluginManager` class
- [ ] 1.4 Create adapter metadata model (`AdapterMetadata` Pydantic class)
- [ ] 1.5 Add entry point specification in `pyproject.toml` template (group: `medical_kg.adapters`)
- [ ] 1.6 Implement dynamic adapter loading from entry points using `importlib.metadata`
- [ ] 1.7 Add adapter capability querying API via Pluggy hook calls
- [ ] 1.8 Write unit tests for Pluggy plugin manager

## 2. Canonical Data Models

- [ ] 2.1 Define `AdapterRequest` Pydantic model with tenant context
- [ ] 2.2 Define `AdapterResponse` model with pagination support
- [ ] 2.3 Define `ValidationOutcome` model with error details
- [ ] 2.4 Define `AdapterConfig` base model with pydantic-settings
- [ ] 2.5 Create domain-specific payload extensions (Medical, Financial, Legal)
- [ ] 2.6 Add JSON Schema export for API documentation
- [ ] 2.7 Implement model versioning strategy
- [ ] 2.8 Write comprehensive model validation tests

## 3. Unified Resilience Layer

- [ ] 3.1 Create `ResilienceConfig` Pydantic model
- [ ] 3.2 Implement `@retry_on_failure` decorator using **Tenacity** (tenacity>=8.2.0)
- [ ] 3.3 Implement `@rate_limit` decorator with token bucket algorithm
- [ ] 3.4 Add circuit breaker pattern for external services using Tenacity
- [ ] 3.5 Create `BackoffStrategy` enum (exponential, linear, jitter)
- [ ] 3.6 Implement `ResilientHTTPClient` wrapper with Tenacity automatic retries
- [ ] 3.7 Add Prometheus metrics for retry attempts and circuit breaker state
- [ ] 3.8 Write resilience layer integration tests

## 4. Configuration & Secret Management

- [ ] 4.1 Define `AdapterSettings` using **pydantic-settings** (pydantic-settings>=2.0.0)
- [ ] 4.2 Implement Vault secret provider integration
- [ ] 4.3 Add environment variable mapping with `MK_ADAPTER_` prefix
- [ ] 4.4 Create configuration validation on startup
- [ ] 4.5 Implement hot-reload for configuration changes
- [ ] 4.6 Add configuration schema documentation generator
- [ ] 4.7 Migrate existing YAML configs to environment-based settings
- [ ] 4.8 Write configuration management tests

## 5. BaseAdapter Contract Rewrite

- [ ] 5.1 Define new `BaseAdapter` abstract class with Pluggy `@hookimpl` decorators
- [ ] 5.2 Implement `fetch()` method signature with `AdapterRequest`
- [ ] 5.3 Implement `parse()` method signature with `AdapterResponse`
- [ ] 5.4 Add `validate()` hook for schema validation
- [ ] 5.5 Add `get_metadata()` method for capability declaration (Pluggy hook)
- [ ] 5.6 Implement `health_check()` for adapter readiness (Pluggy hook)
- [ ] 5.7 Add `estimate_cost()` for rate limit planning (Pluggy hook)
- [ ] 5.8 Write BaseAdapter contract tests

## 6. Domain-Specific Adapter Namespaces

- [ ] 6.1 Create `adapters.biomedical` package structure
- [ ] 6.2 Create `adapters.financial` package structure (scaffold)
- [ ] 6.3 Create `adapters.legal` package structure (scaffold)
- [ ] 6.4 Define domain-specific metadata schemas
- [ ] 6.5 Implement cross-domain adapter registry
- [ ] 6.6 Add domain filtering in plugin discovery
- [ ] 6.7 Create domain-specific test fixtures
- [ ] 6.8 Write cross-domain integration tests

## 7. Migrate Biomedical Adapters

- [ ] 7.1 Migrate `ClinicalTrialsAdapter` to new contract
- [ ] 7.2 Migrate `OpenFDAAdapter` (drug labels, adverse events, devices)
- [ ] 7.3 Migrate `OpenAlexAdapter` to new contract
- [ ] 7.4 Migrate `UnpaywallAdapter` to new contract
- [ ] 7.5 Migrate `CrossrefAdapter` to new contract
- [ ] 7.6 Migrate `COREAdapter` to new contract
- [ ] 7.7 Migrate `PMCAdapter` to new contract
- [ ] 7.8 Migrate `RxNormAdapter` to new contract
- [ ] 7.9 Migrate `ICD11Adapter` to new contract
- [ ] 7.10 Migrate `MeSHAdapter` to new contract
- [ ] 7.11 Migrate `ChEMBLAdapter` to new contract
- [ ] 7.12 Add backward compatibility layer for old adapters

## 8. Orchestration Integration

- [ ] 8.1 Update `Orchestrator` to use Pluggy plugin manager
- [ ] 8.2 Replace manual adapter registry with Pluggy plugin discovery
- [ ] 8.3 Implement adapter selection based on metadata from Pluggy hooks
- [ ] 8.4 Add dynamic pipeline construction for multi-domain ingestion
- [ ] 8.5 Update job ledger to track adapter plugin versions
- [ ] 8.6 Add adapter health checks to orchestrator startup (via Pluggy hooks)
- [ ] 8.7 Implement adapter fallback strategies
- [ ] 8.8 Write orchestration integration tests

## 9. Gateway API Updates

- [ ] 9.1 Add `GET /v1/adapters` endpoint for adapter listing
- [ ] 9.2 Add `GET /v1/adapters/{name}/metadata` endpoint
- [ ] 9.3 Add `GET /v1/adapters/{name}/health` endpoint
- [ ] 9.4 Add `GET /v1/adapters/{name}/config-schema` endpoint
- [ ] 9.5 Update OpenAPI specification with adapter endpoints
- [ ] 9.6 Add GraphQL queries for adapter discovery
- [ ] 9.7 Update REST API documentation
- [ ] 9.8 Write API contract tests

## 10. Testing & Documentation

- [ ] 10.1 Create mock adapter plugin for testing (using Pluggy hooks)
- [ ] 10.2 Write unit tests for Pluggy plugin framework (>90% coverage)
- [ ] 10.3 Write integration tests for adapter lifecycle
- [ ] 10.4 Write performance tests for Pluggy adapter discovery overhead
- [ ] 10.5 Update developer documentation for adapter authoring with Pluggy
- [ ] 10.6 Create adapter migration guide (manual registry → Pluggy)
- [ ] 10.7 Add example adapters for financial and legal domains
- [ ] 10.8 Update `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md`

## 11. Migration & Deprecation

- [ ] 11.1 Create automated migration script for entry points
- [ ] 11.2 Add deprecation warnings to old adapter registry
- [ ] 11.3 Implement feature flag for plugin framework toggle
- [ ] 11.4 Update CI/CD to test both old and new systems
- [ ] 11.5 Create rollback procedure documentation
- [ ] 11.6 Schedule old registry removal for release v0.3.0
- [ ] 11.7 Monitor adapter performance metrics post-migration
- [ ] 11.8 Remove backward compatibility layer
