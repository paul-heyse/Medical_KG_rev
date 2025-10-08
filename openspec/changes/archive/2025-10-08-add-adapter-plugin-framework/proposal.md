# Adapter Plugin Framework Proposal

## Why

The current adapter implementation suffers from several architectural limitations that hinder extensibility, maintainability, and cross-domain scalability:

1. **Opaque Discovery**: Adapters are manually registered in custom registries with no standard metadata about capabilities, domains, or requirements
2. **Fragmented Resilience**: Retry logic, rate limiting, and error handling are implemented inconsistently across 11+ biomedical adapters
3. **Configuration Sprawl**: Each adapter manages its own configuration parsing with YAML files scattered across the codebase
4. **Domain Coupling**: The adapter contract is biomedical-specific, making it difficult to extend to financial (XBRL) or legal (LegalDocML) domains
5. **Testing Overhead**: Mock adapters and test fixtures require manual maintenance due to lack of standardized interfaces

These issues create friction when adding new data sources, extending to new domains, debugging ingestion failures, and onboarding contributors.

## What Changes

This proposal introduces a **standardized adapter plugin framework** with four major components:

### 1. Pluggy-Based Plugin Discovery System

- Replace custom adapter registries with **Pluggy** (<https://pluggy.readthedocs.io/>) hook specifications
- Self-registering adapters via `importlib.metadata.entry_points`
- Structured metadata: domain tags, capability declarations, auth requirements
- Dynamic loading with dependency injection support
- **Library**: `pluggy>=1.3.0` - The same plugin framework used by pytest

### 2. Unified Resilience Layer

- Centralized retry policies using **Tenacity** (<https://tenacity.readthedocs.io/>) decorators
- Configurable backoff strategies (exponential, jitter)
- Circuit breaker pattern for failing external services
- Consolidated rate limiting via token bucket algorithm
- **Library**: `tenacity>=8.2.0` - Retry library with declarative syntax

### 3. Standardized Configuration & Validation

- Canonical Pydantic models for adapter payloads (`AdapterRequest`, `AdapterResponse`, `ValidationOutcome`)
- Integration with **pydantic-settings** for environment-based configuration
- Secret management via HashiCorp Vault or environment variables
- Schema-driven validation replacing ad-hoc checks
- **Library**: `pydantic-settings>=2.0.0` - Type-safe configuration management

### 4. Domain-Specific Adapter Namespaces

- Partition adapters: `adapters.biomedical.*`, `adapters.financial.*`, `adapters.legal.*`
- Shared `BaseAdapter` contract across all domains
- Domain-specific payload extensions using discriminated unions
- Cross-domain ingestion orchestration support

### Breaking Changes

- **BREAKING**: Adapter registration moves from manual registry to entry points in `pyproject.toml`
- **BREAKING**: `BaseAdapter` signature changes to use Pydantic request/response models
- **BREAKING**: Adapter configuration files migrate from YAML to environment-based settings
- **BREAKING**: Custom retry logic in adapters must be removed (handled by framework)

## Impact

### Affected Specifications

- `specs/biomedical-adapters/spec.md` - Complete rewrite of adapter contract
- `specs/ingestion-orchestration/spec.md` - Update pipeline to use plugin discovery
- `specs/multi-protocol-gateway/spec.md` - Update REST endpoints for adapter metadata
- `specs/configuration-management/spec.md` - New configuration schema
- `specs/error-handling/spec.md` - Standardized adapter error taxonomy

### Affected Code

- `src/Medical_KG_rev/adapters/base.py` - BaseAdapter interface rewrite
- `src/Medical_KG_rev/adapters/registry.py` - Replace with pluggy hook manager
- `src/Medical_KG_rev/adapters/biomedical.py` - Refactor 11 adapters to new contract
- `src/Medical_KG_rev/adapters/resilience.py` - **NEW** - Unified resilience utilities
- `src/Medical_KG_rev/adapters/models.py` - **NEW** - Canonical Pydantic models
- `src/Medical_KG_rev/adapters/config.py` - **NEW** - pydantic-settings integration
- `src/Medical_KG_rev/orchestration/orchestrator.py` - Update to use plugin discovery
- `src/Medical_KG_rev/gateway/rest/router.py` - Add adapter metadata endpoints
- `tests/adapters/` - Comprehensive test suite rewrite
- `pyproject.toml` - Add entry points for adapter registration

### Migration Path

1. **Phase 1** (Week 1-2): Implement plugin framework and canonical models
2. **Phase 2** (Week 3-4): Migrate existing biomedical adapters (backward compatibility layer)
3. **Phase 3** (Week 5-6): Add financial/legal adapter examples
4. **Phase 4** (Week 7): Deprecate old registry, remove backward compatibility

### Rollback Plan

- Maintain old `AdapterRegistry` for 2 releases with deprecation warnings
- Feature flag `USE_PLUGIN_FRAMEWORK` to toggle between old/new systems
- Automated migration script to generate entry points from existing adapters
