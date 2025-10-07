# OpenSpec Change Proposal: add-adapter-plugin-framework

## ✅ Successfully Created

This OpenSpec change proposal has been successfully created and validated according to the protocol in `AGENTS.md`.

## 📁 Structure

```
openspec/changes/add-adapter-plugin-framework/
├── proposal.md              # Why, what, and impact summary
├── tasks.md                 # 88 implementation tasks across 11 work streams
├── design.md                # Technical decisions and architecture
└── specs/                   # Delta specifications for affected capabilities
    ├── biomedical-adapters/spec.md
    ├── configuration-management/spec.md
    ├── error-handling/spec.md
    ├── ingestion-orchestration/spec.md
    └── multi-protocol-gateway/spec.md
```

## 📊 Change Statistics

- **Total Tasks**: 88 tasks across 11 work streams
- **Estimated Timeline**: 8 weeks (4 phases)
- **Affected Specifications**: 5 capabilities
- **Breaking Changes**: 4 major breaking changes
- **New Requirements**: 11 new requirements
- **Modified Requirements**: 3 existing requirements
- **Removed Requirements**: 1 obsolete requirement

## 🎯 Key Components

### 1. Pluggy-Based Plugin Discovery

- Self-registering adapters via entry points
- Structured metadata (domain, capabilities, auth)
- Dynamic loading with dependency injection
- **Library**: Pluggy 1.3.0+ (<https://pluggy.readthedocs.io/>) - The same plugin framework used by pytest

### 2. Unified Resilience Layer

- Centralized retry policies using `tenacity`
- Circuit breaker pattern for failing services
- Consolidated rate limiting
- **Library**: Tenacity 8.2.0+ (<https://tenacity.readthedocs.io/>)

### 3. Configuration & Validation

- Pydantic models for all adapter I/O
- Environment-based configuration with `pydantic-settings`
- HashiCorp Vault integration for secrets
- **Library**: pydantic-settings 2.0.0+ (<https://docs.pydantic.dev/latest/concepts/pydantic_settings/>)

### 4. Domain-Specific Namespaces

- `adapters.biomedical`, `adapters.financial`, `adapters.legal`
- Shared `BaseAdapter` contract
- Cross-domain ingestion support

### 5. Developer Workflow

- Implement adapters by subclassing `Medical_KG_rev.adapters.plugins.base.BaseAdapterPlugin` (or the domain-specific helpers) and explicitly composing any legacy adapters that still power the integration logic.
- Register new adapters via `register_biomedical_plugins` or by defining entry points exposed through `scripts/migrate_adapter_entry_points.py`.
- Access the singleton plugin manager via `Medical_KG_rev.adapters.get_plugin_manager()`; feature flag `MK_USE_PLUGIN_FRAMEWORK` can disable the system for rollback scenarios.
- Domain-specific metadata helpers (`BiomedicalAdapterMetadata`, `FinancialAdapterMetadata`, `LegalAdapterMetadata`) ensure consistent documentation and filtering across REST/GraphQL surfaces.
- `AdapterPluginManager.execute()` returns an `AdapterExecutionState`, allowing orchestration flows to compose additional stages without modifying individual plugins.

### 6. Migration Guide

- Run `python scripts/migrate_adapter_entry_points.py --print` to inspect recommended entry points for legacy adapters.
- Use the new REST endpoints (`/v1/adapters`, `/v1/adapters/{name}/metadata`, `/v1/adapters/{name}/health`, `/v1/adapters/{name}/config-schema`) to validate plugin registration and configuration.
- GraphQL queries `adapters`, `adapter`, and `adapterHealth` expose the same metadata for multi-protocol clients.
- Existing YAML-based adapters can be wrapped by the domain-specific plugins (for example `ClinicalTrialsAdapterPlugin`) while teams migrate business logic into first-class plugin implementations.

## 📋 Specification Deltas

### biomedical-adapters

- **MODIFIED**: Adapter Registration (2 scenarios)
- **MODIFIED**: Adapter Contract (2 scenarios)
- **ADDED**: Unified Resilience Layer (2 scenarios)
- **ADDED**: Configuration Management (2 scenarios)
- **ADDED**: Domain-Specific Namespaces (2 scenarios)
- **ADDED**: Adapter Health Checks (2 scenarios)
- **ADDED**: Adapter Cost Estimation (1 scenario)
- **REMOVED**: Manual Adapter Registry

### ingestion-orchestration

- **MODIFIED**: Adapter Selection (2 scenarios)
- **ADDED**: Plugin Manager Integration (2 scenarios)
- **ADDED**: Dynamic Pipeline Construction (1 scenario)

### multi-protocol-gateway

- **ADDED**: Adapter Metadata API (3 scenarios)
- **ADDED**: GraphQL Adapter Queries (2 scenarios)
- **ADDED**: OpenAPI Specification Updates (1 scenario)

### configuration-management

- **MODIFIED**: Configuration Schema (2 scenarios)
- **ADDED**: Secret Management (2 scenarios)
- **ADDED**: Environment Variable Conventions (2 scenarios)
- **ADDED**: Configuration Hot-Reload (2 scenarios)

### error-handling

- **ADDED**: Adapter Error Taxonomy (3 scenarios)
- **ADDED**: Structured Error Responses (2 scenarios)
- **ADDED**: Error Observability (2 scenarios)
- **ADDED**: Circuit Breaker State Management (2 scenarios)

## 🔧 Implementation Phases

### Phase 1: Foundation (Week 1-2)

- Install dependencies (pluggy, tenacity, pydantic-settings)
- Implement plugin manager and hook specifications
- Create canonical Pydantic models
- Add backward compatibility shims

### Phase 2: Adapter Migration (Week 3-4)

- Migrate ClinicalTrials adapter (pilot)
- Migrate remaining 10 biomedical adapters
- Run dual systems in parallel (feature flag)
- Comprehensive testing

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

## ✅ Validation Status

```bash
$ openspec validate add-adapter-plugin-framework --strict
Change 'add-adapter-plugin-framework' is valid
```

✅ All requirements have at least one scenario
✅ All scenarios use proper formatting (`#### Scenario:`)
✅ All delta operations are properly structured
✅ All affected capabilities have delta files
✅ No validation errors or warnings

## 🚀 Next Steps

1. **Review**: Share proposal with team for technical review
2. **Approval**: Get approval from architecture and product teams
3. **Implementation**: Begin Phase 1 tasks (DO NOT start without approval)
4. **Testing**: Comprehensive test coverage for all new components
5. **Documentation**: Update developer guides and API documentation
6. **Rollout**: Gradual migration with feature flags
7. **Archive**: Move to archive after successful deployment

## 📚 Related Documentation

- `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md` - System architecture overview
- `src/Medical_KG_rev/adapters/base.py` - Current adapter implementation
- `src/Medical_KG_rev/adapters/biomedical.py` - Existing biomedical adapters
- `src/Medical_KG_rev/orchestration/orchestrator.py` - Pipeline orchestration
- `pyproject.toml` - Project dependencies and entry points

## 🔗 Commands

```bash
# View proposal
openspec show add-adapter-plugin-framework

# Validate changes
openspec validate add-adapter-plugin-framework --strict

# View specific spec deltas
openspec show add-adapter-plugin-framework --json --deltas-only

# After implementation, archive the change
openspec archive add-adapter-plugin-framework
```

---

**Created**: $(date +%Y-%m-%d)
**Status**: Pending Approval
**Priority**: High
**Complexity**: High (88 tasks, 8 weeks)
