# Change Proposal: Foundation Infrastructure

## Why

Establish the core foundation for the multi-protocol API gateway system including the unified data model (federated Intermediate Representation), configuration management, shared utilities, and base project structure. This foundation enables all subsequent components to build on consistent data structures and conventions.

## What Changes

- Create Pydantic-based federated data model (Document, Block, Section, Entity, Claim, Organization)
- Implement domain-specific overlays (medical/FHIR, finance/XBRL, legal/LegalDocML)
- Build configuration management system with environment-based settings
- Add structured logging with OpenTelemetry integration
- Create shared HTTP client with retry/backoff/rate limiting
- Implement Adapter SDK base classes for plug-in architecture
- Add validation utilities (ID format validation, span validation, schema validation)
- Set up common error handling and RFC 7807 Problem Details
- Create base storage abstractions (object store, ledger/state tracking)
- Implement provenance tracking models
- Add utility functions for ID generation, hashing, and timestamps

## Impact

- **Affected specs**: NEW capability `foundation`
- **Affected code**:
  - `src/Medical_KG_rev/models/` - Core data models
  - `src/Medical_KG_rev/config/` - Configuration management
  - `src/Medical_KG_rev/adapters/base.py` - Adapter SDK
  - `src/Medical_KG_rev/utils/` - Shared utilities
  - `src/Medical_KG_rev/storage/` - Storage abstractions
  - `pyproject.toml` - Add core dependencies (pydantic, httpx, opentelemetry, etc.)
  - `tests/unit/` - Comprehensive unit tests for all foundation components
