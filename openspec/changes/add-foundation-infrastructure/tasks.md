# Implementation Tasks: Foundation Infrastructure

## 1. Core Data Models

- [ ] 1.1 Create `models/ir.py` with Document, Block, Section, Table, Span classes
- [ ] 1.2 Create `models/entities.py` with Entity, Claim, Evidence, ExtractionActivity
- [ ] 1.3 Create `models/organization.py` with Organization, TenantContext
- [ ] 1.4 Create `models/overlays/medical.py` for FHIR alignment (ResearchStudy, Evidence)
- [ ] 1.5 Create `models/overlays/finance.py` for XBRL structures
- [ ] 1.6 Create `models/overlays/legal.py` for LegalDocML structures
- [ ] 1.7 Add comprehensive Pydantic validators for all models
- [ ] 1.8 Write unit tests for all data models (>90% coverage)

## 2. Configuration Management

- [ ] 2.1 Create `config/settings.py` with Pydantic Settings
- [ ] 2.2 Add environment-specific configs (dev, staging, prod)
- [ ] 2.3 Add secrets management integration (Vault/environment)
- [ ] 2.4 Create `config/domains.py` for multi-domain configuration
- [ ] 2.5 Add feature flags support
- [ ] 2.6 Write configuration validation tests

## 3. Shared Utilities

- [ ] 3.1 Create `utils/http_client.py` with retry, backoff, rate limiting
- [ ] 3.2 Create `utils/logging.py` with structured logging and OpenTelemetry
- [ ] 3.3 Create `utils/validation.py` with ID validators (NCT, DOI, PMCID, etc.)
- [ ] 3.4 Create `utils/identifiers.py` for doc_id generation and hashing
- [ ] 3.5 Create `utils/errors.py` with RFC 7807 Problem Details
- [ ] 3.6 Create `utils/spans.py` for span validation and manipulation
- [ ] 3.7 Write comprehensive utility tests

## 4. Adapter SDK

- [ ] 4.1 Create `adapters/base.py` with BaseAdapter abstract class
- [ ] 4.2 Add adapter lifecycle methods (fetch, parse, validate, write)
- [ ] 4.3 Create `adapters/registry.py` for adapter discovery
- [ ] 4.4 Add `adapters/yaml_parser.py` for declarative adapter configs
- [ ] 4.5 Create adapter testing utilities
- [ ] 4.6 Write adapter SDK documentation
- [ ] 4.7 Add example adapter implementation

## 5. Storage Abstractions

- [ ] 5.1 Create `storage/base.py` with abstract storage interfaces
- [ ] 5.2 Create `storage/object_store.py` for S3/MinIO integration
- [ ] 5.3 Create `storage/ledger.py` for state tracking and job coordination
- [ ] 5.4 Add `storage/cache.py` for Redis integration
- [ ] 5.5 Write storage integration tests

## 6. Provenance & Metadata

- [ ] 6.1 Create `models/provenance.py` with ExtractionActivity, DataSource
- [ ] 6.2 Add timestamp utilities with UTC enforcement
- [ ] 6.3 Create version tracking utilities
- [ ] 6.4 Add metadata extraction helpers

## 7. Project Setup

- [ ] 7.1 Update `pyproject.toml` with all core dependencies
- [ ] 7.2 Create `requirements.txt` and `requirements-dev.txt`
- [ ] 7.3 Set up pre-commit hooks (black, ruff, mypy)
- [ ] 7.4 Create `.env.example` with all configuration variables
- [ ] 7.5 Add VS Code / Cursor workspace settings
- [ ] 7.6 Create initial CI/CD workflow (lint, type-check, test)

## 8. Documentation

- [ ] 8.1 Create `docs/architecture/foundation.md`
- [ ] 8.2 Create `docs/guides/data-models.md`
- [ ] 8.3 Create `docs/guides/adapter-sdk.md`
- [ ] 8.4 Add inline docstrings for all public APIs
- [ ] 8.5 Generate API documentation with pdoc or sphinx
