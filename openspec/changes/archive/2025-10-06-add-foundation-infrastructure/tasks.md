# Implementation Tasks: Foundation Infrastructure

## 1. Core Data Models

- [x] 1.1 Create `models/ir.py` with Document, Block, Section, Table, Span classes
- [x] 1.2 Create `models/entities.py` with Entity, Claim, Evidence, ExtractionActivity
- [x] 1.3 Create `models/organization.py` with Organization, TenantContext
- [x] 1.4 Create `models/overlays/medical.py` for FHIR alignment (ResearchStudy, Evidence)
- [x] 1.5 Create `models/overlays/finance.py` for XBRL structures
- [x] 1.6 Create `models/overlays/legal.py` for LegalDocML structures
- [x] 1.7 Add comprehensive Pydantic validators for all models
- [x] 1.8 Write unit tests for all data models (>90% coverage)

## 2. Configuration Management

- [x] 2.1 Create `config/settings.py` with Pydantic Settings
- [x] 2.2 Add environment-specific configs (dev, staging, prod)
- [x] 2.3 Add secrets management integration (Vault/environment)
- [x] 2.4 Create `config/domains.py` for multi-domain configuration
- [x] 2.5 Add feature flags support
- [x] 2.6 Write configuration validation tests

## 3. Shared Utilities

- [x] 3.1 Create `utils/http_client.py` with retry, backoff, rate limiting
- [x] 3.2 Create `utils/logging.py` with structured logging and OpenTelemetry
- [x] 3.3 Create `utils/validation.py` with ID validators (NCT, DOI, PMCID, etc.)
- [x] 3.4 Create `utils/identifiers.py` for doc_id generation and hashing
- [x] 3.5 Create `utils/errors.py` with RFC 7807 Problem Details
- [x] 3.6 Create `utils/spans.py` for span validation and manipulation
- [x] 3.7 Write comprehensive utility tests

## 4. Adapter SDK

- [x] 4.1 Create `adapters/base.py` with BaseAdapter abstract class
- [x] 4.2 Add adapter lifecycle methods (fetch, parse, validate, write)
- [x] 4.3 Create `adapters/registry.py` for adapter discovery
- [x] 4.4 Add `adapters/yaml_parser.py` for declarative adapter configs
- [x] 4.5 Create adapter testing utilities
- [x] 4.6 Write adapter SDK documentation
- [x] 4.7 Add example adapter implementation

## 5. Storage Abstractions

- [x] 5.1 Create `storage/base.py` with abstract storage interfaces
- [x] 5.2 Create `storage/object_store.py` for S3/MinIO integration
- [x] 5.3 Create `storage/ledger.py` for state tracking and job coordination
- [x] 5.4 Add `storage/cache.py` for Redis integration
- [x] 5.5 Write storage integration tests

## 6. Provenance & Metadata

- [x] 6.1 Create `models/provenance.py` with ExtractionActivity, DataSource
- [x] 6.2 Add timestamp utilities with UTC enforcement
- [x] 6.3 Create version tracking utilities
- [x] 6.4 Add metadata extraction helpers

## 7. Project Setup

- [x] 7.1 Update `pyproject.toml` with all core dependencies
- [x] 7.2 Create `requirements.txt` and `requirements-dev.txt`
- [x] 7.3 Set up pre-commit hooks (black, ruff, mypy)
- [x] 7.4 Create `.env.example` with all configuration variables
- [x] 7.5 Add VS Code / Cursor workspace settings
- [x] 7.6 Create initial CI/CD workflow (lint, type-check, test)

## 8. Documentation

- [x] 8.1 Create `docs/architecture/foundation.md`
- [x] 8.2 Create `docs/guides/data-models.md`
- [x] 8.3 Create `docs/guides/adapter-sdk.md`
- [x] 8.4 Add inline docstrings for all public APIs
- [x] 8.5 Generate API documentation with pdoc or sphinx
