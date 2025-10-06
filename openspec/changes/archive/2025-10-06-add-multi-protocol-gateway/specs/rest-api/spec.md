# REST API Specification

## ADDED Requirements

### Requirement: OpenAPI 3.1 Compliance

The system SHALL expose a complete REST API documented with OpenAPI 3.1 specification.

#### Scenario: OpenAPI spec generation

- **WHEN** FastAPI application starts
- **THEN** `/openapi.json` MUST serve valid OpenAPI 3.1 JSON schema

#### Scenario: Swagger UI availability

- **WHEN** accessing `/docs/openapi`
- **THEN** interactive Swagger UI MUST be displayed

### Requirement: JSON:API Response Format

All REST responses SHALL conform to JSON:API v1.1 specification using `application/vnd.api+json` content type.

#### Scenario: Successful resource response

- **WHEN** retrieving a resource
- **THEN** response MUST include `{data: {type, id, attributes}, meta}` structure

#### Scenario: Error response format

- **WHEN** error occurs
- **THEN** response MUST include `{errors: [{status, title, detail, code}]}`

### Requirement: Ingestion Endpoints

The system SHALL provide POST endpoints for ingesting data from various sources.

#### Scenario: Clinical trials ingestion

- **WHEN** POST `/ingest/clinicaltrials` with NCT IDs
- **THEN** system MUST fetch from ClinicalTrials.gov API and return status per ID

#### Scenario: Batch ingestion with partial failure

- **WHEN** some IDs succeed and others fail
- **THEN** system MUST return 207 Multi-Status with per-ID results

### Requirement: OData Query Support

The system SHALL support OData query syntax for filtering, selecting fields, and pagination.

#### Scenario: Filter query

- **WHEN** GET with `?$filter=status eq 'active'`
- **THEN** only active resources MUST be returned

#### Scenario: Field selection

- **WHEN** GET with `?$select=title,date`
- **THEN** only specified fields MUST be included in response

### Requirement: RFC 7807 Problem Details

All error responses SHALL use RFC 7807 Problem Details format with type URI, title, status, and detail.

#### Scenario: Validation error

- **WHEN** invalid input provided
- **THEN** response MUST be 400 with Problem Details JSON including field-specific errors

### Requirement: API Versioning
The system SHALL support API versioning to enable backward-compatible changes and deprecation cycles.

#### Scenario: Version in URL path
- **WHEN** accessing REST API
- **THEN** all endpoints MUST be prefixed with version (e.g., `/v1/ingest/clinicaltrials`)

#### Scenario: Deprecated version warning
- **WHEN** client uses deprecated API version
- **THEN** response MUST include `Deprecation` header with sunset date

### Requirement: Health Check Endpoints
The system SHALL provide health check endpoints for monitoring and orchestration.

#### Scenario: Liveness check
- **WHEN** GET `/v1/health` is called
- **THEN** system MUST return 200 OK with basic status

#### Scenario: Readiness check with dependencies
- **WHEN** GET `/v1/ready` is called
- **THEN** system MUST check dependencies and return 200 if ready or 503 if not
