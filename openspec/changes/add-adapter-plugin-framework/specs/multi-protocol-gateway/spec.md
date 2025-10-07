# Multi-Protocol Gateway Specification Deltas

## ADDED Requirements

### Requirement: Adapter Metadata API

The gateway SHALL expose REST endpoints for adapter discovery and management: `GET /v1/adapters` (list all), `GET /v1/adapters/{name}/metadata` (details), `GET /v1/adapters/{name}/health` (health check), `GET /v1/adapters/{name}/config-schema` (configuration schema). All endpoints MUST return JSON:API formatted responses.

#### Scenario: List available adapters

- **GIVEN** 11 biomedical adapters registered
- **WHEN** client requests `GET /v1/adapters?domain=biomedical`
- **THEN** response includes all biomedical adapter metadata
- **AND** response format is JSON:API with `data`, `links`, `meta`
- **AND** each adapter includes `name`, `version`, `capabilities`, `rate_limit_default`

#### Scenario: Get adapter configuration schema

- **GIVEN** ClinicalTrialsAdapter with pydantic-settings configuration
- **WHEN** client requests `GET /v1/adapters/clinicaltrials/config-schema`
- **THEN** response includes JSON Schema derived from Pydantic model
- **AND** schema includes required fields, types, and descriptions
- **AND** sensitive fields (api_key) are marked as `writeOnly`

#### Scenario: Adapter health check via API

- **GIVEN** multiple registered adapters
- **WHEN** client requests `GET /v1/adapters/clinicaltrials/health`
- **THEN** response includes current health status
- **AND** response includes response time in milliseconds
- **AND** if unhealthy, response includes error message

### Requirement: GraphQL Adapter Queries

The GraphQL API SHALL support querying adapters via `query { adapters(domain: BIOMEDICAL) { name version capabilities healthStatus } }`. Adapter health status SHALL be resolved asynchronously using DataLoader pattern to prevent N+1 queries.

#### Scenario: GraphQL adapter query with health status

- **GIVEN** multiple adapters registered
- **WHEN** client executes GraphQL query for adapters with health status
- **THEN** DataLoader batches health check requests
- **AND** response includes adapter metadata with current health status
- **AND** query completes within P95 < 500ms

#### Scenario: GraphQL adapter filtering

- **GIVEN** adapters across multiple domains
- **WHEN** client queries `adapters(domain: BIOMEDICAL, capability: PDF_SUPPORT)`
- **THEN** only biomedical adapters with PDF support are returned
- **AND** results are sorted alphabetically by name

### Requirement: OpenAPI Specification Updates

The OpenAPI specification MUST include adapter metadata endpoints with complete request/response schemas. All adapter-related endpoints SHALL include examples for each supported domain (biomedical, financial, legal).

#### Scenario: OpenAPI schema includes adapter endpoints

- **GIVEN** the OpenAPI specification is generated
- **WHEN** a developer views the specification
- **THEN** adapter endpoints are documented under `/v1/adapters` path
- **AND** each endpoint includes request/response examples
- **AND** schema definitions for `AdapterMetadata` are included
