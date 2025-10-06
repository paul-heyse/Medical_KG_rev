# Foundation Infrastructure Specification

## ADDED Requirements

### Requirement: Federated Data Model

The system SHALL provide a unified Intermediate Representation (IR) with core entities and domain-specific overlays using Pydantic models.

#### Scenario: Medical document with FHIR overlay

- **WHEN** a clinical trial is ingested
- **THEN** it MUST create a Document with medical metadata conforming to FHIR ResearchStudy structure

#### Scenario: Document ID generation

- **WHEN** any document is created
- **THEN** it MUST have a globally unique ID in format `{source}:{source_id}#{version}:{hash12}`

#### Scenario: Cross-domain coexistence

- **WHEN** medical and financial documents exist in the system
- **THEN** both MUST use the same core Document model with domain-specific metadata

### Requirement: Pydantic Model Validation

All data models SHALL use Pydantic v2 with strict validation and comprehensive field validators.

#### Scenario: Required field enforcement

- **WHEN** creating a Document without required fields
- **THEN** Pydantic MUST raise ValidationError before any processing

#### Scenario: Type coercion prevention

- **WHEN** a field receives wrong type (e.g., string for integer)
- **THEN** Pydantic MUST reject with ValidationError in strict mode

#### Scenario: Custom validator execution

- **WHEN** an NCT ID is provided
- **THEN** a validator MUST check format matches `NCT\d{8}` pattern

### Requirement: Adapter SDK Base Classes

The system SHALL provide abstract base classes for implementing data source adapters with standardized lifecycle methods.

#### Scenario: Adapter implements required methods

- **WHEN** a new adapter is created
- **THEN** it MUST implement fetch(), parse(), validate(), and write() methods from BaseAdapter

#### Scenario: Adapter registration

- **WHEN** an adapter is defined
- **THEN** it MUST be discoverable via the adapter registry by source name

#### Scenario: YAML-driven adapter configuration

- **WHEN** a simple REST API adapter is needed
- **THEN** it MAY be defined entirely in YAML without Python code

### Requirement: Configuration Management

The system SHALL use environment-based configuration with Pydantic Settings and support for secrets management.

#### Scenario: Environment-specific settings

- **WHEN** running in production
- **THEN** configuration MUST load from environment variables or Vault

#### Scenario: Required secrets validation

- **WHEN** application starts
- **THEN** it MUST validate all required secrets are present or fail fast

#### Scenario: Multi-domain configuration

- **WHEN** multiple knowledge domains are enabled
- **THEN** configuration MUST specify which domain adapters are active

### Requirement: Structured Logging and Tracing

The system SHALL provide structured logging with JSON output and OpenTelemetry integration for distributed tracing.

#### Scenario: Sensitive data scrubbing

- **WHEN** logging user data
- **THEN** PII and secrets MUST be scrubbed from log output

#### Scenario: Trace context propagation

- **WHEN** making HTTP or gRPC calls
- **THEN** OpenTelemetry trace context MUST be propagated via headers

#### Scenario: Correlation IDs

- **WHEN** any request is processed
- **THEN** a correlation ID MUST be generated and included in all related logs

### Requirement: HTTP Client with Resilience

The system SHALL provide a shared HTTP client with retry logic, exponential backoff, timeout handling, and rate limiting.

#### Scenario: Transient error retry

- **WHEN** an HTTP request receives 5xx error
- **THEN** client MUST retry with exponential backoff up to configured max attempts

#### Scenario: Timeout enforcement

- **WHEN** an HTTP request exceeds configured timeout
- **THEN** client MUST cancel request and raise TimeoutError

#### Scenario: Rate limit respect

- **WHEN** making requests to rate-limited API
- **THEN** client MUST track request counts and delay to stay within limits

#### Scenario: Circuit breaker

- **WHEN** an endpoint has failed multiple times
- **THEN** client MAY open circuit breaker to fail fast without attempting requests

### Requirement: Input Validation Utilities

The system SHALL provide validation utilities for common biomedical identifiers and data formats.

#### Scenario: NCT ID validation

- **WHEN** validating an NCT ID
- **THEN** validator MUST accept format `NCT\d{8}` and reject invalid formats

#### Scenario: DOI validation

- **WHEN** validating a DOI
- **THEN** validator MUST accept format `10.\d{4,}/.*` and handle URL vs plain formats

#### Scenario: PMCID validation

- **WHEN** validating a PubMed Central ID
- **THEN** validator MUST accept format `PMC\d+`

#### Scenario: Span coordinate validation

- **WHEN** validating a text span
- **THEN** validator MUST ensure start < end and both are within document bounds

### Requirement: RFC 7807 Problem Details

All error responses SHALL use RFC 7807 Problem Details format for consistent error handling.

#### Scenario: Validation error response

- **WHEN** validation fails
- **THEN** error MUST include type, title, status, detail, and field-specific errors

#### Scenario: Error type URIs

- **WHEN** any error occurs
- **THEN** type field MUST contain a URI identifying the error class

#### Scenario: Machine-readable error codes

- **WHEN** client receives an error
- **THEN** it MUST include a code field for programmatic handling

### Requirement: Storage Abstractions

The system SHALL provide abstract interfaces for object storage, caching, and job ledger without coupling to specific implementations.

#### Scenario: Object store abstraction

- **WHEN** storing a PDF
- **THEN** code MUST use ObjectStore interface allowing S3 or MinIO implementations

#### Scenario: Async storage operations

- **WHEN** performing storage operations
- **THEN** all methods MUST support async/await for non-blocking I/O

#### Scenario: Storage retry logic

- **WHEN** storage operation fails transiently
- **THEN** abstraction MUST retry with backoff

### Requirement: Provenance Tracking Models

The system SHALL track provenance for all extracted data including source, extraction method, model version, and timestamp.

#### Scenario: Extraction activity recording

- **WHEN** data is extracted
- **THEN** an ExtractionActivity node MUST be created with model_name, prompt_version, and timestamp

#### Scenario: Data source tracking

- **WHEN** ingesting from external API
- **THEN** DataSource metadata MUST include api_version, fetch_timestamp, and original_url

#### Scenario: Immutable provenance

- **WHEN** provenance is recorded
- **THEN** it MUST NOT be modifiable after creation

### Requirement: Fail-Fast Validation

The system SHALL validate all inputs at entry points and reject invalid data immediately before processing.

#### Scenario: Invalid ID rejection

- **WHEN** an invalid NCT ID is submitted to ingest endpoint
- **THEN** system MUST return 400 error without making external API calls

#### Scenario: Schema validation before save

- **WHEN** writing to storage
- **THEN** data MUST pass Pydantic validation or operation fails

#### Scenario: Span validation

- **WHEN** extraction includes spans
- **THEN** all span coordinates MUST be validated before accepting results
