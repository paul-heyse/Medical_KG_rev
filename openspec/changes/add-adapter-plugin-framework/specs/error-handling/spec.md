# Error Handling Specification Deltas

## ADDED Requirements

### Requirement: Adapter Error Taxonomy

Adapter errors SHALL be classified into categories: `TRANSIENT` (retryable), `PERMANENT` (not retryable), `RATE_LIMIT` (backoff required), `AUTH` (credential issue), `VALIDATION` (data issue). Each error category MUST have a corresponding HTTP status code and retry policy.

#### Scenario: Transient error with retry

- **GIVEN** adapter fetch returns HTTP 503 (Service Unavailable)
- **WHEN** error is classified
- **THEN** error category is `TRANSIENT`
- **AND** resilience layer retries with exponential backoff
- **AND** error is logged with `severity=warning`

#### Scenario: Permanent error without retry

- **GIVEN** adapter fetch returns HTTP 404 (Not Found)
- **WHEN** error is classified
- **THEN** error category is `PERMANENT`
- **AND** no retry is attempted
- **AND** error is logged with `severity=error`
- **AND** job is moved to dead letter queue

#### Scenario: Rate limit error with backoff

- **GIVEN** adapter fetch returns HTTP 429 (Too Many Requests)
- **WHEN** error is classified
- **THEN** error category is `RATE_LIMIT`
- **AND** adapter waits for time specified in `Retry-After` header
- **AND** request is retried after backoff period
- **AND** Prometheus metric `adapter_rate_limit_hits_total` increments

### Requirement: Structured Error Responses

All adapter errors SHALL return structured error responses following RFC 7807 Problem Details format. Error responses MUST include error code, human-readable message, adapter name, timestamp, and correlation ID.

#### Scenario: Adapter error response format

- **GIVEN** adapter encounters validation error
- **WHEN** error is returned to client
- **THEN** response follows RFC 7807 format
- **AND** response includes `type`, `title`, `status`, `detail`, `instance`
- **AND** extensions include `adapter_name`, `correlation_id`, `timestamp`
- **AND** HTTP status code matches error category

#### Scenario: Error message localization

- **GIVEN** client provides `Accept-Language: es` header
- **WHEN** adapter error occurs
- **THEN** error message is returned in Spanish if available
- **AND** fallback to English if translation unavailable
- **AND** error code remains language-independent

### Requirement: Error Observability

All adapter errors SHALL be tracked via Prometheus metrics with labels for adapter name, error category, and HTTP status code. Error rates exceeding 5% SHALL trigger alerts. Detailed error information SHALL be sent to Sentry for investigation.

#### Scenario: Error metrics tracking

- **GIVEN** adapter encounters multiple errors
- **WHEN** errors are processed
- **THEN** Prometheus counter `adapter_errors_total{adapter="clinicaltrials", category="TRANSIENT"}` increments
- **AND** error rate is calculated as errors/total_requests
- **AND** alert triggers if error rate > 5% for 5 minutes

#### Scenario: Sentry error reporting

- **GIVEN** adapter encounters unexpected exception
- **WHEN** error is caught by exception handler
- **THEN** full stack trace is sent to Sentry
- **AND** Sentry event includes adapter name, request details, and context
- **AND** similar errors are grouped by fingerprint

### Requirement: Circuit Breaker State Management

The circuit breaker SHALL track failure rates per adapter and open when threshold is exceeded. Circuit breaker state transitions SHALL be: CLOSED → OPEN → HALF_OPEN → CLOSED. State transitions MUST be logged and exposed via metrics.

#### Scenario: Circuit breaker state transitions

- **GIVEN** adapter with failure threshold of 50% over 10 requests
- **WHEN** 6 out of 10 requests fail
- **THEN** circuit breaker transitions from CLOSED to OPEN
- **AND** metric `adapter_circuit_breaker_state{adapter="openfda"}` = 1 (OPEN)
- **AND** log entry records state transition with timestamp

#### Scenario: Circuit breaker half-open state

- **GIVEN** circuit breaker is OPEN for 30 seconds
- **WHEN** timeout expires
- **THEN** circuit breaker transitions to HALF_OPEN
- **AND** next request is allowed through as test
- **AND** if request succeeds, transition to CLOSED
- **AND** if request fails, return to OPEN state
