# AsyncAPI Events Specification

## ADDED Requirements

### Requirement: Server-Sent Events (SSE)

The system SHALL provide SSE endpoints for streaming real-time job updates to clients.

#### Scenario: Job event stream

- **WHEN** connecting to `/jobs/{id}/events`
- **THEN** client MUST receive SSE stream with job status updates

#### Scenario: Event format

- **WHEN** job status changes
- **THEN** SSE event MUST be JSON with type, jobId, status, progress fields

### Requirement: AsyncAPI Documentation

The system SHALL document all event channels and message schemas using AsyncAPI specification.

#### Scenario: AsyncAPI spec availability

- **WHEN** accessing `/docs/asyncapi`
- **THEN** AsyncAPI UI MUST display all channels and message schemas

#### Scenario: Event payload schemas

- **WHEN** viewing event definition
- **THEN** complete JSON schema MUST be provided for each event type

### Requirement: Event Types

The system SHALL emit events for job lifecycle (started, progress, completed, failed).

#### Scenario: Job started event

- **WHEN** job begins processing
- **THEN** `jobs.started` event MUST be emitted with job metadata

#### Scenario: Progress updates

- **WHEN** job makes progress
- **THEN** `jobs.progress` event MUST include percentage and current step

#### Scenario: Job completion

- **WHEN** job finishes
- **THEN** `jobs.completed` event MUST include result summary or error details
