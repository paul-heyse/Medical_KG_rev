## ADDED Requirements

### Requirement: Gateway Integration Testing

The test suite SHALL include comprehensive testing of gateway pipeline resolution and job submission for PDF documents.

#### Scenario: PDF pipeline resolution testing

- **GIVEN** a PDF document ingestion request
- **WHEN** the gateway processes the request
- **THEN** tests verify correct pipeline resolution to `pdf-two-phase`
- **AND** validate job submission to the orchestrator
- **AND** confirm proper error handling for resolution failures
- **AND** test fallback behavior for non-PDF documents
