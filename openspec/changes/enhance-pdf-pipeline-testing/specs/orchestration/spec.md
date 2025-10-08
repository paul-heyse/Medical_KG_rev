## ADDED Requirements

### Requirement: Comprehensive PDF Pipeline Testing

The test suite SHALL include comprehensive end-to-end testing of the PDF processing pipeline, including real downloads, MinerU processing, state transitions, and sensor-based resumption.

#### Scenario: End-to-end PDF pipeline testing

- **GIVEN** a complete PDF processing environment
- **WHEN** integration tests execute the full PDF pipeline
- **THEN** they test real PDF downloads from actual URLs
- **AND** validate MinerU processing with real PDF files
- **AND** verify all ledger state transitions occur correctly
- **AND** test sensor-based job resumption functionality

#### Scenario: Two-phase execution testing

- **GIVEN** a PDF pipeline with gate definitions
- **WHEN** tests execute the pipeline
- **THEN** they validate pre-gate stage execution
- **AND** test gate condition evaluation and waiting
- **AND** verify sensor-triggered resume job execution
- **AND** confirm post-gate stages complete successfully

### Requirement: State Management Testing

The test suite SHALL validate ledger state management throughout PDF pipeline execution, ensuring proper state transitions and consistency.

#### Scenario: State transition validation

- **GIVEN** a PDF pipeline execution
- **WHEN** stages complete or fail
- **THEN** tests verify correct ledger state updates
- **AND** validate state transition rules are enforced
- **AND** test state consistency across execution phases
- **AND** verify state cleanup for failed operations

## MODIFIED Requirements

### Requirement: Integration Test Coverage

Integration tests SHALL cover the complete PDF processing workflow including download, MinerU processing, state management, and sensor integration.

#### Scenario: Enhanced integration testing

- **GIVEN** the PDF processing pipeline
- **WHEN** integration tests are executed
- **THEN** they exercise real file I/O operations
- **AND** test actual service integrations (MinerU, storage)
- **AND** validate state management and sensor triggering
- **AND** include performance and error scenario testing

## REMOVED Requirements

### Requirement: Simulation-Only PDF Testing

**Reason**: Replaced by comprehensive testing that exercises real PDF processing components, state management, and pipeline execution to ensure the PDF pipeline functions correctly in production scenarios
**Migration**: Existing simulation-based tests continue to validate infrastructure while comprehensive tests are added to validate end-to-end functionality
