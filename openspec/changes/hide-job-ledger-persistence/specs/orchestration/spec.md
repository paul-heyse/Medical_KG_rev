## ADDED Requirements

### Requirement: Repository Pattern for Job Ledger

The job ledger SHALL use a repository pattern to separate persistence concerns from business logic orchestration, with clear interfaces for state management.

#### Scenario: Repository Interface Definition

- **GIVEN** the job ledger architecture using repository pattern
- **WHEN** managing job state and transitions
- **THEN** persistence logic SHALL be encapsulated in repository implementations
- **AND** business logic SHALL interact through standardized repository interfaces
- **AND** state transitions SHALL be validated at the repository layer

#### Scenario: Invariant Enforcement in Repository

- **WHEN** performing job state transitions
- **THEN** the repository SHALL enforce business invariants
- **AND** validate PDF gate requirements before allowing stage progression
- **AND** enforce retry count limits and status transition rules
- **AND** provide clear error messages for invalid state transitions

#### Scenario: Multiple Repository Implementations

- **GIVEN** the repository interface design
- **WHEN** deploying the orchestration system
- **THEN** different persistence backends SHALL be selectable
- **AND** in-memory implementation SHALL be available for testing
- **AND** durable storage implementations SHALL be pluggable
- **AND** repository behavior SHALL be consistent across implementations

### Requirement: LedgerRepository Interface Design

The LedgerRepository SHALL provide a clear interface for job lifecycle management, state transitions, and metadata tracking.

#### Scenario: Job Lifecycle Management

- **GIVEN** the LedgerRepository interface
- **WHEN** managing job execution state
- **THEN** it SHALL provide methods for job creation, updates, and queries
- **AND** support stage transition tracking and validation
- **AND** enable metadata and history management
- **AND** provide thread-safe operations for concurrent access

#### Scenario: State Transition Validation

- **GIVEN** job state transition requests
- **WHEN** validating and applying state changes
- **THEN** the repository SHALL enforce valid transition rules
- **AND** validate PDF-related invariants before allowing progression
- **AND** maintain job history and audit trails
- **AND** provide atomic state updates with rollback capability

## MODIFIED Requirements

### Requirement: Job State Management Architecture

The orchestration system SHALL use a repository pattern for job ledger management to separate persistence concerns from business logic and enable durable storage.

#### Scenario: Repository Integration with Runtime

- **WHEN** the Dagster runtime manages job execution
- **THEN** it SHALL use the LedgerRepository interface for all state operations
- **AND** business logic SHALL be independent of persistence implementation
- **AND** state transitions SHALL be validated at the repository layer
- **AND** error handling SHALL use repository-specific error types

#### Scenario: State Management and Persistence

- **GIVEN** the repository-based ledger architecture
- **WHEN** persisting job state across pipeline executions
- **THEN** the repository SHALL handle durable storage concerns
- **AND** maintain state consistency across failures and restarts
- **AND** provide query capabilities for job monitoring and debugging
- **AND** support both transient and persistent storage backends

## RENAMED Requirements

- FROM: `### Requirement: In-Memory Job Ledger with Mixed Concerns`
- TO: `### Requirement: Repository Pattern for Job State Management`
