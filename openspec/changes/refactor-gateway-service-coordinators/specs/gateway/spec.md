## ADDED Requirements

### Requirement: Modular Gateway Service Architecture
The gateway service SHALL be decomposed into focused coordinators that encapsulate domain logic, ledger operations, and error handling for specific workflows, replacing the monolithic service locator pattern.

#### Scenario: Coordinator-Based Service Composition
- **GIVEN** the modular gateway architecture with focused coordinators
- **WHEN** handling API requests across different protocols
- **THEN** each workflow SHALL use a dedicated coordinator (IngestionCoordinator, EmbeddingCoordinator, etc.)
- **AND** coordinators SHALL encapsulate domain logic, ledger operations, and error mapping
- **AND** protocol handlers SHALL depend on narrow coordinator interfaces
- **AND** coordinators SHALL compose through dependency injection

#### Scenario: Coordinator Interface Design
- **GIVEN** a coordinator implementing a specific workflow interface
- **WHEN** used by protocol handlers
- **THEN** it SHALL expose a narrow, focused interface (e.g., `embed(texts, namespace) -> EmbeddingResult`)
- **AND** hide implementation details like adapter discovery, validation, and persistence
- **AND** provide structured result objects for consistent response formatting
- **AND** handle error translation to protocol-agnostic error types

#### Scenario: Coordinator Lifecycle Management
- **WHEN** coordinators are initialized and used
- **THEN** they SHALL follow a consistent lifecycle (initialize → health_check → cleanup)
- **AND** support dependency injection for configuration and collaborators
- **AND** provide health monitoring and performance metrics
- **AND** enable graceful shutdown and resource cleanup

### Requirement: JobLifecycleManager Service
The system SHALL provide a dedicated JobLifecycleManager that encapsulates job state transitions, ledger operations, and event streaming for all workflow coordinators.

#### Scenario: Centralized Job Management
- **GIVEN** the JobLifecycleManager service
- **WHEN** coordinators need to manage job lifecycle
- **THEN** it SHALL provide methods for job creation, state transitions, and completion
- **AND** handle ledger operations and event streaming integration
- **AND** enforce job state machine rules and validation
- **AND** provide audit trails and debugging capabilities

#### Scenario: Job State Machine Implementation
- **GIVEN** job state transitions managed by JobLifecycleManager
- **WHEN** jobs progress through their lifecycle
- **THEN** the manager SHALL enforce valid state transitions (queued → processing → completed/failed/cancelled)
- **AND** validate transition rules and business logic constraints
- **AND** maintain job metadata and timing information
- **AND** provide idempotency guarantees for duplicate operations

#### Scenario: Event Streaming Integration
- **GIVEN** job lifecycle events managed by JobLifecycleManager
- **WHEN** jobs progress through stages
- **THEN** it SHALL emit structured events for real-time progress updates
- **AND** integrate with Server-Sent Events for live job monitoring
- **AND** provide event correlation and debugging capabilities
- **AND** support event filtering and subscription management

## MODIFIED Requirements

### Requirement: Gateway Service Architecture
The gateway service architecture SHALL use focused coordinators that encapsulate domain logic and job management, replacing the monolithic service locator pattern with composable, testable components.

#### Scenario: Coordinator Composition and Dependencies
- **WHEN** the gateway service is composed from coordinators
- **THEN** each coordinator SHALL have a single, well-defined responsibility
- **AND** dependencies SHALL be injected through constructor parameters or factory methods
- **AND** coordinators SHALL be independently testable with mocked dependencies
- **AND** the composition SHALL maintain backward compatibility with existing protocol handlers

#### Scenario: Error Handling and Recovery
- **GIVEN** coordinators with encapsulated error handling
- **WHEN** errors occur during workflow execution
- **THEN** coordinators SHALL translate domain exceptions to protocol-agnostic error types
- **AND** provide structured error information for consistent API responses
- **AND** implement recovery strategies and fallback mechanisms
- **AND** maintain error context for debugging and monitoring

#### Scenario: Performance Monitoring and Metrics
- **GIVEN** coordinators with integrated performance monitoring
- **WHEN** workflows execute across coordinators
- **THEN** each coordinator SHALL collect and report performance metrics
- **AND** support distributed tracing and correlation ID propagation
- **AND** provide performance profiling and optimization capabilities
- **AND** enable alerting based on coordinator-specific thresholds

## RENAMED Requirements

- FROM: `### Requirement: Monolithic Gateway Service`
- TO: `### Requirement: Coordinator-Based Gateway Architecture`
