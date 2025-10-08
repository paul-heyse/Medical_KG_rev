## 1. Design & Analysis Phase

- [ ] 1.1 Analyze current `_submit_dagster_job` implementation and identify separation boundaries
- [ ] 1.2 Design `DagsterIngestionClient` interface with clean `submit(...) -> DagsterSubmissionResult` contract
- [ ] 1.3 Identify pipeline resolution, domain mapping, and telemetry concerns for encapsulation
- [ ] 1.4 Design typed result hierarchy (`DagsterSubmissionResult`, `SubmissionSuccess`, `SubmissionDuplicate`, `SubmissionFailure`)
- [ ] 1.5 Plan integration with existing `JobLifecycleManager` for consistent job metadata handling
- [ ] 1.6 Design error handling strategy for Dagster-specific failures vs gateway-level failures
- [ ] 1.7 Plan testing strategy for orchestration client with Dagster mocking
- [ ] 1.8 Create interface contract documentation and usage examples
- [ ] 1.9 Design performance monitoring integration for orchestration operations
- [ ] 1.10 Plan configuration management for pipeline selection and domain mapping

## 2. Core Interface Implementation

- [ ] 2.1 Create `DagsterSubmissionResult` base class and concrete result types
- [ ] 2.2 Define `DagsterIngestionClient` abstract interface with `submit` method contract
- [ ] 2.3 Create `DagsterIngestionRequest` dataclass for structured submission parameters
- [ ] 2.4 Implement result validation and type safety checks
- [ ] 2.5 Add result metadata enrichment with job correlation and timing information
- [ ] 2.6 Create result serialization for logging and debugging
- [ ] 2.7 Implement result comparison utilities for testing and validation
- [ ] 2.8 Add result error context preservation for troubleshooting
- [ ] 2.9 Create result factory methods for different submission outcomes
- [ ] 2.10 Implement result metrics collection and performance tracking

## 3. Pipeline Resolution Encapsulation

- [ ] 3.1 Create `PipelineResolver` class to handle pipeline topology selection and configuration
- [ ] 3.2 Implement dataset-to-pipeline mapping logic with fallback strategies
- [ ] 3.3 Add pipeline validation and compatibility checking
- [ ] 3.4 Create pipeline configuration loading and caching mechanisms
- [ ] 3.5 Implement pipeline version management and compatibility checks
- [ ] 3.6 Add pipeline performance monitoring and optimization
- [ ] 3.7 Create pipeline error handling with graceful degradation
- [ ] 3.8 Implement pipeline configuration hot-reloading capabilities
- [ ] 3.9 Add pipeline dependency resolution and ordering logic
- [ ] 3.10 Create pipeline debugging and introspection utilities
- [ ] 3.11 Implement PDF download stage integration with JobLedger.set_pdf_downloaded
- [ ] 3.12 Add PDF gate stage integration with JobLedger.pdf_ir_ready sensor logic
- [ ] 3.13 Create stage factory extensions for download and gate stage implementations
- [ ] 3.14 Add PDF pipeline instantiation support for pdf-two-phase topology

## 4. Domain Resolution Encapsulation

- [ ] 4.1 Create `DomainResolver` class to handle adapter domain mapping and configuration
- [ ] 4.2 Implement adapter discovery and capability matching logic
- [ ] 4.3 Add domain validation and normalization utilities
- [ ] 4.4 Create domain-specific configuration loading and validation
- [ ] 4.5 Implement domain compatibility checking and fallback strategies
- [ ] 4.6 Add domain performance monitoring and optimization
- [ ] 4.7 Create domain error handling with contextual information
- [ ] 4.8 Implement domain configuration caching and invalidation
- [ ] 4.9 Add domain debugging and introspection capabilities
- [ ] 4.10 Create domain migration utilities for adapter evolution

## 5. Telemetry Encapsulation

- [ ] 5.1 Create `OrchestrationTelemetry` class to handle metrics, tracing, and logging
- [ ] 5.2 Implement submission timing and performance metrics collection
- [ ] 5.3 Add distributed tracing integration for orchestration flows
- [ ] 5.4 Create telemetry configuration and filtering logic
- [ ] 5.5 Implement telemetry aggregation and reporting utilities
- [ ] 5.6 Add telemetry error correlation and debugging support
- [ ] 5.7 Create telemetry performance optimization strategies
- [ ] 5.8 Implement telemetry configuration hot-reloading
- [ ] 5.9 Add telemetry security and privacy considerations
- [ ] 5.10 Create telemetry debugging and troubleshooting tools

## 6. DagsterIngestionClient Implementation

- [ ] 6.1 Create concrete `DagsterIngestionClient` implementation using existing Dagster integration
- [ ] 6.2 Implement pipeline resolution integration with `PipelineResolver`
- [ ] 6.3 Add domain resolution integration with `DomainResolver`
- [ ] 6.4 Integrate telemetry collection with `OrchestrationTelemetry`
- [ ] 6.5 Implement error handling with contextual information and recovery strategies
- [ ] 6.6 Add submission result validation and enrichment
- [ ] 6.7 Create submission retry logic with exponential backoff
- [ ] 6.8 Implement submission deduplication using job ledger integration
- [ ] 6.9 Add submission performance monitoring and optimization
- [ ] 6.10 Create submission debugging and introspection capabilities

## 7. Integration with IngestionCoordinator

- [ ] 7.1 Update `IngestionCoordinator` to use `DagsterIngestionClient` interface
- [ ] 7.2 Implement result transformation from `DagsterSubmissionResult` to API response format
- [ ] 7.3 Add error translation from orchestration errors to coordinator error types
- [ ] 7.4 Create coordinator configuration for client selection and fallback strategies
- [ ] 7.5 Implement coordinator performance monitoring integration
- [ ] 7.6 Add coordinator debugging and troubleshooting integration
- [ ] 7.7 Create coordinator health checking for orchestration client dependencies
- [ ] 7.8 Implement coordinator configuration hot-reloading for orchestration settings
- [ ] 7.9 Add coordinator metrics aggregation from orchestration client
- [ ] 7.10 Create coordinator operational monitoring and alerting

## 8. Testing & Validation

- [ ] 8.1 Create comprehensive unit tests for `DagsterSubmissionResult` hierarchy
- [ ] 8.2 Create unit tests for `PipelineResolver` with mocked pipeline configurations
- [ ] 8.3 Create unit tests for `DomainResolver` with mocked adapter registry
- [ ] 8.4 Create unit tests for `OrchestrationTelemetry` with mocked metrics collection
- [ ] 8.5 Create unit tests for `DagsterIngestionClient` with mocked Dagster dependencies
- [ ] 8.6 Create integration tests for complete submission flow with real Dagster
- [ ] 8.7 Create performance tests for orchestration client overhead
- [ ] 8.8 Test error handling and recovery mechanisms
- [ ] 8.9 Test configuration management and hot-reloading
- [ ] 8.10 Test observability and debugging capabilities

## 9. Migration & Deployment

- [ ] 9.1 Create migration script to extract `_submit_dagster_job` logic into `DagsterIngestionClient`
- [ ] 9.2 Update `IngestionCoordinator` to use new orchestration client interface
- [ ] 9.3 Add orchestration client configuration to application settings
- [ ] 9.4 Update Docker Compose and Kubernetes deployments for new orchestration structure
- [ ] 9.5 Create orchestration client health checks for deployment validation
- [ ] 9.6 Add orchestration metrics to monitoring dashboards
- [ ] 9.7 Create orchestration debugging tools and operational runbooks
- [ ] 9.8 Update API documentation for new orchestration-based ingestion
- [ ] 9.9 Create orchestration migration guide for existing integrations
- [ ] 9.10 Add orchestration performance tuning and optimization guides

## 10. Documentation & Developer Experience

- [ ] 10.1 Create comprehensive orchestration client architecture documentation
- [ ] 10.2 Add orchestration interface documentation with usage examples
- [ ] 10.3 Create orchestration interaction diagrams and sequence flows
- [ ] 10.4 Update API documentation for orchestration-based ingestion endpoints
- [ ] 10.5 Create orchestration debugging and troubleshooting guides
- [ ] 10.6 Add orchestration performance tuning documentation
- [ ] 10.7 Create orchestration extension and customization guides
- [ ] 10.8 Add orchestration migration documentation for existing code
- [ ] 10.9 Create orchestration testing best practices and examples
- [ ] 10.10 Add orchestration operational monitoring and alerting guides
