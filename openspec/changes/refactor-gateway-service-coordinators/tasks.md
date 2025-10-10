## 1. Design & Analysis Phase

- [ ] 1.1 Analyze current `GatewayService` class structure and identify responsibility boundaries using dependency analysis tools (e.g., `pyan3` for call graphs, `objgraph` for object relationships)
- [ ] 1.2 Map existing methods to proposed coordinator responsibilities (ingestion, embedding, retrieval, chunking, entity linking, extraction) with specific method signatures and dependencies
- [x] 1.3 Identify shared patterns in job lifecycle management (`_new_job`, `_complete_job`, `_fail_job`) for extraction into `JobLifecycleManager`
- [ ] 1.4 Design narrow interface contracts for each coordinator (e.g., `IngestionCoordinator.ingest(...) -> IngestionResult`)
- [ ] 1.5 Plan dependency injection strategy for coordinator composition and service locator replacement
- [ ] 1.6 Design error mapping interfaces for consistent error handling across coordinators
- [ ] 1.7 Plan backward compatibility layer for existing protocol handler dependencies
- [ ] 1.8 Create coordinator interface hierarchy and dependency graph documentation
- [ ] 1.9 Design coordinator factory pattern for service composition and configuration
- [ ] 1.10 Plan performance monitoring and metrics collection for individual coordinators

### Critical Library Integration Requirements

- [ ] 1.11 **Integrate `pydantic>=2.7,<2.11`**: Design typed coordinator interfaces and result models
- [x] 1.12 **Integrate `structlog`**: Add structured logging for coordinator operations and debugging
- [x] 1.13 **Integrate `tenacity>=9.1.2`**: Add retry logic for coordinator operations and external calls
- [ ] 1.14 **Integrate `pluggy>=1.6.0`**: Ensure coordinator composition works with plugin system
- [x] 1.15 **Integrate `aiolimiter>=1.2.1`**: Implement rate limiting for coordinator operations
- [x] 1.16 **Integrate `pybreaker>=1.4.1`**: Add circuit breaker patterns for coordinator resilience
- [x] 1.17 **Integrate `prometheus-client`**: Add metrics collection for coordinator performance

## 2. Core Infrastructure Implementation

- [x] 2.1 Create `JobLifecycleManager` class to encapsulate job creation, state transitions, and ledger operations with specific methods: `create_job()`, `start_job()`, `complete_job()`, `fail_job()`, `cancel_job()` in `src/Medical_KG_rev/gateway/coordinators/job_lifecycle.py`
- [ ] 2.2 Implement job state machine with validation rules (queued → processing → completed/failed/cancelled) including state transition validation, business rule enforcement, and rollback capabilities
- [x] 2.3 Add event streaming integration for real-time job progress updates via SSE
- [x] 2.4 Create job metadata enrichment utilities for tenant, correlation, and timing information
- [x] 2.5 Implement job deduplication logic for idempotent operations
- [x] 2.6 Add comprehensive error handling with structured error types and recovery strategies
- [ ] 2.7 Create job metrics collection and performance monitoring integration
- [ ] 2.8 Implement job cleanup and resource management for failed/cancelled operations
- [ ] 2.9 Add job audit trail logging for regulatory compliance and debugging
- [ ] 2.10 Create job persistence layer for crash recovery and historical analysis

## 3. Coordinator Interface Definitions

- [x] 3.1 Define `BaseCoordinator` abstract class with common lifecycle methods (initialize, health_check, cleanup)
- [ ] 3.2 Create `IngestionCoordinator` interface with `ingest(dataset, request) -> IngestionResult` contract
- [x] 3.3 Create `EmbeddingCoordinator` interface with `embed(texts, namespace) -> EmbeddingResult` contract
- [ ] 3.4 Create `RetrievalCoordinator` interface with `retrieve(query, filters) -> RetrievalResult` contract
- [x] 3.5 Create `ChunkingCoordinator` interface with `chunk(document, profile) -> ChunkingResult` contract
- [ ] 3.6 Create `ExtractionCoordinator` interface with `extract(document, entities) -> ExtractionResult` contract
- [x] 3.7 Define result dataclass hierarchy (`BaseResult`, `IngestionResult`, `EmbeddingResult`, etc.)
- [x] 3.8 Add coordinator-specific error types extending base error hierarchy
- [x] 3.9 Implement coordinator metrics collection and performance monitoring
- [x] 3.10 Create coordinator configuration interfaces and validation schemas

## 4. IngestionCoordinator Implementation

- [ ] 4.1 Extract ingestion logic from `GatewayService.ingest` into `IngestionCoordinator` class
- [ ] 4.2 Implement adapter discovery and request construction logic
- [ ] 4.3 Add dataset validation and normalization utilities
- [ ] 4.4 Create ingestion pipeline orchestration (validate → fetch → parse → store)
- [ ] 4.5 Implement error handling with retry logic and circuit breaker patterns
- [ ] 4.6 Add ingestion progress tracking and intermediate result reporting
- [ ] 4.7 Create ingestion result transformation for API response formatting
- [ ] 4.8 Implement batch ingestion support with parallel processing
- [ ] 4.9 Add ingestion quota management and rate limiting integration
- [ ] 4.10 Create ingestion metrics collection and performance optimization
- [ ] 4.11 Add PDF document detection logic for OpenAlex adapter responses
- [ ] 4.12 Implement document_type="pdf" flagging for works with downloadable PDFs
- [ ] 4.13 Add pipeline routing logic to direct PDF documents to pdf-two-phase topology
- [ ] 4.14 Create PDF-specific job lifecycle management for two-phase processing

## 5. EmbeddingCoordinator Implementation

- [x] 5.1 Extract embedding logic from `GatewayService.embed` into `EmbeddingCoordinator` class
- [x] 5.2 Implement namespace validation and routing logic
- [x] 5.3 Create text preprocessing and normalization utilities
- [x] 5.4 Add embedding batch processing with GPU service integration
- [x] 5.5 Implement embedding persistence with vector storage abstraction
- [ ] 5.6 Create embedding result validation and quality metrics
- [x] 5.7 Add embedding error handling with fallback strategies
- [ ] 5.8 Implement embedding caching for performance optimization
- [ ] 5.9 Create embedding metrics collection and monitoring
- [ ] 5.10 Add embedding configuration management and namespace switching

## 6. RetrievalCoordinator Implementation

- [ ] 6.1 Extract retrieval logic from `GatewayService.retrieve` into `RetrievalCoordinator` class
- [ ] 6.2 Implement query parsing and normalization utilities
- [ ] 6.3 Create multi-strategy search orchestration (BM25 + SPLADE + dense vectors)
- [ ] 6.4 Add result fusion and ranking with Reciprocal Rank Fusion (RRF)
- [ ] 6.5 Implement result post-processing (highlighting, entity linking, confidence scoring)
- [ ] 6.6 Create retrieval performance monitoring and latency optimization
- [ ] 6.7 Add retrieval caching strategies for frequently accessed queries
- [ ] 6.8 Implement retrieval error handling with graceful degradation
- [ ] 6.9 Create retrieval metrics collection and query analytics
- [ ] 6.10 Add retrieval configuration management and strategy selection

## 7. ChunkingCoordinator Implementation

- [x] 7.1 Extract chunking logic from `GatewayService.chunk_document` into `ChunkingCoordinator` class
- [x] 7.2 Implement document preprocessing and validation utilities
- [ ] 7.3 Create chunking profile selection and configuration logic
- [ ] 7.4 Add chunking pipeline orchestration (preprocess → split → postprocess)
- [ ] 7.5 Implement chunking result validation and quality metrics
- [x] 7.6 Create chunking error handling with fallback strategies
- [x] 7.7 Add chunking performance monitoring and optimization
- [ ] 7.8 Implement chunking caching for repeated document processing
- [ ] 7.9 Create chunking metrics collection and analytics
- [ ] 7.10 Add chunking configuration management and profile switching

## 8. ExtractionCoordinator Implementation

- [ ] 8.1 Extract extraction logic from `GatewayService.extract` into `ExtractionCoordinator` class
- [ ] 8.2 Implement entity recognition and linking pipeline
- [ ] 8.3 Create claim extraction and confidence scoring logic
- [ ] 8.4 Add knowledge graph integration and relationship mapping
- [ ] 8.5 Implement extraction validation and quality assessment
- [ ] 8.6 Create extraction error handling with model fallback strategies
- [ ] 8.7 Add extraction performance monitoring and optimization
- [ ] 8.8 Implement extraction caching for repeated entity processing
- [ ] 8.9 Create extraction metrics collection and analytics
- [ ] 8.10 Add extraction configuration management and model selection

## 9. Protocol Handler Updates

- [ ] 9.1 Update REST router to use coordinator interfaces instead of `GatewayService`
- [ ] 9.2 Update GraphQL resolvers to use coordinator interfaces
- [ ] 9.3 Update gRPC service implementations to use coordinator interfaces
- [ ] 9.4 Update SOAP handlers to use coordinator interfaces
- [ ] 9.5 Update SSE handlers to use coordinator interfaces
- [ ] 9.6 Create coordinator factory for dependency injection across protocols
- [ ] 9.7 Implement error translation layer for coordinator exceptions to protocol-specific errors
- [ ] 9.8 Add coordinator health checking and circuit breaker integration
- [ ] 9.9 Create coordinator performance monitoring and alerting
- [ ] 9.10 Add coordinator configuration hot-reloading capabilities

## 10. Testing & Validation

- [ ] 10.1 Create comprehensive unit tests for `JobLifecycleManager`
- [ ] 10.2 Create unit tests for each coordinator class with mocked dependencies
- [ ] 10.3 Create integration tests for coordinator interaction and composition
- [ ] 10.4 Create performance tests for coordinator overhead and scalability
- [ ] 10.5 Test coordinator error handling and recovery mechanisms
- [ ] 10.6 Test coordinator configuration and hot-reloading
- [ ] 10.7 Create end-to-end tests for complete workflow execution
- [ ] 10.8 Test coordinator metrics collection and monitoring
- [ ] 10.9 Create coordinator debugging and introspection tests
- [ ] 10.10 Test coordinator backward compatibility with existing protocol handlers

## 11. Migration & Deployment

- [ ] 11.1 Create migration script for existing `GatewayService` usage patterns
- [x] 11.2 Update service initialization to use coordinator composition
- [ ] 11.3 Add coordinator configuration to application settings
- [ ] 11.4 Update Docker Compose and Kubernetes deployments for new structure
- [ ] 11.5 Create coordinator health checks for deployment validation
- [ ] 11.6 Add coordinator metrics to monitoring dashboards
- [ ] 11.7 Create coordinator debugging tools and operational runbooks
- [ ] 11.8 Update API documentation for new coordinator-based interfaces
- [ ] 11.9 Create coordinator migration guide for existing integrations
- [ ] 11.10 Add coordinator performance tuning and optimization guides

## 12. Documentation & Developer Experience

- [ ] 12.1 Create comprehensive coordinator architecture documentation
- [ ] 12.2 Add coordinator interface documentation with usage examples
- [ ] 12.3 Create coordinator interaction diagrams and sequence flows
- [ ] 12.4 Update API documentation for coordinator-based endpoints
- [ ] 12.5 Create coordinator debugging and troubleshooting guides
- [ ] 12.6 Add coordinator performance tuning documentation
- [ ] 12.7 Create coordinator extension and customization guides
- [ ] 12.8 Add coordinator migration documentation for existing code
- [ ] 12.9 Create coordinator testing best practices and examples
- [ ] 12.10 Add coordinator operational monitoring and alerting guides

## 13. Legacy Code Decommissioning

### Phase 1: Remove Monolithic GatewayService (Week 1)

- [ ] 13.1 **DECOMMISSION**: Remove `src/Medical_KG_rev/gateway/services.py` monolithic GatewayService class
- [ ] 13.2 **DECOMMISSION**: Delete hardcoded service locator pattern with 12+ collaborators
- [ ] 13.3 **DECOMMISSION**: Remove legacy job lifecycle methods (`_new_job`, `_complete_job`, `_fail_job`)
- [ ] 13.4 **DECOMMISSION**: Delete unused service utility functions and helpers
- [ ] 13.5 **DECOMMISSION**: Remove legacy service configuration and validation code

### Phase 2: Clean Up Dependencies (Week 2)

- [ ] 13.6 **DECOMMISSION**: Remove unused service import statements and dependencies
- [ ] 13.7 **DECOMMISSION**: Delete legacy service error handling and fallback mechanisms
- [ ] 13.8 **DECOMMISSION**: Remove legacy service test fixtures and mocks
- [ ] 13.9 **DECOMMISSION**: Clean up unused service debugging and introspection tools
- [ ] 13.10 **DECOMMISSION**: Remove legacy service performance monitoring code

### Phase 3: Documentation and Cleanup (Week 3)

- [ ] 13.11 **DECOMMISSION**: Update documentation to remove references to monolithic GatewayService
- [ ] 13.12 **DECOMMISSION**: Remove legacy service examples and configuration templates
- [ ] 13.13 **DECOMMISSION**: Clean up unused service configuration files
- [ ] 13.14 **DECOMMISSION**: Remove legacy service API documentation
- [ ] 13.15 **DECOMMISSION**: Final cleanup of unused files and directories
