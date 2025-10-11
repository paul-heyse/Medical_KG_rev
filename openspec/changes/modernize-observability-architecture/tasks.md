## 1. Domain-Specific Metric Registries

### 1.1 Create Metric Registry Infrastructure

- [ ] Create `BaseMetricRegistry` abstract class with common functionality
- [ ] Implement `GPUMetricRegistry` for GPU service operations only
- [ ] Implement `HTTPMetricRegistry` for API gateway and HTTP client operations
- [ ] Implement `PipelineMetricRegistry` for orchestration pipeline state and execution
- [ ] Implement `CacheMetricRegistry` for caching layer performance and hit rates
- [ ] Implement `RerankingMetricRegistry` for search result reranking operations

### 1.2 Migrate Existing Metrics

- [ ] Audit current metrics usage across codebase
- [ ] Move GPU-specific metrics to `GPUMetricRegistry`
- [ ] Move HTTP request metrics to `HTTPMetricRegistry`
- [ ] Move pipeline state metrics to `PipelineMetricRegistry`
- [ ] Move cache performance metrics to `CacheMetricRegistry`
- [ ] Move reranking metrics to `RerankingMetricRegistry`

### 1.3 Update Metrics Collection Points

- [ ] Update GPU service metrics collection to use new registry
- [ ] Update HTTP client metrics collection to use new registry
- [ ] Update pipeline stage metrics collection to use new registry
- [ ] Update cache operation metrics collection to use new registry
- [ ] Update reranking operation metrics collection to use new registry

### 1.4 Validation and Testing

- [ ] Create unit tests for each metric registry
- [ ] Validate metric label cardinality is reduced
- [ ] Ensure backward compatibility with existing monitoring
- [ ] Update Prometheus configuration for new metric structure

## 2. Typed EmbeddingStage Contracts

### 2.1 Design Typed Contracts

- [ ] Create `EmbeddingRequest` Pydantic model with proper validation
- [ ] Create `EmbeddingResult` Pydantic model for structured responses
- [ ] Create `EmbeddingError` exception classes for proper error handling
- [ ] Design embedding options and configuration models

### 2.2 Refactor EmbeddingStage Implementation

- [ ] Replace dynamic request object fabrication with `EmbeddingRequest` construction
- [ ] Replace context mutation with `EmbeddingResult` return value
- [ ] Add comprehensive input validation using Pydantic models
- [ ] Implement proper error translation and structured logging

### 2.3 Update Pipeline Integration

- [ ] Update pipeline context to handle `EmbeddingResult` objects
- [ ] Add result transformation utilities for backward compatibility
- [ ] Update downstream stages to consume structured embedding results
- [ ] Add feature flag for gradual migration

### 2.4 Testing and Validation

- [ ] Create unit tests for new `EmbeddingStage` implementation
- [ ] Create integration tests for pipeline embedding flow
- [ ] Validate error handling and edge cases
- [ ] Performance test new implementation

## 3. Enforce Torch Isolation Architecture

### 3.1 Create gRPC GPU Service Client

- [ ] Design `GPUEmbeddingClient` interface for gRPC communication
- [ ] Implement `Qwen3GRPCClient` for GPU microservice communication
- [ ] Add service discovery for GPU microservice endpoints
- [ ] Implement circuit breaker integration for GPU service resilience

### 3.2 Refactor Qwen3 Service Implementation

- [ ] Replace in-process model loading with gRPC client initialization
- [ ] Update `embed_texts` method to use gRPC calls instead of local inference
- [ ] Maintain API compatibility while changing internal implementation
- [ ] Add proper error handling for gRPC service failures

### 3.3 Update Service Registration

- [ ] Update embedding service registry to use gRPC-based Qwen3 service
- [ ] Add configuration for GPU microservice endpoints
- [ ] Update health checks to validate gRPC connectivity
- [ ] Add monitoring for gRPC call performance

### 3.4 Testing and Validation

- [ ] Create unit tests for gRPC client implementation
- [ ] Create integration tests for GPU microservice communication
- [ ] Validate embedding quality is maintained with gRPC approach
- [ ] Performance test gRPC vs in-process implementation

## 4. Remove Legacy Simulation Artifacts

### 4.1 Identify Simulation Artifacts

- [ ] Catalog all MinerU/VLLM simulation classes and mock implementations
- [ ] Identify test files using simulation artifacts
- [ ] Find load tests and performance tests using mocks
- [ ] Document integration gaps that simulations were masking

### 4.2 Remove Simulation Classes

- [ ] Delete `VLLMClient` simulation class and related code
- [ ] Remove `MinerUSimulator` and mock implementations
- [ ] Delete simulated health check and status endpoints
- [ ] Remove mock circuit breaker implementations

### 4.3 Update Test Infrastructure

- [ ] Replace simulation-based tests with proper integration tests
- [ ] Update test fixtures to use real service interfaces
- [ ] Add proper mocking for external dependencies
- [ ] Create contract tests for service interfaces

### 4.4 Clean Up Load Tests

- [ ] Remove obsolete load tests that don't reflect real service behavior
- [ ] Update performance tests to use actual service endpoints
- [ ] Add realistic performance benchmarks for GPU services
- [ ] Update CI/CD pipeline to reflect real service testing

## 5. Integration and Validation

### 5.1 Cross-Component Integration

- [ ] Validate metric registries work correctly across all domains
- [ ] Test EmbeddingStage integration with pipeline orchestration
- [ ] Verify Qwen3 service gRPC integration with GPU microservices
- [ ] Ensure all simulation artifacts have been removed

### 5.2 Performance Validation

- [ ] Benchmark new metric collection performance
- [ ] Measure EmbeddingStage performance with typed contracts
- [ ] Validate Qwen3 service performance with gRPC calls
- [ ] Ensure overall system performance meets SLO requirements

### 5.3 Monitoring and Alerting

- [ ] Update Grafana dashboards for new metric structure
- [ ] Update Prometheus alerting rules for domain-specific metrics
- [ ] Add monitoring for gRPC service health and performance
- [ ] Validate alert semantics are improved with new registries

## 6. Documentation and Migration

### 6.1 Update Documentation

- [ ] Document new metric registry structure and usage
- [ ] Update EmbeddingStage API documentation
- [ ] Document gRPC-based Qwen3 service implementation
- [ ] Update architecture documentation for torch isolation

### 6.2 Migration Guides

- [ ] Create migration guide for metric collection changes
- [ ] Document EmbeddingStage API changes for downstream consumers
- [ ] Provide guidance for updating GPU service integration
- [ ] Document removal of simulation artifacts

### 6.3 Training and Rollout

- [ ] Create training materials for new observability approach
- [ ] Plan phased rollout with feature flags
- [ ] Create rollback procedures for each component
- [ ] Update operational runbooks for new architecture
