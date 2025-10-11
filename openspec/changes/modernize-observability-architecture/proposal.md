## Why

The current codebase has several architectural issues that compromise observability effectiveness, torch isolation integrity, and code maintainability:

1. **Shared GPU metrics are diluted**: Prometheus metrics for GPU services are being reused for HTTP traffic, pipeline bookkeeping, cache performance, and reranking, creating high-cardinality label sets that reduce alert semantics and monitoring clarity.

2. **EmbeddingStage violates separation of concerns**: The ingestion pipeline's EmbeddingStage dynamically fabricates request objects and mutates pipeline context instead of using typed contracts and returning structured results, reducing composability and traceability.

3. **Torch isolation is incomplete**: The Qwen3 service still loads Hugging Face models in-process instead of using gRPC calls to GPU microservices, violating the documented torch-isolated architecture.

4. **Legacy simulation artifacts persist**: MinerU/VLLM simulation classes and mocked clients remain in the codebase, masking integration gaps and creating confusion about actual service availability.

These issues compromise the system's observability goals, violate the torch isolation architecture, and introduce technical debt that impedes maintainability.

## What Changes

### 1. Domain-Specific Metric Registries

Carve out dedicated metric registries for each domain to improve observability clarity:

- **GPU Metrics**: Exclusive to GPU hardware metrics (memory, utilization, temperature, device status)
- **gRPC Metrics**: Internal service-to-service communication via gRPC (all Docker container communication)
- **External API Metrics**: External-facing HTTP APIs (REST/GraphQL/SOAP) and adapter HTTP clients to external services
- **Pipeline Metrics**: Focused on orchestration pipeline state and execution
- **Cache Metrics**: Isolated to caching layer performance and hit rates
- **Reranking Metrics**: Specific to search result reranking operations

Each registry will use appropriate label sets without cross-domain pollution.

**Architectural Note**: All internal service communication uses gRPC (not HTTP). HTTP is ONLY used for external client-facing APIs and connections to external databases/services.

### 2. Typed EmbeddingStage Contracts

Replace the current EmbeddingStage implementation:

- **Replace dynamic request fabrication** with strongly-typed `EmbeddingRequest` and `EmbeddingResult` contracts
- **Return structured results** instead of mutating pipeline context for better composability
- **Add comprehensive validation** with Pydantic models and proper error handling
- **Improve traceability** with structured logging and correlation IDs

### 3. Enforce Torch Isolation Architecture

Update Qwen3 service to use gRPC microservice calls:

- **Remove in-process model loading** and replace with gRPC client to GPU services
- **Implement proper service discovery** for GPU microservice endpoints
- **Add circuit breaker integration** for GPU service resilience
- **Maintain embedding API compatibility** while changing internal implementation

### 4. Remove Legacy Simulation Artifacts

Clean up MinerU/VLLM simulation code:

- **Remove simulation classes** (`VLLMClient`, `MinerUSimulator`, etc.)
- **Remove mocked test clients** that mask integration gaps
- **Remove obsolete load tests** that don't reflect real service behavior
- **Update integration tests** to use real service interfaces or proper mocks

## Impact

### Affected Capabilities

- **observability**: Domain-specific metric registries (6 total: GPU, gRPC, External API, Pipeline, Cache, Reranking) and improved monitoring clarity
- **orchestration**: Typed EmbeddingStage contracts and structured pipeline results
- **services**: Torch-isolated GPU service integration and complete gRPC-based internal communication
- **retrieval**: Updated embedding service dependencies and performance monitoring

### Breaking Changes

- **EmbeddingStage API**: The `execute()` method signature and return type will change
- **Qwen3 Service**: Internal implementation changes from in-process to gRPC-based
- **Metrics Collection**: Existing metric labels may need updates for new registry structure

### Migration Path

1. **Phase 1**: Create new metric registries alongside existing ones (backward compatible)
2. **Phase 2**: Migrate EmbeddingStage to typed contracts with feature flags
3. **Phase 3**: Update Qwen3 service to use gRPC calls to GPU services
4. **Phase 4**: Remove legacy simulation artifacts and update tests
5. **Phase 5**: Remove old metric registries and finalize new implementations

### Benefits

- **Improved Observability**: Clear domain separation in metrics reduces alert fatigue and improves debugging
- **Better Architecture**: Typed contracts improve code reliability and maintainability
- **Torch Isolation Compliance**: Complete removal of torch dependencies from main gateway
- **Cleaner Codebase**: Removal of simulation artifacts reduces confusion and technical debt

### Risks

- **Metric Migration**: Existing dashboards and alerts may need updates
- **Performance Impact**: gRPC calls may introduce latency compared to in-process operations
- **Test Coverage**: Removal of simulation code requires comprehensive integration testing
