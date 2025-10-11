## Context

The Medical_KG_rev codebase has evolved to include torch isolation architecture, multi-protocol API gateway, and GPU-based microservices. However, several implementation gaps have emerged that compromise observability effectiveness, architectural consistency, and code maintainability:

1. **Metric Registry Pollution**: The current `src/Medical_KG_rev/observability/metrics.py` file contains ~190 lines mixing GPU, HTTP, pipeline, cache, and reranking metrics into shared Prometheus collectors. Functions like `observe_job_duration()` and `record_business_event()` abuse GPU service metrics for non-GPU operations (lines 59-67).

2. **EmbeddingStage Anti-Pattern**: `src/Medical_KG_rev/orchestration/ingestion_pipeline.py:EmbeddingStage.execute()` (lines 26-84) dynamically fabricates request objects using `type('Request', (), {...})()` (line 45) and mutates `context.data` dictionaries instead of using typed contracts.

3. **Torch Isolation Violation**: `src/Medical_KG_rev/services/retrieval/qwen3_service.py:Qwen3Service.__init__()` loads Hugging Face models in-process (lines 86-100) despite documented torch-isolated architecture requiring gRPC calls to GPU microservices.

4. **Simulation Artifacts**: Legacy MinerU/VLLM simulation code remains in production codebase:
   - `src/Medical_KG_rev/services/mineru/vllm_client.py` - Mock VLLMClient (86 lines)
   - `src/Medical_KG_rev/services/mineru/cli_wrapper.py` - Simulation wrapper
   - `tests/performance/vllm_load_test.py` - Obsolete load tests
   - `tests/integration/test_vllm_integration.py` - Mock-based integration tests

### Architectural Constraints

- **gRPC Proto Contracts**: Existing `src/Medical_KG_rev/proto/embedding_service.proto` and `src/Medical_KG_rev/proto/gpu_service.proto` define gRPC interfaces
- **Pipeline State Model**: `src/Medical_KG_rev/orchestration/stages/contracts.py:PipelineState` (lines 890-2209) uses dataclass with typed fields
- **Prometheus Compatibility**: Must maintain Prometheus exposition format and existing dashboard compatibility during migration
- **Zero Downtime**: Changes must support phased rollout with feature flags

## Goals / Non-Goals

### Goals

1. **Domain-Separated Observability**: Reduce metric label cardinality by 60%+ through domain-specific registries
2. **Type-Safe Pipeline Contracts**: Eliminate dynamic request fabrication in favor of Pydantic models
3. **Complete Torch Isolation**: Remove all in-process torch/transformers usage from main gateway
4. **Clean Test Infrastructure**: Replace simulation artifacts with real service contracts or proper mocks

### Non-Goals

- Changing Prometheus metric exposition format (backward compatibility required)
- Modifying existing gRPC proto definitions (use existing contracts)
- Rewriting pipeline orchestration engine (minimize scope)
- Adding new GPU microservices (focus on integration, not new services)

## Decisions

### Decision 1: Metric Registry Architecture

**Choice**: Implement domain-specific metric registry classes inheriting from `BaseMetricRegistry`

**Alternatives Considered**:

- **Option A**: Namespace-prefixed metrics in single registry → Rejected: doesn't solve label cardinality
- **Option B**: Separate Prometheus registries per domain → Rejected: breaks Prometheus scraping expectations
- **Option C**: Domain-specific classes with shared Prometheus collectors → **Selected**

**Rationale**: Python classes can encapsulate domain logic while still exporting to single Prometheus registry. Provides type safety and clear boundaries without breaking existing monitoring infrastructure.

**Implementation Details**:

```python
# src/Medical_KG_rev/observability/registries/base.py
class BaseMetricRegistry(ABC):
    def __init__(self, domain: str):
        self._domain = domain
        self._collectors = {}

    @abstractmethod
    def register_collector(self, name: str, collector: Collector) -> None:
        ...

# src/Medical_KG_rev/observability/registries/gpu.py
class GPUMetricRegistry(BaseMetricRegistry):
    def __init__(self):
        super().__init__(domain="gpu")
        self.service_calls = Counter(
            "gpu_service_calls_total",
            "GPU service operations",
            ["service", "method", "status"]
        )
        # Only GPU-relevant labels
```

**Migration Strategy**:

1. Create new registries alongside existing metrics (weeks 1-2)
2. Update call sites with feature flag `USE_DOMAIN_REGISTRIES` (weeks 3-4)
3. Deprecate old metric functions (week 5)
4. Remove deprecated functions after 2-week deprecation notice (week 6)

### Decision 2: EmbeddingStage Contract Design

**Choice**: Pydantic v2 models with explicit validation and immutable results

**Alternatives Considered**:

- **Option A**: TypedDict with runtime checks → Rejected: no validation, mutable
- **Option B**: dataclasses with manual validation → Rejected: verbose, error-prone
- **Option C**: Pydantic models with strict mode → **Selected**

**Rationale**: Pydantic v2 provides:

- Automatic validation with clear error messages
- JSON schema generation for documentation
- Immutability with `frozen=True`
- Performance comparable to dataclasses

**Implementation Details**:

```python
# src/Medical_KG_rev/orchestration/stages/embedding/contracts.py
from pydantic import BaseModel, Field, ConfigDict, field_validator

class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    texts: tuple[str, ...] = Field(..., min_length=1, max_length=1000)
    namespace: str = Field(..., pattern=r"^[a-z0-9_]+$")
    model_id: str = Field(..., min_length=1)
    correlation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("texts")
    @classmethod
    def validate_text_length(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for text in v:
            if len(text) > 10000:
                raise ValueError(f"Text exceeds 10000 chars: {len(text)}")
        return v

class EmbeddingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    vectors: tuple[EmbeddingVector, ...]
    model_id: str
    namespace: str
    processing_time_ms: float
    gpu_memory_used_mb: int | None = None
    correlation_id: str | None = None
```

**Backward Compatibility**: Provide adapter functions for existing pipeline context mutations:

```python
def result_to_context_data(result: EmbeddingResult) -> dict[str, Any]:
    """Convert EmbeddingResult to legacy context.data format"""
    return {
        "embeddings": [v.model_dump() for v in result.vectors],
        "metrics": {"embedding": {"vectors": len(result.vectors)}},
        "embedding_summary": {...}
    }
```

### Decision 3: Qwen3 Service gRPC Integration

**Choice**: Create `Qwen3GRPCClient` implementing existing `src/Medical_KG_rev/proto/embedding_service.proto`

**Alternatives Considered**:

- **Option A**: Modify Qwen3Service to be GPU microservice → Rejected: out of scope
- **Option B**: Create new embedding proto → Rejected: existing proto sufficient
- **Option C**: Use existing proto with client wrapper → **Selected**

**Rationale**: The existing `embedding_service.proto` defines:

```protobuf
service EmbeddingService {
  rpc Embed(EmbedRequest) returns (EmbedResponse);
  rpc BatchEmbed(BatchEmbedRequest) returns (BatchEmbedResponse);
  rpc GetHealth(HealthRequest) returns (HealthResponse);
}
```

This supports our requirements. We only need client-side implementation.

**Implementation Details**:

```python
# src/Medical_KG_rev/services/clients/qwen3_grpc_client.py
import grpc
from Medical_KG_rev.proto import embedding_service_pb2, embedding_service_pb2_grpc

class Qwen3GRPCClient:
    def __init__(self, endpoint: str, timeout: float = 30.0):
        self.channel = grpc.insecure_channel(endpoint)
        self.stub = embedding_service_pb2_grpc.EmbeddingServiceStub(self.channel)
        self.timeout = timeout

    def embed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
        request = embedding_service_pb2.BatchEmbedRequest(
            texts=texts,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            **kwargs
        )
        try:
            response = self.stub.BatchEmbed(request, timeout=self.timeout)
            return [list(emb.values) for emb in response.embeddings]
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise GPUServiceUnavailableError(f"GPU service unavailable: {e.details()}")
            raise
```

**Service Discovery**: Use environment variables for endpoints:

```python
QWEN3_GRPC_ENDPOINT = os.getenv("QWEN3_GRPC_ENDPOINT", "localhost:50052")
```

**Circuit Breaker Integration**:

```python
from Medical_KG_rev.services.mineru.circuit_breaker import CircuitBreaker

class Qwen3GRPCClient:
    def __init__(self, ...):
        ...
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            name="qwen3_grpc"
        )
```

### Decision 4: Simulation Artifact Cleanup Strategy

**Choice**: Delete simulation code, update tests to use testcontainers or proper mocks

**Alternatives Considered**:

- **Option A**: Move simulations to separate package → Rejected: still masks integration
- **Option B**: Keep simulations for unit tests → Rejected: unit tests should mock at boundaries
- **Option C**: Delete and replace with real integration tests → **Selected**

**Rationale**: Simulation code creates false confidence. Real integration tests or proper mocking better reflects production behavior.

**Test Replacement Strategy**:

| Current Test | Replacement Approach |
|-------------|---------------------|
| `tests/integration/test_vllm_integration.py` | Use testcontainers with real vLLM Docker image |
| `tests/performance/vllm_load_test.py` | Rewrite to target real GPU service endpoint in staging |
| `tests/services/test_gpu_microservices.py` | Mock at gRPC stub level, not service level |

**Files to Delete**:

```
src/Medical_KG_rev/services/mineru/vllm_client.py
src/Medical_KG_rev/services/mineru/cli_wrapper.py
src/Medical_KG_rev/services/mineru/output_parser.py (if unused)
tests/performance/vllm_load_test.py
```

**Files to Update**:

```
tests/integration/test_vllm_integration.py → Use testcontainers
tests/integration/conftest.py → Add GPU service fixtures
tests/services/test_gpu_microservices.py → Mock grpc.Channel, not VLLMClient
```

## Risks / Trade-offs

### Risk 1: Metric Migration Breaks Dashboards

**Impact**: Medium | **Likelihood**: High

**Mitigation**:

1. Maintain backward-compatible metric names during migration
2. Export both old and new metrics with feature flag
3. Update Grafana dashboards before deprecating old metrics
4. Provide dashboard migration guide

**Rollback**: Feature flag `USE_DOMAIN_REGISTRIES=false` reverts to old metrics

### Risk 2: gRPC Latency Degrades Performance

**Impact**: High | **Likelihood**: Medium

**Mitigation**:

1. Implement connection pooling for gRPC channels
2. Use batch processing where possible
3. Add circuit breaker to fail fast on service issues
4. Performance benchmark before/after in staging

**Acceptance Criteria**: P95 latency increase < 50ms

**Rollback**: Keep in-process implementation as fallback with feature flag `QWEN3_USE_GRPC=false`

### Risk 3: Test Coverage Gaps After Simulation Removal

**Impact**: High | **Likelihood**: Medium

**Mitigation**:

1. Audit test coverage before deletion (pytest-cov > 80%)
2. Add contract tests for gRPC interfaces
3. Implement testcontainers for integration tests
4. Run full regression suite in CI

**Acceptance Criteria**: No reduction in test coverage percentage

## Migration Plan

### Phase 1: Metric Registry Infrastructure (Week 1-2)

1. Create `src/Medical_KG_rev/observability/registries/` directory
2. Implement `BaseMetricRegistry`, `GPUMetricRegistry`, `HTTPMetricRegistry`, `PipelineMetricRegistry`, `CacheMetricRegistry`, `RerankingMetricRegistry`
3. Add feature flag `USE_DOMAIN_REGISTRIES` to `src/Medical_KG_rev/config/settings.py`
4. Unit tests for each registry (100% coverage target)
5. Documentation in `docs/devops/observability.md`

**Deliverables**: 6 registry classes, feature flag, tests, docs

### Phase 2: EmbeddingStage Refactoring (Week 3-4)

1. Create `src/Medical_KG_rev/orchestration/stages/embedding/contracts.py` with Pydantic models
2. Implement `EmbeddingStageV2` with typed contracts
3. Add feature flag `USE_TYPED_EMBEDDING_STAGE`
4. Update `PipelineState` to handle both old and new result formats
5. Integration tests for embedding pipeline
6. Performance benchmarks (no regression)

**Deliverables**: Pydantic contracts, v2 implementation, feature flag, tests

### Phase 3: Qwen3 gRPC Migration (Week 5-6)

1. Create `src/Medical_KG_rev/services/clients/qwen3_grpc_client.py`
2. Implement circuit breaker integration
3. Add feature flag `QWEN3_USE_GRPC`
4. Update `Qwen3Service` to use gRPC client when flag enabled
5. Integration tests with mock gRPC server
6. Performance benchmarks in staging environment

**Deliverables**: gRPC client, circuit breaker, feature flag, tests, performance report

### Phase 4: Simulation Cleanup (Week 7-8)

1. Audit files using simulation artifacts
2. Implement testcontainers for vLLM integration tests
3. Rewrite performance tests for real endpoints
4. Delete simulation files
5. Update CI/CD pipeline
6. Verify test coverage maintained

**Deliverables**: Updated tests, deleted simulation files, CI/CD updates

### Phase 5: Finalization (Week 9-10)

1. Enable all feature flags by default
2. Deprecate old metric functions with warnings
3. Update Grafana dashboards
4. Update operational runbooks
5. Conduct performance validation
6. Remove deprecated code after 2-week notice

**Deliverables**: Production-ready implementation, updated dashboards, documentation

## Open Questions

1. **Q**: Should we implement connection pooling for gRPC channels?
   **A**: Yes, implement pooling with max 5 connections per service endpoint

2. **Q**: How do we handle GPU service discovery in Kubernetes?
   **A**: Use Kubernetes service DNS (`qwen3-grpc-service.default.svc.cluster.local:50052`)

3. **Q**: What's the rollback procedure if metrics migration breaks alerts?
   **A**: Feature flag rollback + emergency Grafana dashboard restore from backup

4. **Q**: Should we support both in-process and gRPC embedding simultaneously?
   **A**: Yes, during migration phase only (Phases 3-5), then remove in-process

5. **Q**: How do we test gRPC services in CI without GPU?
   **A**: Mock gRPC stubs for unit tests, testcontainers with CPU-only images for integration tests
