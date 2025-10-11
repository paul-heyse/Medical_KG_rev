# Modernize Observability Architecture - Change Proposal

## Quick Links

- **Proposal**: [proposal.md](./proposal.md) - Why, what changes, and impact
- **Design**: [design.md](./design.md) - Technical decisions, architecture, and rationale
- **Tasks**: [tasks.md](./tasks.md) - High-level implementation checklist
- **Detailed Tasks**: [DETAILED_TASKS.md](./DETAILED_TASKS.md) - Comprehensive AI agent implementation guide
- **Gap Assessment**: [GAP_ASSESSMENT.md](./GAP_ASSESSMENT.md) - Completeness evaluation and action items

## Overview

This change proposal addresses four critical improvements to the Medical_KG_rev codebase:

1. **Domain-Specific Metric Registries**: Separate Prometheus metrics by operational domain (GPU, HTTP, Pipeline, Cache, Reranking) to reduce label cardinality by 60%+ and improve observability clarity.

2. **Typed EmbeddingStage Contracts**: Replace dynamic request fabrication with Pydantic v2 models, returning structured results instead of mutating pipeline context for better composability and traceability.

3. **Torch-Isolated GPU Services**: Enforce documented torch isolation architecture by replacing in-process Hugging Face model loading with gRPC calls to GPU microservices.

4. **Simulation Artifact Cleanup**: Remove MinerU/VLLM simulation classes and mocked clients, replacing with real service integration tests or proper mocks.

## Current Status

✅ **Validation**: Passed `openspec validate modernize-observability-architecture --strict`
✅ **Implementation Readiness**: 92% (from 38% baseline)
✅ **Approval Status**: **READY FOR REVIEW**

## Problem Statement

The current codebase has several architectural issues:

### Issue 1: Metric Registry Pollution

**File**: `src/Medical_KG_rev/observability/metrics.py` (190 lines)

**Problem**: Shared GPU Prometheus metrics are reused for HTTP traffic, pipeline bookkeeping, cache performance, and reranking, creating:

- High-cardinality label sets (e.g., `GPU_SERVICE_CALLS_TOTAL` with labels mixing service types)
- Diluted alert semantics (GPU alerts triggered by HTTP traffic)
- Difficult debugging (cannot isolate domain-specific issues)

**Example** (line 67):

```python
GPU_SERVICE_CALLS_TOTAL.labels(service="gateway", method=event, status="success").inc()
```

This uses GPU metrics for gateway business events (NOT GPU operations).

### Issue 2: EmbeddingStage Anti-Pattern

**File**: `src/Medical_KG_rev/orchestration/ingestion_pipeline.py:EmbeddingStage.execute()` (lines 26-84)

**Problem**: Dynamically fabricates request objects and mutates pipeline context:

```python
request = type('Request', (), {
    'texts': texts,
    'namespaces': self.namespaces,
    'models': self.models
})()  # Dynamic object creation!

context.data["embeddings"] = response.vectors  # Context mutation!
```

**Consequences**:

- No type safety or validation
- Difficult to test and debug
- Poor composability with other stages

### Issue 3: Torch Isolation Violation

**File**: `src/Medical_KG_rev/services/retrieval/qwen3_service.py:Qwen3Service.__init__()` (lines 86-100)

**Problem**: Loads Hugging Face models in-process despite documented torch-isolated architecture:

```python
self.model = AutoModel.from_pretrained(model_name)  # In-process loading!
```

**Documentation Violation**: System architecture specifies "All GPU operations SHALL use gRPC to isolated GPU microservices."

### Issue 4: Simulation Artifacts Mask Integration

**Files**:

- `src/Medical_KG_rev/services/mineru/vllm_client.py` - Mock VLLMClient (86 lines)
- `src/Medical_KG_rev/services/mineru/cli_wrapper.py` - Simulation wrapper
- `tests/performance/vllm_load_test.py` - Obsolete load tests

**Problem**: Simulation code creates false confidence that integration works, masking real service issues.

## Solution Summary

### Solution 1: Domain-Specific Registries

Create 5 isolated metric registries:

- `GPUMetricRegistry` - GPU operations only
- `HTTPMetricRegistry` - API gateway and HTTP clients
- `PipelineMetricRegistry` - Orchestration pipeline state
- `CacheMetricRegistry` - Caching layer performance
- `RerankingMetricRegistry` - Search reranking operations

**Benefits**:

- 60%+ reduction in label cardinality
- Clear domain boundaries
- Improved alert precision
- Easier debugging

### Solution 2: Pydantic Contracts

Replace dynamic objects with typed contracts:

```python
class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    texts: tuple[str, ...] = Field(..., min_length=1, max_length=1000)
    namespace: str = Field(..., pattern=r"^[a-z0-9_]+$")
    model_id: str
    correlation_id: str | None = None

class EmbeddingResult(BaseModel):
    vectors: tuple[EmbeddingVector, ...]
    processing_time_ms: float
    # ... immutable, validated result
```

**Benefits**:

- Type safety with mypy validation
- Runtime validation with Pydantic
- Immutability (frozen=True)
- Better composability

### Solution 3: gRPC GPU Integration

Replace in-process model loading with gRPC client:

```python
class Qwen3GRPCClient:
    def __init__(self, endpoint: str):
        self.channel = grpc.insecure_channel(endpoint)
        self.stub = embedding_service_pb2_grpc.EmbeddingServiceStub(self.channel)
        self.circuit_breaker = CircuitBreaker(...)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # gRPC call to GPU microservice
        response = self.stub.BatchEmbed(request, timeout=self.timeout)
        return [list(emb.values) for emb in response.embeddings]
```

**Benefits**:

- Complete torch isolation
- GPU resource isolation
- Independent scaling
- Fail-fast behavior

### Solution 4: Real Integration Tests

Replace simulation with testcontainers:

```python
@pytest.fixture
def vllm_container():
    container = DockerContainer("vllm/vllm-openai:latest")
    with container:
        yield container

def test_vllm_integration(vllm_container):
    # Test against REAL service
    client = Qwen3GRPCClient(endpoint)
    embeddings = client.embed_texts(["test"])
    assert len(embeddings) == 1
```

**Benefits**:

- Real service behavior validation
- Integration gaps exposed
- Confidence in production readiness

## Implementation Approach

### Phase 1: Metric Registries (Weeks 1-2)

- Create 5 domain-specific registry classes
- Add feature flag `USE_DOMAIN_REGISTRIES`
- Migrate call sites with backward compatibility
- 100% test coverage

**Deliverables**: 6 registry classes, feature flag, 30+ tests

### Phase 2: EmbeddingStage (Weeks 3-4)

- Create Pydantic models for request/result
- Implement `EmbeddingStageV2` with typed contracts
- Add feature flag `USE_TYPED_EMBEDDING_STAGE`
- Transformation utilities for backward compat

**Deliverables**: 3 Pydantic models, new stage impl, feature flag, 20+ tests

### Phase 3: Qwen3 gRPC (Weeks 5-6)

- Implement `Qwen3GRPCClient` using existing proto
- Integrate circuit breaker
- Add feature flag `QWEN3_USE_GRPC`
- Performance benchmarks (< 50ms P95 increase)

**Deliverables**: gRPC client, circuit breaker integration, benchmarks, 15+ tests

### Phase 4: Simulation Cleanup (Weeks 7-8)

- Catalog all simulation artifacts
- Implement testcontainers for integration tests
- Delete simulation files
- Update CI/CD pipeline

**Deliverables**: Artifact catalog, testcontainer tests, CI updates, 10+ tests

### Phase 5: Finalization (Weeks 9-10)

- Enable all feature flags by default
- Deprecate old implementations (2-week notice)
- Update Grafana dashboards
- Final performance validation

**Deliverables**: Production-ready system, updated dashboards, migration complete

## Feature Flags

All changes use feature flags for safe rollout:

| Flag | Default (Migration) | Default (Final) | Environment Variable |
|------|---------------------|-----------------|---------------------|
| `use_domain_registries` | False | True | `USE_DOMAIN_REGISTRIES` |
| `use_typed_embedding_stage` | False | True | `USE_TYPED_EMBEDDING_STAGE` |
| `qwen3_use_grpc` | False | True | `QWEN3_USE_GRPC` |

## Breaking Changes

### EmbeddingStage API

**Change**: `execute()` returns `EmbeddingResult` instead of mutating context

**Migration**:

```python
# Old way
stage.execute(context)  # Mutates context.data
embeddings = context.data["embeddings"]

# New way (with feature flag)
result = stage.execute(context)  # Returns typed result
embeddings = result.vectors
```

**Backward Compatibility**: Transformation utilities provided to convert between formats

### Qwen3 Service Internal Implementation

**Change**: Uses gRPC client instead of in-process model

**Migration**: No API changes for callers, only internal implementation

### Metrics Collection

**Change**: New metric registry API

**Migration**:

```python
# Old way
GPU_SERVICE_CALLS_TOTAL.labels(service="gpu", method="embed", status="success").inc()

# New way (with feature flag)
get_gpu_registry().record_service_call(service="gpu", method="embed", status="success")
```

**Backward Compatibility**: Old metrics still exported during migration

## Testing Strategy

### Unit Tests

- **Target**: 100% coverage for new code, 90%+ for modified code
- **Framework**: pytest with pytest-cov
- **Validation**: `mypy --strict` on all new code

### Integration Tests

- **Metric Registries**: Test with both feature flags enabled/disabled
- **EmbeddingStage**: Full pipeline tests with typed contracts
- **Qwen3 gRPC**: Mock gRPC stub for unit tests, testcontainers for integration

### Performance Tests

- **Baseline**: Measure before Phase 1
- **Acceptance**: < 5% performance regression
- **P95 Latency**: < 50ms increase for embedding operations

### Contract Tests

- **gRPC**: Buf breaking change detection on proto files
- **REST**: Schemathesis on OpenAPI specs
- **GraphQL**: GraphQL Inspector on schema

## Rollback Procedures

### Per-Phase Rollback

```bash
# Phase 1: Disable domain registries
export USE_DOMAIN_REGISTRIES=false
# Restart services

# Phase 2: Disable typed embedding
export USE_TYPED_EMBEDDING_STAGE=false
# Restart services

# Phase 3: Disable gRPC
export QWEN3_USE_GRPC=false
# Restart services (falls back to in-process)
```

### Emergency Rollback

1. Git revert to previous commit
2. Redeploy previous version
3. Restore Grafana dashboards from backup
4. Conduct incident post-mortem

## Monitoring During Rollout

### Key Metrics

- `gpu_service_calls_total` - Domain separation
- `embedding_stage_duration_seconds` - Performance impact
- `qwen3_grpc_errors_total` - Service reliability
- `circuit_breaker_state` - Resilience

### Alert Thresholds

- P95 latency > 500ms → Page
- Error rate > 5% → Page
- Circuit breaker open > 5 min → Warning
- Memory usage increase > 20% → Warning

## Documentation Impact

### Files to Update

- `docs/devops/observability.md` - New metric registry structure
- `docs/operational-runbook.md` - Feature flag rollback procedures
- `docs/architecture/overview.md` - Torch isolation architecture
- `docs/guides/developer_guide.md` - Typed contract usage

### New Documentation

- Migration guide for metric collection
- EmbeddingStage API migration guide
- gRPC GPU service integration guide
- Testcontainers setup guide

## Success Criteria

### Metrics

- ✅ Label cardinality reduced by 60%+
- ✅ Metric collection overhead < 5%
- ✅ Dashboard compatibility maintained

### Code Quality

- ✅ 100% type hint coverage in new code
- ✅ 90%+ test coverage
- ✅ Zero `mypy --strict` errors
- ✅ All docstrings follow Google style

### Performance

- ✅ P95 latency increase < 50ms
- ✅ No memory leaks detected
- ✅ Circuit breakers functional

### Architecture

- ✅ Zero in-process torch dependencies in main gateway
- ✅ All simulation artifacts removed
- ✅ gRPC integration complete

## Risks and Mitigation

### Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| Metric migration breaks dashboards | Medium | High | Backward compatible exports + dashboard backup |
| gRPC latency impacts performance | High | Medium | Circuit breaker + performance benchmarks |
| Test coverage gaps after cleanup | High | Medium | 100% coverage requirement + testcontainers |

### Operational Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| Rollout coordination issues | Medium | Low | Feature flags + phased deployment |
| Monitoring blind spots | Medium | Low | Explicit metrics list + alert validation |
| Rollback complexity | High | Low | Per-phase rollback procedures |

## Action Items Before Implementation

### High Priority (Blocking)

1. **Proto Contract Validation** (1 hour)
   - Read `src/Medical_KG_rev/proto/embedding_service.proto`
   - Validate against `Qwen3GRPCClient` implementation in DETAILED_TASKS.md
   - Adjust if proto differs

2. **Dashboard Inventory** (2-4 hours)
   - List all dashboards in `ops/monitoring/`
   - Identify dashboards querying GPU metrics
   - Create dashboard migration checklist

3. **CI Docker Support** (1 hour)
   - Test testcontainers in CI pipeline
   - Document infrastructure requirements
   - Plan CI configuration changes

### Medium Priority (Recommended)

4. Benchmark sync vs async gRPC (2 hours)
5. Audit operational runbooks (1 hour)
6. Plan communication to team (30 minutes)

## Timeline

| Phase | Duration | Start After | Completion Criteria |
|-------|----------|-------------|-------------------|
| Phase 1 | 2 weeks | Action items 1-2 | 100% registry coverage |
| Phase 2 | 2 weeks | Phase 1 | Typed contracts deployed |
| Phase 3 | 2 weeks | Phase 2 + action item 1 | gRPC integration live |
| Phase 4 | 2 weeks | Phase 3 + action item 3 | Simulations removed |
| Phase 5 | 2 weeks | All phases | Flags enabled by default |
| **Total** | **10 weeks** | | **Production-ready** |

## Approval Checklist

- ✅ Proposal validated with `openspec validate --strict`
- ✅ Gap assessment shows 92% implementation readiness
- ✅ All tasks have acceptance criteria
- ✅ Feature flags enable safe rollout
- ✅ Rollback procedures documented
- ✅ Testing strategy comprehensive
- ⚠️ Action items 1-3 required before Phase 1 start

## Questions or Concerns?

- **Technical Questions**: Review [design.md](./design.md) for detailed decisions
- **Implementation Questions**: See [DETAILED_TASKS.md](./DETAILED_TASKS.md) for step-by-step guide
- **Risk Concerns**: See [GAP_ASSESSMENT.md](./GAP_ASSESSMENT.md) for risk analysis

---

**Status**: ✅ **READY FOR REVIEW AND APPROVAL**
**Next Step**: Complete action items 1-3, then begin Phase 1 implementation
