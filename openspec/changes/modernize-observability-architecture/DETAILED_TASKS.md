# Detailed Implementation Tasks for AI Agents

This document provides comprehensive, unambiguous implementation tasks with exact file paths, code locations, and acceptance criteria for AI programming agents.

## Task Execution Guidelines

- Each task includes: **File Path** | **Action** | **Acceptance Criteria** | **Dependencies**
- Read referenced files completely before making changes
- Run tests after each subsection
- Update feature flags as specified
- Validate with `pytest` and `mypy` after each phase

---

## Phase 1: Domain-Specific Metric Registries

### 1.1.1 Create BaseMetricRegistry Abstract Class

**File**: `src/Medical_KG_rev/observability/registries/base.py` (NEW)

**Action**: Create abstract base class for domain-specific metric registries

**Implementation**:

```python
from abc import ABC, abstractmethod
from typing import ClassVar
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry

class BaseMetricRegistry(ABC):
    """Abstract base class for domain-specific Prometheus metric registries."""

    _domain: str
    _collectors: dict[str, Counter | Gauge | Histogram]

    def __init__(self, domain: str, registry: CollectorRegistry | None = None):
        self._domain = domain
        self._collectors = {}
        self._registry = registry  # Use default registry if None

    @abstractmethod
    def initialize_collectors(self) -> None:
        """Initialize domain-specific Prometheus collectors."""
        ...

    def get_collector(self, name: str) -> Counter | Gauge | Histogram:
        """Retrieve a registered collector by name."""
        if name not in self._collectors:
            raise KeyError(f"Collector '{name}' not found in {self._domain} registry")
        return self._collectors[name]

    @property
    def domain(self) -> str:
        return self._domain
```

**Acceptance Criteria**:

- [x] File exists at specified path
- [x] `BaseMetricRegistry` is abstract with `ABC` inheritance
- [x] `initialize_collectors()` is abstract method
- [x] `get_collector()` returns typed collectors
- [x] Type hints use `from __future__ import annotations`
- [x] Docstrings follow Google style
- [x] Passes `mypy --strict`

**Dependencies**: None

---

### 1.1.2 Implement GPUMetricRegistry

**File**: `src/Medical_KG_rev/observability/registries/gpu.py` (NEW)

**Action**: Create GPU-specific metric registry with only GPU-relevant labels

**Implementation**:

```python
from prometheus_client import Counter, Gauge, Histogram
from Medical_KG_rev.observability.registries.base import BaseMetricRegistry

class GPUMetricRegistry(BaseMetricRegistry):
    """Metric registry for GPU service operations."""

    def __init__(self, registry=None):
        super().__init__(domain="gpu", registry=registry)
        self.initialize_collectors()

    def initialize_collectors(self) -> None:
        self._collectors["service_calls"] = Counter(
            "gpu_service_calls_total",
            "Total GPU service calls",
            ["service", "method", "status"],
            registry=self._registry
        )

        self._collectors["call_duration"] = Histogram(
            "gpu_service_call_duration_seconds",
            "GPU service call duration",
            ["service", "method"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self._registry
        )

        self._collectors["errors"] = Counter(
            "gpu_service_errors_total",
            "GPU service errors",
            ["service", "method", "error_type"],
            registry=self._registry
        )

        self._collectors["memory_usage"] = Gauge(
            "gpu_memory_usage_mb",
            "GPU memory usage in MB",
            ["device_id", "device_name"],
            registry=self._registry
        )

        self._collectors["utilization"] = Gauge(
            "gpu_utilization_percentage",
            "GPU utilization percentage",
            ["device_id", "device_name"],
            registry=self._registry
        )

        self._collectors["health_status"] = Gauge(
            "gpu_service_health_status",
            "GPU service health (1=healthy, 0=unhealthy)",
            ["service_name"],
            registry=self._registry
        )

    def record_service_call(self, service: str, method: str, status: str) -> None:
        """Record a GPU service call."""
        self.get_collector("service_calls").labels(
            service=service, method=method, status=status
        ).inc()

    def observe_call_duration(self, service: str, method: str, duration: float) -> None:
        """Observe GPU service call duration in seconds."""
        self.get_collector("call_duration").labels(
            service=service, method=method
        ).observe(duration)

    def record_error(self, service: str, method: str, error_type: str) -> None:
        """Record GPU service error."""
        self.get_collector("errors").labels(
            service=service, method=method, error_type=error_type
        ).inc()

    def set_memory_usage(self, device_id: str, device_name: str, usage_mb: float) -> None:
        """Set GPU memory usage."""
        self.get_collector("memory_usage").labels(
            device_id=device_id, device_name=device_name
        ).set(usage_mb)

    def set_utilization(self, device_id: str, device_name: str, percent: float) -> None:
        """Set GPU utilization percentage."""
        self.get_collector("utilization").labels(
            device_id=device_id, device_name=device_name
        ).set(percent)

    def set_health_status(self, service_name: str, healthy: bool) -> None:
        """Set GPU service health status."""
        self.get_collector("health_status").labels(
            service_name=service_name
        ).set(1 if healthy else 0)
```

**Acceptance Criteria**:

- [x] File exists at specified path
- [x] Inherits from `BaseMetricRegistry`
- [x] Only GPU-related labels in collectors
- [x] NO HTTP, pipeline, or cache labels mixed in
- [x] Helper methods for common operations
- [x] Type hints on all methods
- [x] Passes `mypy --strict`
- [x] Unit tests in `tests/observability/test_registries_gpu.py`

**Dependencies**: Task 1.1.1

---

### 1.1.3 Implement ExternalAPIMetricRegistry (renamed from HTTPMetricRegistry)

**File**: `src/Medical_KG_rev/observability/registries/external_api.py` (NEW)

**Action**: Create External API-specific metric registry for external-facing HTTP traffic

**Reference**: See current HTTP metrics in `src/Medical_KG_rev/observability/metrics.py:REQUEST_COUNTER` (line 47-51)

**Implementation**: Similar to 1.1.2 but with External API-specific collectors:

- `external_api_requests_total` with labels: `["method", "endpoint", "status_code"]`
- `external_api_request_duration_seconds` with labels: `["method", "endpoint", "status_code"]`
- `external_api_in_flight_requests` with labels: `["method", "endpoint"]`
- `external_api_errors_total` with labels: `["method", "endpoint", "error_type"]`

**Acceptance Criteria**: Same as 1.1.2 but for External API domain

**Dependencies**: Task 1.1.1

---

### 1.1.4 Implement PipelineMetricRegistry

**File**: `src/Medical_KG_rev/observability/registries/pipeline.py` (NEW)

**Action**: Create pipeline orchestration metric registry

**Reference**: Current pipeline metrics in `src/Medical_KG_rev/observability/metrics.py`:

- `PIPELINE_STATE_SERIALISATIONS` (line 70-74)
- `PIPELINE_STATE_CACHE_*` (lines 76-93)
- `record_pipeline_stage` function (lines 186-189)

**Implementation**: Collectors for:

- `pipeline_stage_duration_seconds` with labels: `["stage_name", "status"]`
- `pipeline_items_processed_total` with labels: `["stage_name", "item_type"]`
- `pipeline_errors_total` with labels: `["stage_name", "error_type"]`
- `pipeline_active_stages` with labels: `["stage_name"]`

**Acceptance Criteria**: Same as 1.1.2 but for pipeline domain

**Dependencies**: Task 1.1.1

---

### 1.1.5 Implement CacheMetricRegistry

**File**: `src/Medical_KG_rev/observability/registries/cache.py` (NEW)

**Action**: Create caching layer metric registry

**Reference**: Current cache metrics:

- `PIPELINE_STATE_CACHE_HITS` (lines 77-81)
- `PIPELINE_STATE_CACHE_MISSES` (lines 83-87)
- `PIPELINE_STATE_CACHE_SIZE` (lines 89-93)
- Functions `record_cache_hit_rate`, `record_cache_miss_rate` (lines 161-167)

**Implementation**: Collectors for:

- `cache_hits_total` with labels: `["cache_name", "key_type"]`
- `cache_misses_total` with labels: `["cache_name", "key_type"]`
- `cache_gets_total` with labels: `["cache_name", "key_type"]`
- `cache_sets_total` with labels: `["cache_name", "key_type"]`
- `cache_evictions_total` with labels: `["cache_name", "reason"]`
- `cache_item_count` with labels: `["cache_name"]`
- `cache_read_duration_seconds` with labels: `["cache_name", "key_type"]`
- `cache_write_duration_seconds` with labels: `["cache_name", "key_type"]`

**Acceptance Criteria**: Same as 1.1.2 but for cache domain

**Dependencies**: Task 1.1.1

---

### 1.1.6 Implement RerankingMetricRegistry

**File**: `src/Medical_KG_rev/observability/registries/reranking.py` (NEW)

**Action**: Create search reranking metric registry

**Reference**: Current reranking metrics:

- `record_reranking_operation` function (lines 177-180)
- `record_reranking_error` function (lines 182-184)

**Implementation**: Collectors for:

- `reranking_requests_total` with labels: `["model_name", "status"]`
- `reranking_request_duration_seconds` with labels: `["model_name", "status"]`
- `reranking_inference_duration_seconds` with labels: `["model_name"]`
- `reranking_batch_size` with labels: `["model_name"]`
- `reranking_errors_total` with labels: `["model_name", "error_type"]`

**Acceptance Criteria**: Same as 1.1.2 but for reranking domain

**Dependencies**: Task 1.1.1

---

### 1.1.7 Implement gRPCMetricRegistry (NEW)

**File**: `src/Medical_KG_rev/observability/registries/grpc.py` (NEW)

**Action**: Create gRPC-specific metric registry for internal service-to-service communication

**Implementation**: Collectors for:

- `grpc_calls_total` with labels: `["service", "method", "status"]`
- `grpc_call_duration_seconds` with labels: `["service", "method", "status"]`
- `grpc_in_flight_calls` with labels: `["service", "method"]`
- `grpc_errors_total` with labels: `["service", "method", "error_type"]`
- `grpc_stream_messages_total` with labels: `["service", "method", "direction"]`
- `grpc_circuit_breaker_state` with labels: `["service"]`
- `grpc_connection_pool_active` with labels: `["service"]`

**Acceptance Criteria**: Same as 1.1.2 but for gRPC domain

**Dependencies**: Task 1.1.1

---

### 1.2.1 Add Feature Flag for Metric Registries

**File**: `src/Medical_KG_rev/config/settings.py`

**Action**: Add feature flag to toggle domain-specific registries

**Current State**: Check if `Settings` class exists, likely uses Pydantic `BaseSettings`

**Implementation**: Add field to settings class:

```python
class Settings(BaseSettings):
    # ... existing fields ...

    # Observability feature flags
    use_domain_registries: bool = Field(
        default=False,
        description="Enable domain-specific metric registries",
        env="USE_DOMAIN_REGISTRIES"
    )
```

**Acceptance Criteria**:

- [x] Feature flag added to `Settings` class
- [x] Default value is `False` (backward compatible)
- [x] Environment variable `USE_DOMAIN_REGISTRIES` supported
- [x] Type hint is `bool`
- [x] Field has description
- [x] No breaking changes to existing settings

**Dependencies**: None

---

### 1.2.2 Create Registry Factory

**File**: `src/Medical_KG_rev/observability/registries/__init__.py` (NEW)

**Action**: Create factory for instantiating registries based on feature flag

**Implementation**:

```python
from Medical_KG_rev.config.settings import get_settings
from Medical_KG_rev.observability.registries.gpu import GPUMetricRegistry
from Medical_KG_rev.observability.registries.http import HTTPMetricRegistry
from Medical_KG_rev.observability.registries.pipeline import PipelineMetricRegistry
from Medical_KG_rev.observability.registries.cache import CacheMetricRegistry
from Medical_KG_rev.observability.registries.reranking import RerankingMetricRegistry

_GPU_REGISTRY: GPUMetricRegistry | None = None
_HTTP_REGISTRY: HTTPMetricRegistry | None = None
_PIPELINE_REGISTRY: PipelineMetricRegistry | None = None
_CACHE_REGISTRY: CacheMetricRegistry | None = None
_RERANKING_REGISTRY: RerankingMetricRegistry | None = None

def get_gpu_registry() -> GPUMetricRegistry:
    global _GPU_REGISTRY
    if _GPU_REGISTRY is None:
        _GPU_REGISTRY = GPUMetricRegistry()
    return _GPU_REGISTRY

# Similar for other registries...

__all__ = [
    "get_gpu_registry",
    "get_http_registry",
    "get_pipeline_registry",
    "get_cache_registry",
    "get_reranking_registry",
]
```

**Acceptance Criteria**:

- [x] Singleton instances for each registry
- [x] Factory functions return cached instances
- [x] `__all__` exports factory functions
- [x] Type hints on all functions
- [x] Module imports successfully

**Dependencies**: Tasks 1.1.2-1.1.7

---

### 1.3.1 Migrate GPU Service Call Sites

**File**: Multiple files using GPU metrics

**Action**: Update GPU service metric collection to use new registry

**Search Command**: `rg "GPU_SERVICE_CALLS_TOTAL\\.labels" --type py`

**Current Usage Example** (`src/Medical_KG_rev/observability/metrics.py:67`):

```python
GPU_SERVICE_CALLS_TOTAL.labels(service="gateway", method=event, status="success").inc()
```

**New Usage**:

```python
from Medical_KG_rev.observability.registries import get_gpu_registry
from Medical_KG_rev.config.settings import get_settings

settings = get_settings()
if settings.use_domain_registries:
    get_gpu_registry().record_service_call(service="gateway", method=event, status="success")
else:
    GPU_SERVICE_CALLS_TOTAL.labels(service="gateway", method=event, status="success").inc()
```

**Files to Update** (run search to identify all):

- `src/Medical_KG_rev/services/*/`
- `src/Medical_KG_rev/gateway/*/`
- `src/Medical_KG_rev/orchestration/*/`

**Acceptance Criteria**:

- [x] All GPU metric call sites updated
- [x] Feature flag checked before using new registry
- [x] Old metric calls still work when flag disabled
- [x] No duplicate metric exports
- [x] Tests pass with flag enabled and disabled

**Dependencies**: Tasks 1.1.2, 1.2.1, 1.2.2

---

### 1.3.2-1.3.5 Migrate Other Domain Call Sites

**Similar to 1.3.1 but for**:

- HTTP metrics (1.3.2)
- Pipeline metrics (1.3.3)
- Cache metrics (1.3.4)
- Reranking metrics (1.3.5)

**Dependencies**: Respective registry implementations + 1.2.1 + 1.2.2

---

### 1.4.1 Create Unit Tests for GPU Registry

**File**: `tests/observability/registries/test_gpu_registry.py` (NEW)

**Action**: Comprehensive unit tests for `GPUMetricRegistry`

**Test Cases**:

```python
import pytest
from Medical_KG_rev.observability.registries.gpu import GPUMetricRegistry

def test_gpu_registry_initialization():
    registry = GPUMetricRegistry()
    assert registry.domain == "gpu"
    assert "service_calls" in registry._collectors

def test_record_service_call():
    registry = GPUMetricRegistry()
    registry.record_service_call("embedding", "embed", "success")
    # Assert metric value increased

def test_observe_call_duration():
    registry = GPUMetricRegistry()
    registry.observe_call_duration("embedding", "embed", 1.5)
    # Assert histogram recorded value

def test_get_collector_not_found():
    registry = GPUMetricRegistry()
    with pytest.raises(KeyError, match="not found in gpu registry"):
        registry.get_collector("nonexistent")

# Add tests for all public methods...
```

**Acceptance Criteria**:

- [ ] 100% code coverage for `GPUMetricRegistry`
- [ ] Tests for all public methods
- [ ] Edge case tests (invalid inputs, None values)
- [ ] Parametrized tests for different label combinations
- [ ] Passes `pytest -v`

**Dependencies**: Task 1.1.2

---

### 1.4.2-1.4.5 Create Unit Tests for Other Registries

**Similar structure to 1.4.1 for**:

- HTTP registry (1.4.2)
- Pipeline registry (1.4.3)
- Cache registry (1.4.4)
- Reranking registry (1.4.5)

**Dependencies**: Respective registry implementations

---

### 1.4.6 Integration Test for Feature Flag Toggle

**File**: `tests/observability/test_metric_registry_integration.py` (NEW)

**Action**: Test that feature flag correctly toggles metric collection

**Test Implementation**:

```python
import pytest
from unittest.mock import patch
from Medical_KG_rev.config.settings import Settings
from Medical_KG_rev.observability.registries import get_gpu_registry

def test_feature_flag_disabled_uses_old_metrics():
    with patch.object(Settings, "use_domain_registries", False):
        # Call code that should use old metrics
        # Verify old metric updated, new registry not called

def test_feature_flag_enabled_uses_new_registries():
    with patch.object(Settings, "use_domain_registries", True):
        # Call code that should use new registries
        # Verify new registry called, old metric not updated
```

**Acceptance Criteria**:

- [ ] Tests pass with flag enabled
- [ ] Tests pass with flag disabled
- [ ] No duplicate metrics exported
- [ ] Performance impact < 5% overhead

**Dependencies**: All 1.3.* tasks

---

## Phase 2: Typed EmbeddingStage Contracts

### 2.1.1 Create EmbeddingRequest Pydantic Model

**File**: `src/Medical_KG_rev/orchestration/stages/embedding/contracts.py` (NEW)

**Action**: Define typed request contract for embedding operations

**Current Anti-Pattern** (`src/Medical_KG_rev/orchestration/ingestion_pipeline.py:45-49`):

```python
request = type('Request', (), {
    'texts': texts,
    'namespaces': self.namespaces,
    'models': self.models
})()
```

**New Implementation**:

```python
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Any

class EmbeddingRequest(BaseModel):
    """Typed request for embedding generation."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    texts: tuple[str, ...] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Text chunks to embed"
    )

    namespace: str = Field(
        ...,
        pattern=r"^[a-z0-9_]+$",
        min_length=1,
        max_length=100,
        description="Embedding namespace"
    )

    model_id: str = Field(
        ...,
        min_length=1,
        description="Embedding model identifier"
    )

    correlation_id: str | None = Field(
        default=None,
        description="Correlation ID for tracing"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for processing"
    )

    @field_validator("texts")
    @classmethod
    def validate_text_length(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        """Validate individual text length."""
        for i, text in enumerate(v):
            if len(text) > 10000:
                raise ValueError(
                    f"Text at index {i} exceeds 10000 chars: {len(text)}"
                )
            if not text.strip():
                raise ValueError(f"Text at index {i} is empty or whitespace only")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata_size(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Limit metadata size to prevent memory issues."""
        if len(str(v)) > 10000:
            raise ValueError("Metadata too large (>10KB)")
        return v
```

**Acceptance Criteria**:

- [ ] Pydantic v2 `BaseModel` with `ConfigDict`
- [ ] All fields have type hints and descriptions
- [ ] Frozen=True for immutability
- [ ] Validators for data quality
- [ ] Field constraints (min/max lengths, patterns)
- [ ] Comprehensive docstrings
- [ ] Passes `mypy --strict`
- [ ] JSON schema generation works

**Dependencies**: None

---

### 2.1.2 Create EmbeddingResult Pydantic Model

**File**: Same as 2.1.1

**Action**: Define typed response contract

**Current Anti-Pattern** (`src/Medical_KG_rev/orchestration/ingestion_pipeline.py:55-75`):

```python
context.data["embeddings"] = response.vectors
context.data["metrics"] = {"embedding": {"vectors": len(response.vectors)}}
context.data["embedding_summary"] = {...}
```

**New Implementation**:

```python
from datetime import datetime

class EmbeddingVector(BaseModel):
    """Single embedding vector."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    vector: tuple[float, ...] = Field(..., min_length=1)
    model_id: str
    namespace: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class EmbeddingResult(BaseModel):
    """Typed result from embedding generation."""

    model_config = ConfigDict(frozen=True)

    vectors: tuple[EmbeddingVector, ...] = Field(..., min_length=1)
    model_id: str
    namespace: str
    processing_time_ms: float = Field(..., ge=0)
    gpu_memory_used_mb: int | None = Field(default=None, ge=0)
    correlation_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def vector_count(self) -> int:
        return len(self.vectors)

    @property
    def per_namespace_counts(self) -> dict[str, int]:
        """Count vectors by namespace."""
        counts: dict[str, int] = {}
        for vec in self.vectors:
            counts[vec.namespace] = counts.get(vec.namespace, 0) + 1
        return counts
```

**Acceptance Criteria**: Same as 2.1.1

**Dependencies**: None

---

### 2.1.3 Create EmbeddingError Exception Hierarchy

**File**: Same as 2.1.1

**Action**: Define structured exceptions for embedding operations

**Implementation**:

```python
class EmbeddingError(Exception):
    """Base exception for embedding operations."""

    def __init__(self, message: str, *, correlation_id: str | None = None):
        super().__init__(message)
        self.correlation_id = correlation_id

class EmbeddingValidationError(EmbeddingError):
    """Validation failed for embedding request."""

    def __init__(self, message: str, *, field: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.field = field

class EmbeddingProcessingError(EmbeddingError):
    """Processing failed during embedding generation."""

    def __init__(
        self,
        message: str,
        *,
        retry_possible: bool = False,
        error_type: str = "unknown",
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.retry_possible = retry_possible
        self.error_type = error_type

class EmbeddingServiceUnavailableError(EmbeddingProcessingError):
    """Embedding service is unavailable."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            retry_possible=True,
            error_type="service_unavailable",
            **kwargs
        )
```

**Acceptance Criteria**:

- [ ] Clear exception hierarchy
- [ ] Structured error information (field, retry_possible, error_type)
- [ ] Correlation ID propagation
- [ ] Docstrings for each exception
- [ ] Type hints

**Dependencies**: None

---

### 2.2.1 Implement EmbeddingStageV2 with Typed Contracts

**File**: `src/Medical_KG_rev/orchestration/stages/embedding/stage_v2.py` (NEW)

**Action**: Refactor EmbeddingStage to use typed contracts

**Current Implementation** (`src/Medical_KG_rev/orchestration/ingestion_pipeline.py:11-84`)

**New Implementation**:

```python
from typing import Protocol
from Medical_KG_rev.orchestration.stages.types import PipelineContext
from Medical_KG_rev.orchestration.stages.embedding.contracts import (
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingValidationError,
    EmbeddingProcessingError,
)
import structlog

logger = structlog.get_logger(__name__)

class EmbeddingWorkerProtocol(Protocol):
    """Protocol for embedding workers."""
    def run(self, request: EmbeddingRequest) -> EmbeddingResult: ...

class EmbeddingStageV2:
    """Embedding stage with typed contracts."""

    def __init__(
        self,
        worker: EmbeddingWorkerProtocol,
        namespace: str,
        model_id: str,
    ):
        self.worker = worker
        self.namespace = namespace
        self.model_id = model_id

    def execute(self, context: PipelineContext) -> EmbeddingResult:
        """Execute embedding with typed contracts.

        Args:
            context: Pipeline context containing chunks

        Returns:
            EmbeddingResult with vectors and metadata

        Raises:
            EmbeddingValidationError: Invalid input
            EmbeddingProcessingError: Processing failed
        """
        # Validate input
        if not context.data.get("chunks"):
            raise EmbeddingValidationError(
                "No chunks provided",
                field="chunks",
                correlation_id=context.correlation_id
            )

        try:
            # Extract chunks
            chunks = context.data["chunks"]
            texts = tuple(chunk["body"] for chunk in chunks)

            # Create typed request
            request = EmbeddingRequest(
                texts=texts,
                namespace=self.namespace,
                model_id=self.model_id,
                correlation_id=context.correlation_id,
                metadata=context.metadata
            )

            logger.info(
                "embedding.stage.execute",
                text_count=len(texts),
                namespace=self.namespace,
                model_id=self.model_id,
                correlation_id=context.correlation_id
            )

            # Call worker with typed request
            result = self.worker.run(request)

            logger.info(
                "embedding.stage.complete",
                vector_count=result.vector_count,
                processing_time_ms=result.processing_time_ms,
                correlation_id=context.correlation_id
            )

            return result

        except ValueError as e:
            raise EmbeddingValidationError(
                str(e),
                correlation_id=context.correlation_id
            ) from e
        except Exception as e:
            # Check for GPU errors
            if "GPU" in str(e) or "gpu" in str(e).lower():
                raise EmbeddingProcessingError(
                    str(e),
                    error_type="gpu_unavailable",
                    correlation_id=context.correlation_id
                ) from e
            raise EmbeddingProcessingError(
                str(e),
                correlation_id=context.correlation_id
            ) from e
```

**Acceptance Criteria**:

- [ ] Uses `EmbeddingRequest` instead of dynamic object
- [ ] Returns `EmbeddingResult` instead of mutating context
- [ ] Comprehensive validation with clear errors
- [ ] Structured logging with correlation IDs
- [ ] Type hints on all methods
- [ ] Docstrings with Args/Returns/Raises
- [ ] Passes `mypy --strict`
- [ ] Unit tests with 100% coverage

**Dependencies**: Tasks 2.1.1, 2.1.2, 2.1.3

---

### 2.2.2 Add Feature Flag for Typed EmbeddingStage

**File**: `src/Medical_KG_rev/config/settings.py`

**Action**: Add feature flag to toggle new embedding stage

**Implementation**:

```python
class Settings(BaseSettings):
    # ... existing fields ...

    use_typed_embedding_stage: bool = Field(
        default=False,
        description="Enable typed EmbeddingStage contracts",
        env="USE_TYPED_EMBEDDING_STAGE"
    )
```

**Acceptance Criteria**:

- [ ] Feature flag added
- [ ] Default is `False`
- [ ] Environment variable supported
- [ ] No breaking changes

**Dependencies**: None

---

### 2.3.1 Update PipelineState to Handle EmbeddingResult

**File**: `src/Medical_KG_rev/orchestration/stages/contracts.py`

**Action**: Add support for `EmbeddingResult` alongside existing embedding_batch

**Current State**: Line 899 has `embedding_batch: EmbeddingBatch | None = None`

**Changes**:

```python
from Medical_KG_rev.orchestration.stages.embedding.contracts import EmbeddingResult

@dataclass(slots=True)
class PipelineState:
    # ... existing fields ...

    embedding_batch: EmbeddingBatch | None = None
    embedding_result: EmbeddingResult | None = None  # NEW

    # ... existing methods ...

    def set_embedding_result(self, result: EmbeddingResult) -> None:
        """Set typed embedding result."""
        self.embedding_result = result
        self._mark_dirty()

    def require_embedding_result(self) -> EmbeddingResult:
        """Get embedding result or raise."""
        if self.embedding_result is None:
            raise ValueError("PipelineState does not contain embedding result")
        return self.embedding_result

    def has_embedding_result(self) -> bool:
        """Check if embedding result exists."""
        return self.embedding_result is not None
```

**Acceptance Criteria**:

- [ ] New field `embedding_result` added
- [ ] Accessor methods follow existing patterns
- [ ] No breaking changes to existing `embedding_batch` usage
- [ ] Type hints correct
- [ ] Tests updated

**Dependencies**: Task 2.1.2

---

### 2.3.2 Create Result Transformation Utilities

**File**: `src/Medical_KG_rev/orchestration/stages/embedding/transforms.py` (NEW)

**Action**: Provide utilities to convert between old and new formats

**Implementation**:

```python
from typing import Any
from Medical_KG_rev.orchestration.stages.embedding.contracts import EmbeddingResult

def result_to_context_data(result: EmbeddingResult) -> dict[str, Any]:
    """Convert EmbeddingResult to legacy context.data format.

    For backward compatibility with stages expecting old format.
    """
    return {
        "embeddings": [
            {
                "chunk_id": vec.chunk_id,
                "vector": list(vec.vector),
                "model_id": vec.model_id,
                "namespace": vec.namespace,
                "metadata": vec.metadata,
            }
            for vec in result.vectors
        ],
        "metrics": {
            "embedding": {
                "vectors": result.vector_count,
                "processing_time_ms": result.processing_time_ms,
            }
        },
        "embedding_summary": {
            "vectors": result.vector_count,
            "per_namespace": result.per_namespace_counts,
            "model_id": result.model_id,
            "timestamp": result.timestamp.isoformat(),
        }
    }

def context_data_to_result(data: dict[str, Any]) -> EmbeddingResult:
    """Convert legacy context.data format to EmbeddingResult.

    For migrating existing pipeline state.
    """
    from Medical_KG_rev.orchestration.stages.embedding.contracts import (
        EmbeddingResult,
        EmbeddingVector,
    )

    vectors = tuple(
        EmbeddingVector(
            chunk_id=emb["chunk_id"],
            vector=tuple(emb["vector"]),
            model_id=emb["model_id"],
            namespace=emb["namespace"],
            metadata=emb.get("metadata", {}),
        )
        for emb in data["embeddings"]
    )

    return EmbeddingResult(
        vectors=vectors,
        model_id=data["embedding_summary"]["model_id"],
        namespace=vectors[0].namespace if vectors else "default",
        processing_time_ms=data["metrics"]["embedding"]["processing_time_ms"],
    )
```

**Acceptance Criteria**:

- [ ] Bidirectional conversion utilities
- [ ] Handles edge cases (empty results, missing fields)
- [ ] Type hints
- [ ] Unit tests with various input formats
- [ ] Docstrings

**Dependencies**: Task 2.1.2

---

## Phase 3: Torch-Isolated Qwen3 Service

### 3.1.1 Create Qwen3GRPCClient

**File**: `src/Medical_KG_rev/services/clients/qwen3_grpc_client.py` (NEW)

**Action**: Implement gRPC client for Qwen3 embedding service

**Existing Proto**: `src/Medical_KG_rev/proto/embedding_service.proto`

**Implementation**:

```python
import grpc
from typing import Any
from Medical_KG_rev.proto import embedding_service_pb2, embedding_service_pb2_grpc
from Medical_KG_rev.services.mineru.circuit_breaker import CircuitBreaker, CircuitState
import structlog

logger = structlog.get_logger(__name__)

class GPUServiceUnavailableError(Exception):
    """GPU service is unavailable."""
    pass

class Qwen3GRPCClient:
    """gRPC client for Qwen3 embedding service."""

    def __init__(
        self,
        endpoint: str,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """Initialize gRPC client.

        Args:
            endpoint: gRPC endpoint (host:port)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries

        # Create channel
        self.channel = grpc.insecure_channel(endpoint)
        self.stub = embedding_service_pb2_grpc.EmbeddingServiceStub(self.channel)

        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            name="qwen3_grpc"
        )

        logger.info(
            "qwen3_grpc.client.initialized",
            endpoint=endpoint,
            timeout=timeout
        )

    def embed_texts(
        self,
        texts: list[str],
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        **kwargs: Any
    ) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            model_name: Model identifier
            **kwargs: Additional parameters

        Returns:
            List of embedding vectors

        Raises:
            GPUServiceUnavailableError: Service unavailable
        """
        if self.circuit_breaker.state == CircuitState.OPEN:
            raise GPUServiceUnavailableError(
                f"Circuit breaker OPEN for Qwen3 service: {self.endpoint}"
            )

        request = embedding_service_pb2.BatchEmbedRequest(
            texts=texts,
            model_name=model_name,
        )

        try:
            logger.debug(
                "qwen3_grpc.embed_texts.start",
                text_count=len(texts),
                model=model_name
            )

            response = self.stub.BatchEmbed(request, timeout=self.timeout)

            embeddings = [list(emb.values) for emb in response.embeddings]

            logger.info(
                "qwen3_grpc.embed_texts.success",
                text_count=len(texts),
                embedding_count=len(embeddings)
            )

            self.circuit_breaker.record_success()
            return embeddings

        except grpc.RpcError as e:
            self.circuit_breaker.record_failure()

            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise GPUServiceUnavailableError(
                    f"GPU service unavailable: {e.details()}"
                ) from e

            logger.error(
                "qwen3_grpc.embed_texts.error",
                error_code=e.code(),
                error_details=e.details()
            )
            raise

    def health_check(self) -> bool:
        """Check if service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            request = embedding_service_pb2.HealthRequest()
            response = self.stub.GetHealth(request, timeout=5.0)
            return response.status == "healthy"
        except grpc.RpcError:
            return False

    def close(self) -> None:
        """Close gRPC channel."""
        if self.channel:
            self.channel.close()
            logger.info("qwen3_grpc.client.closed", endpoint=self.endpoint)
```

**Acceptance Criteria**:

- [ ] Implements existing `embedding_service.proto` contract
- [ ] Circuit breaker integration
- [ ] Comprehensive error handling
- [ ] Structured logging
- [ ] Type hints
- [ ] Docstrings with Args/Returns/Raises
- [ ] Connection pooling (optional)
- [ ] Unit tests with mock gRPC
- [ ] Integration tests with test server

**Dependencies**: Existing proto definitions, circuit breaker module

---

### 3.1.2 Add Feature Flag for Qwen3 gRPC

**File**: `src/Medical_KG_rev/config/settings.py`

**Action**: Add feature flag to toggle gRPC-based Qwen3 service

**Implementation**:

```python
class Settings(BaseSettings):
    # ... existing fields ...

    qwen3_use_grpc: bool = Field(
        default=False,
        description="Use gRPC for Qwen3 embedding service",
        env="QWEN3_USE_GRPC"
    )

    qwen3_grpc_endpoint: str = Field(
        default="localhost:50052",
        description="Qwen3 gRPC service endpoint",
        env="QWEN3_GRPC_ENDPOINT"
    )

    qwen3_grpc_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Qwen3 gRPC request timeout (seconds)",
        env="QWEN3_GRPC_TIMEOUT"
    )
```

**Acceptance Criteria**:

- [ ] Three related settings added
- [ ] Defaults are sensible
- [ ] Environment variables supported
- [ ] Type validation on timeout

**Dependencies**: None

---

### 3.2.1 Refactor Qwen3Service to Use gRPC

**File**: `src/Medical_KG_rev/services/retrieval/qwen3_service.py`

**Action**: Modify service to use gRPC client when flag enabled

**Current State**: Lines 86-100 load model in-process

**Changes**:

```python
from Medical_KG_rev.config.settings import get_settings
from Medical_KG_rev.services.clients.qwen3_grpc_client import (
    Qwen3GRPCClient,
    GPUServiceUnavailableError,
)

class Qwen3Service:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        embedding_dimension: int = 4096,
        batch_size: int = 8,
        max_seq_length: int = 2048,
        gpu_manager: Any = None,
    ):
        self.model_name = model_name
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        settings = get_settings()

        if settings.qwen3_use_grpc:
            # Use gRPC client
            self.grpc_client = Qwen3GRPCClient(
                endpoint=settings.qwen3_grpc_endpoint,
                timeout=settings.qwen3_grpc_timeout,
            )
            self.use_grpc = True
            logger.info(
                "qwen3.service.initialized",
                mode="grpc",
                endpoint=settings.qwen3_grpc_endpoint
            )
        else:
            # Legacy in-process model loading
            # TODO: Remove after migration complete
            self.grpc_client = None
            self.use_grpc = False
            # ... existing initialization code ...
            logger.warning(
                "qwen3.service.initialized",
                mode="in_process",
                deprecation_warning="In-process mode will be removed in future version"
            )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings (gRPC or in-process)."""
        if self.use_grpc:
            return self.grpc_client.embed_texts(
                texts=texts,
                model_name=self.model_name
            )
        else:
            # Legacy in-process implementation
            # ... existing code ...
```

**Acceptance Criteria**:

- [ ] Feature flag determines mode
- [ ] gRPC client used when flag enabled
- [ ] Legacy code path still works when disabled
- [ ] Deprecation warning logged for in-process mode
- [ ] API compatibility maintained
- [ ] Health check uses gRPC when appropriate
- [ ] Tests cover both code paths
- [ ] No breaking changes to callers

**Dependencies**: Tasks 3.1.1, 3.1.2

---

## Phase 4: Remove Simulation Artifacts

### 4.1.1 Catalog Simulation Artifacts

**File**: `openspec/changes/modernize-observability-architecture/SIMULATION_ARTIFACT_CATALOG.md` (NEW)

**Action**: Document all simulation code to be removed

**Search Commands**:

```bash
rg "class.*Simulator" --type py
rg "Mock.*Client" --type py
rg "# Simulate|# Mock" --type py
find . -name "*vllm*" -o -name "*mineru*" | grep -E "\.(py|yaml)$"
```

**Documentation Format**:

```markdown
## Files to Delete

### src/Medical_KG_rev/services/mineru/vllm_client.py
- **Lines**: 1-86
- **Type**: Mock VLLMClient
- **Used By**: tests/integration/test_vllm_integration.py
- **Replacement**: Real gRPC client or testcontainers

### src/Medical_KG_rev/services/mineru/cli_wrapper.py
- **Lines**: All
- **Type**: Simulation wrapper
- **Used By**: tests/services/test_gpu_microservices.py
- **Replacement**: Mock at gRPC level

## Integration Gaps Masked

1. **VLLM Connection**: Simulation hides actual connection failures
2. **MinerU Processing**: Mock doesn't reflect real processing times
3. **GPU Availability**: Simulation always returns "available"
```

**Acceptance Criteria**:

- [ ] All simulation files cataloged
- [ ] Usage documented
- [ ] Integration gaps identified
- [ ] Replacement strategy for each

**Dependencies**: None

---

### 4.2.1 Delete VLLMClient Simulation

**File**: `src/Medical_KG_rev/services/mineru/vllm_client.py`

**Action**: Delete file

**Command**: `rm src/Medical_KG_rev/services/mineru/vllm_client.py`

**Before Deletion**:

1. Verify no production code imports this
2. Document tests that need updating
3. Create replacement mocks

**Acceptance Criteria**:

- [ ] File deleted
- [ ] No import errors in production code
- [ ] Tests documented for updates
- [ ] Git commit with rationale

**Dependencies**: Task 4.1.1

---

### 4.3.1 Update VLLM Integration Tests

**File**: `tests/integration/test_vllm_integration.py`

**Action**: Replace simulation with testcontainers

**Current State**: Uses `VLLMClient` mock

**New Implementation**:

```python
import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_strategies import wait_for_logs

@pytest.fixture(scope="session")
def vllm_container():
    """Start real vLLM container for integration tests."""
    container = DockerContainer("vllm/vllm-openai:latest")
    container.with_exposed_ports(8000)
    container.with_env("MODEL", "Qwen/Qwen2.5-7B-Instruct")
    container.with_command("--model Qwen/Qwen2.5-7B-Instruct")

    with container:
        wait_for_logs(container, "Uvicorn running", timeout=60)
        yield container

def test_vllm_embedding_generation(vllm_container):
    """Test embedding generation with real vLLM service."""
    endpoint = f"localhost:{vllm_container.get_exposed_port(8000)}"

    # Use real gRPC client
    from Medical_KG_rev.services.clients.qwen3_grpc_client import Qwen3GRPCClient

    client = Qwen3GRPCClient(endpoint)
    embeddings = client.embed_texts(["test text"])

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 4096  # Qwen3 dimension
```

**Acceptance Criteria**:

- [ ] Uses testcontainers
- [ ] Real service integration
- [ ] Tests pass with actual service
- [ ] Proper cleanup
- [ ] CI configuration updated

**Dependencies**: Tasks 4.2.1, 3.1.1

---

## Phase 5: Finalization

### 5.1.1 Enable Feature Flags by Default

**File**: `src/Medical_KG_rev/config/settings.py`

**Action**: Change default values for feature flags

**Changes**:

```python
class Settings(BaseSettings):
    use_domain_registries: bool = Field(default=True, ...)  # Changed from False
    use_typed_embedding_stage: bool = Field(default=True, ...)  # Changed from False
    qwen3_use_grpc: bool = Field(default=True, ...)  # Changed from False
```

**Before Enabling**:

1. Run full regression suite
2. Performance validation in staging
3. Dashboard verification
4. Rollback plan documented

**Acceptance Criteria**:

- [ ] Defaults changed
- [ ] All tests pass
- [ ] Performance validated
- [ ] Dashboards updated
- [ ] Rollback procedure documented

**Dependencies**: All previous phases complete

---

### 5.2.1 Remove Deprecated Metric Functions

**File**: `src/Medical_KG_rev/observability/metrics.py`

**Action**: Delete deprecated metric helper functions

**Functions to Remove**:

- `observe_job_duration()` (lines 59-62)
- `record_business_event()` (lines 64-67)
- `record_resilience_circuit_state()` (lines 96-98)
- Other functions using GPU metrics for non-GPU operations

**Before Removal**:

1. Verify no callers remain
2. 2-week deprecation notice complete
3. Migration guide published

**Acceptance Criteria**:

- [ ] Deprecated functions removed
- [ ] No import errors
- [ ] Tests updated
- [ ] Documentation updated

**Dependencies**: Task 5.1.1 + 2-week deprecation period

---

## Testing Strategy

### Per-Task Testing

- Run `pytest tests/path/to/test_file.py -v` after each task
- Run `mypy src/path/to/file.py --strict` for type checking
- Use `pytest-cov` for coverage: `pytest --cov=src/Medical_KG_rev --cov-report=term-missing`

### Integration Testing

- After Phase 1: Test metric collection with both flags
- After Phase 2: Test full pipeline with typed contracts
- After Phase 3: Test gRPC service integration
- After Phase 4: Verify no simulation dependencies

### Performance Benchmarks

- Baseline before Phase 1
- After each phase, compare to baseline
- Acceptance: < 5% performance degradation
- P95 latency < 500ms for critical paths

### Rollback Testing

- Test feature flag rollback for each phase
- Verify old code path still works
- Document rollback procedures

---

## Success Criteria

### Metrics

- [ ] Label cardinality reduced by 60%+
- [ ] Metric collection overhead < 5%
- [ ] Dashboard compatibility maintained

### Code Quality

- [ ] 100% type hint coverage in new code
- [ ] 90%+ test coverage for new code
- [ ] Zero `mypy --strict` errors
- [ ] All docstrings follow Google style

### Performance

- [ ] P95 latency increase < 50ms
- [ ] No memory leaks
- [ ] Circuit breakers functional

### Architecture

- [ ] Zero in-process torch dependencies in main gateway
- [ ] All simulation artifacts removed
- [ ] gRPC integration complete

---

## Rollback Plan

### Phase 1 Rollback

```bash
export USE_DOMAIN_REGISTRIES=false
# Restart services
```

### Phase 2 Rollback

```bash
export USE_TYPED_EMBEDDING_STAGE=false
# Restart services
```

### Phase 3 Rollback

```bash
export QWEN3_USE_GRPC=false
# Restart services, falls back to in-process
```

### Emergency Rollback

1. Git revert to previous commit
2. Redeploy previous version
3. Restore Grafana dashboards from backup
4. Incident post-mortem

---

## Monitoring During Rollout

### Key Metrics to Watch

- `gpu_service_calls_total` - should split by domain
- `embedding_stage_duration_seconds` - should not increase
- `qwen3_grpc_errors_total` - should remain low
- `circuit_breaker_state` - should be mostly closed

### Alerts

- P95 latency > 500ms
- Error rate > 5%
- Circuit breaker open > 5 minutes
- Memory usage increase > 20%

### Dashboard Updates

- Before Phase 1: Update to query new metric names
- Before Phase 5: Switch default dashboards to new metrics
- Backup old dashboards to `dashboards/legacy/`
