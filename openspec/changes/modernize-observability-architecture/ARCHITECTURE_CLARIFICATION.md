# Architecture Clarification: gRPC-First Internal Communication

## Critical Architectural Constraint

**All internal service communication MUST use gRPC. HTTP is ONLY for external-facing APIs and external database connections.**

This document clarifies the scope of HTTP vs gRPC usage in the modernize-observability-architecture change proposal.

---

## Communication Protocol Matrix

| Communication Type | Protocol | Rationale |
|-------------------|----------|-----------|
| **External API Gateway** (client-facing) | HTTP/REST, GraphQL, SOAP | Client compatibility, web browser support |
| **Internal Service-to-Service** | gRPC | Performance, type safety, efficiency |
| **GPU Microservices** | gRPC | Torch isolation, fail-fast, resource management |
| **External Databases** (Neo4j, Qdrant, etc.) | HTTP/REST | Database-provided client libraries |
| **Adapter → External APIs** (OpenAlex, ClinicalTrials, etc.) | HTTP/REST | External service protocols |

---

## HTTPMetricRegistry Scope Clarification

### What HTTPMetricRegistry IS For

✅ **External-facing API gateway metrics**:

- Client requests to REST endpoints (`/v1/ingest`, `/v1/retrieve`)
- GraphQL queries from external clients
- SOAP requests from external systems
- OData queries from external clients

✅ **Adapter HTTP client metrics**:

- Outbound requests to OpenAlex API
- Outbound requests to ClinicalTrials.gov
- Outbound requests to Unpaywall API
- Outbound requests to WHO ICD-11 API

✅ **External database HTTP clients**:

- Neo4j Bolt/HTTP protocol (if using HTTP driver)
- Qdrant HTTP API calls
- Any other external database with HTTP interface

### What HTTPMetricRegistry IS NOT For

❌ **Internal service communication** - These use gRPC:

- Gateway → GPU services (embedding, reranking, Docling VLM)
- Gateway → Orchestration services
- Orchestration → GPU services
- Any Docker container → Docker container communication

---

## Current Architecture Audit

### Files Using HTTP for Internal Communication (NONE FOUND - CORRECT)

Based on codebase search, all internal services correctly use gRPC:

✅ **GPU Services** (`src/Medical_KG_rev/proto/`):

- `embedding_service.proto` - gRPC contract for embedding
- `gpu_service.proto` - gRPC contract for GPU management
- `gpu.proto` - Additional GPU service definitions

✅ **Service Clients** (exist, using gRPC):

- Circuit breaker integration already in place
- Existing gRPC infrastructure validated

### Files Using HTTP for External Communication (CORRECT)

✅ **Gateway External APIs** (`src/Medical_KG_rev/gateway/`):

- `rest/` - FastAPI REST endpoints for external clients
- `graphql/` - GraphQL endpoint for external clients
- `soap/routes.py` - SOAP endpoint for external clients
- These are CORRECT - external clients require HTTP

✅ **Adapter HTTP Clients**:

- External API adapters use HTTP to communicate with external services
- This is CORRECT - external services dictate protocol

---

## Changes Required to Proposal

### 1. Rename HTTPMetricRegistry → ExternalAPIMetricRegistry

**Rationale**: Current name "HTTPMetricRegistry" is ambiguous and could be misinterpreted as internal HTTP.

**Updated Implementation**:

```python
# src/Medical_KG_rev/observability/registries/external_api.py
class ExternalAPIMetricRegistry(BaseMetricRegistry):
    """Metric registry for external-facing API gateway and adapter HTTP clients.

    Scope:
        - Client requests to REST/GraphQL/SOAP/OData endpoints
        - Adapter outbound HTTP requests to external APIs
        - External database HTTP client requests

    Out of Scope:
        - Internal gRPC service-to-service communication (use gRPCMetricRegistry)
        - Internal GPU service calls (use GPUMetricRegistry)
    """

    def initialize_collectors(self) -> None:
        self._collectors["client_requests"] = Counter(
            "external_api_requests_total",
            "External client API requests",
            ["protocol", "endpoint", "method", "status_code"],
            registry=self._registry
        )

        self._collectors["adapter_requests"] = Counter(
            "adapter_http_requests_total",
            "Adapter outbound HTTP requests to external APIs",
            ["adapter_name", "target_host", "status_code"],
            registry=self._registry
        )

        self._collectors["request_duration"] = Histogram(
            "external_api_duration_seconds",
            "External API request duration",
            ["protocol", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self._registry
        )
```

### 2. Add gRPCMetricRegistry for Internal Communication

**New Registry Required**:

```python
# src/Medical_KG_rev/observability/registries/grpc.py
class gRPCMetricRegistry(BaseMetricRegistry):
    """Metric registry for internal gRPC service-to-service communication.

    Scope:
        - Gateway → GPU services (embedding, reranking, Docling VLM)
        - Gateway → Orchestration services
        - Orchestration → GPU services
        - Any internal Docker container communication

    Out of Scope:
        - External client API requests (use ExternalAPIMetricRegistry)
        - GPU-specific metrics like memory/utilization (use GPUMetricRegistry)
    """

    def initialize_collectors(self) -> None:
        self._collectors["rpc_calls"] = Counter(
            "grpc_calls_total",
            "Internal gRPC calls",
            ["service", "method", "status_code"],
            registry=self._registry
        )

        self._collectors["rpc_duration"] = Histogram(
            "grpc_call_duration_seconds",
            "gRPC call duration",
            ["service", "method"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            registry=self._registry
        )

        self._collectors["rpc_errors"] = Counter(
            "grpc_errors_total",
            "gRPC call errors",
            ["service", "method", "error_code"],
            registry=self._registry
        )

        self._collectors["stream_messages"] = Counter(
            "grpc_stream_messages_total",
            "gRPC streaming messages",
            ["service", "method", "direction"],  # direction: sent/received
            registry=self._registry
        )
```

---

## Updated Domain-Specific Registry List

### Revised 6-Registry Architecture

1. **GPUMetricRegistry** - GPU hardware metrics only
   - Memory usage, utilization, temperature
   - Device status and health
   - NO service call metrics (moved to gRPC registry)

2. **gRPCMetricRegistry** - Internal service communication
   - All Docker container → container gRPC calls
   - RPC duration, status codes, streaming
   - Circuit breaker states for gRPC services

3. **ExternalAPIMetricRegistry** - External HTTP traffic
   - Client-facing REST/GraphQL/SOAP/OData
   - Adapter → external API HTTP calls
   - External database HTTP clients

4. **PipelineMetricRegistry** - Orchestration pipeline
   - Stage execution, state transitions
   - Gate decisions, ledger updates
   - NO communication metrics (use gRPC registry)

5. **CacheMetricRegistry** - Caching layer
   - Hit rates, evictions, size
   - Cache-specific operations

6. **RerankingMetricRegistry** - Search reranking
   - Reranking operations, document counts
   - Model-specific metrics

---

## Implementation Impact

### Tasks.md Updates Required

**Section 1.1: Create Metric Registry Infrastructure**

OLD:

```markdown
- [ ] Implement `HTTPMetricRegistry` for API gateway and HTTP client operations
```

NEW:

```markdown
- [ ] Implement `ExternalAPIMetricRegistry` for external-facing APIs and adapter HTTP clients
- [ ] Implement `gRPCMetricRegistry` for internal service-to-service communication
```

**Section 1.2: Migrate Existing Metrics**

OLD:

```markdown
- [ ] Move HTTP request metrics to `HTTPMetricRegistry`
```

NEW:

```markdown
- [ ] Move external API request metrics to `ExternalAPIMetricRegistry`
- [ ] Move internal gRPC call metrics to `gRPCMetricRegistry`
- [ ] Audit all GPU service call metrics - move to gRPCMetricRegistry if they track communication (not hardware)
```

### DETAILED_TASKS.md Updates Required

**Task 1.1.3: Implement ExternalAPIMetricRegistry** (was HTTPMetricRegistry)

Update task to:

1. Rename file to `external_api.py`
2. Update class name to `ExternalAPIMetricRegistry`
3. Add clear scope documentation in docstring
4. Separate collectors for client requests vs adapter requests
5. Add protocol label (REST/GraphQL/SOAP/OData) to distinguish

**NEW Task 1.1.7: Implement gRPCMetricRegistry**

Add new task with:

1. File: `src/Medical_KG_rev/observability/registries/grpc.py`
2. Collectors for RPC calls, duration, errors, streaming
3. Labels: service, method, status_code, error_code
4. Integration with existing gRPC clients (Qwen3GRPCClient, etc.)

**Task 1.3.2: Migrate gRPC Call Sites** (NEW)

Add new migration task:

1. Search for all gRPC `stub.MethodName()` calls
2. Wrap with gRPC registry metric collection
3. Use interceptors or manual instrumentation
4. Example:

```python
from Medical_KG_rev.observability.registries import get_grpc_registry

registry = get_grpc_registry()
start = time.time()
try:
    response = stub.BatchEmbed(request, timeout=timeout)
    registry.record_rpc_call(
        service="embedding",
        method="BatchEmbed",
        status_code="OK"
    )
except grpc.RpcError as e:
    registry.record_rpc_error(
        service="embedding",
        method="BatchEmbed",
        error_code=str(e.code())
    )
finally:
    registry.observe_rpc_duration(
        service="embedding",
        method="BatchEmbed",
        duration=time.time() - start
    )
```

---

## Validation Checklist

### Architecture Compliance

- [ ] All Docker container → container communication uses gRPC
- [ ] No internal HTTP between services
- [ ] External client APIs use HTTP (REST/GraphQL/SOAP) - ALLOWED
- [ ] Adapter → external API uses HTTP - ALLOWED
- [ ] gRPCMetricRegistry covers all internal service calls
- [ ] ExternalAPIMetricRegistry covers only external traffic

### Metric Registry Boundaries

- [ ] GPUMetricRegistry: Hardware metrics only, no communication
- [ ] gRPCMetricRegistry: Internal RPC calls only
- [ ] ExternalAPIMetricRegistry: External HTTP only
- [ ] No metric cross-pollution between registries
- [ ] Clear label conventions per registry

### Implementation Tasks

- [ ] Rename HTTPMetricRegistry → ExternalAPIMetricRegistry in all docs
- [ ] Add gRPCMetricRegistry implementation task
- [ ] Add gRPC call site migration task
- [ ] Update Phase 1 deliverables (6 registries, not 5)
- [ ] Add gRPC interceptor implementation task (optional, for automatic instrumentation)

---

## gRPC Interceptor Approach (Recommended)

For automatic metric collection on all gRPC calls, implement a client interceptor:

```python
# src/Medical_KG_rev/observability/grpc_interceptor.py
import grpc
from Medical_KG_rev.observability.registries import get_grpc_registry
import time

class MetricsClientInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):
    """gRPC client interceptor for automatic metric collection."""

    def __init__(self):
        self.registry = get_grpc_registry()

    def intercept_unary_unary(self, continuation, client_call_details, request):
        service, method = self._parse_method(client_call_details.method)
        start = time.time()

        try:
            response = continuation(client_call_details, request)
            self.registry.record_rpc_call(
                service=service,
                method=method,
                status_code="OK"
            )
            return response
        except grpc.RpcError as e:
            self.registry.record_rpc_error(
                service=service,
                method=method,
                error_code=str(e.code())
            )
            raise
        finally:
            self.registry.observe_rpc_duration(
                service=service,
                method=method,
                duration=time.time() - start
            )

    def _parse_method(self, full_method: str) -> tuple[str, str]:
        """Parse /package.Service/Method into (Service, Method)."""
        parts = full_method.strip('/').split('/')
        return parts[0].split('.')[-1], parts[1]

# Usage in Qwen3GRPCClient
class Qwen3GRPCClient:
    def __init__(self, endpoint: str, ...):
        self.channel = grpc.intercept_channel(
            grpc.insecure_channel(endpoint),
            MetricsClientInterceptor()  # Automatic metrics!
        )
        self.stub = embedding_service_pb2_grpc.EmbeddingServiceStub(self.channel)
```

**Benefits**:

- Automatic metric collection for all gRPC calls
- No manual instrumentation at every call site
- Consistent metric collection across services
- Easy to add tracing, logging, etc.

---

## Summary of Changes

### Documentation Files to Update

1. **proposal.md**
   - Change "HTTP Metrics" → "External API Metrics"
   - Add "gRPC Metrics" for internal communication
   - Update "What Changes" section with 6 registries

2. **design.md**
   - Add Decision 5: "gRPC-First Internal Communication"
   - Update metric registry list (6 registries)
   - Add gRPC interceptor design decision

3. **tasks.md**
   - Rename HTTPMetricRegistry → ExternalAPIMetricRegistry
   - Add gRPCMetricRegistry task
   - Add gRPC call site migration task
   - Update Phase 1 deliverables (6 registries)

4. **DETAILED_TASKS.md**
   - Task 1.1.3: Rename and scope ExternalAPIMetricRegistry
   - NEW Task 1.1.7: Implement gRPCMetricRegistry
   - NEW Task 1.1.8: Implement gRPC interceptor (optional)
   - NEW Task 1.3.6: Migrate gRPC call sites
   - Update all references to HTTP registry

5. **specs/observability/spec.md**
   - Update requirement to list 6 registries
   - Add scenario for gRPC internal communication
   - Clarify external API vs internal service distinction

6. **README.md**
   - Update metric registry list (6 items)
   - Add gRPC internal communication to benefits
   - Update deliverables count

### No Code Changes Required (Already gRPC)

✅ All internal services already use gRPC correctly
✅ GPU services have proto definitions
✅ No HTTP internal communication found

**This change is purely about metric collection organization, not changing communication protocols.**

---

## Action Items

### High Priority (Before Phase 1 Implementation)

1. **Update All Documentation** (2-3 hours)
   - Rename HTTPMetricRegistry → ExternalAPIMetricRegistry
   - Add gRPCMetricRegistry throughout
   - Update all task descriptions and acceptance criteria

2. **Validate No Internal HTTP** (1 hour)
   - Audit codebase for any internal HTTP calls
   - Confirm all Docker services use gRPC
   - Document any exceptions

3. **Add gRPC Interceptor Task** (30 minutes)
   - Add to DETAILED_TASKS.md as optional but recommended
   - Specify implementation details
   - Add acceptance criteria

### Medium Priority

4. **Review External API Boundary** (1 hour)
   - Confirm which HTTP traffic is external vs internal
   - Document adapter HTTP client usage
   - Validate external database protocols

---

## Conclusion

The architecture is CORRECT - all internal services use gRPC. The change proposal only needed clarification that:

1. "HTTPMetricRegistry" should be "ExternalAPIMetricRegistry" (external traffic only)
2. A new "gRPCMetricRegistry" is needed for internal service communication
3. Total of 6 domain-specific registries (not 5)

**No protocol changes required - only metric collection organization.**
