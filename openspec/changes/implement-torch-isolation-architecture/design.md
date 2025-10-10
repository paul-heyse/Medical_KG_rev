## Context

This change implements a comprehensive torch isolation architecture that eliminates torch dependencies from the main API gateway while preserving GPU-accelerated processing capabilities through isolated Docker services. The new architecture:

1. **Torch-Free Main Gateway**: Complete removal of torch from core API gateway for simplified deployment
2. **Isolated GPU Services**: GPU-accelerated services moved to dedicated Docker containers
3. **Docling Integration**: Leverage Docling's superior chunking capabilities instead of torch-based semantic chunking
4. **Service-Oriented Architecture**: HTTP API-based communication between torch-free core and GPU services
5. **Operational Flexibility**: Deploy main gateway without GPU infrastructure when not needed

The system maintains backward compatibility while providing superior document understanding and retrieval accuracy through the new service architecture.

## Goals / Non-Goals

### Goals

- Eliminate all torch dependencies from main API gateway codebase
- Create isolated GPU services in Docker containers for GPU-accelerated processing
- Maintain equivalent functionality and performance through service architecture
- Implement robust service discovery, health monitoring, and circuit breaker patterns
- Achieve deployment flexibility with torch-free main gateway option
- Preserve Docling's superior chunking capabilities for document processing
- Maintain production-ready monitoring, security, and compliance

### Non-Goals

- Redesign the entire document processing pipeline architecture
- Change external API interfaces (maintain compatibility)
- Remove GPU processing capabilities (move to isolated services)
- Implement real-time service communication (batch processing acceptable)
- Create complex service mesh or orchestration frameworks

## Decisions

### 1. Torch-Free Main Gateway Architecture

**Decision**: Implement torch-free main API gateway that communicates with GPU services via gRPC APIs (aligning with project standards for GPU service communication).

**Implementation Details**:

```python
# Torch-free main gateway gRPC service client
import grpc
from Medical_KG_rev.proto import gpu_pb2, gpu_pb2_grpc

class GPUClient:
    def __init__(self, service_address: str = "gpu-services:50051"):
        self.channel = grpc.aio.insecure_channel(service_address)
        self.stub = gpu_pb2_grpc.GPUServiceStub(self.channel)

    async def get_status(self) -> gpu_pb2.StatusResponse:
        # gRPC call instead of direct torch usage
        request = gpu_pb2.StatusRequest()
        response = await self.stub.GetStatus(request)
        return response

    async def allocate_gpu(self, memory_mb: int) -> gpu_pb2.AllocationResponse:
        request = gpu_pb2.AllocationRequest(memory_mb=memory_mb)
        response = await self.stub.AllocateGPU(request)
        return response

# Before: Direct torch usage
gpu_stats = torch.cuda.get_device_properties(0)

# After: gRPC calls
gpu_client = GPUClient()
gpu_stats = await gpu_client.get_status()
```

**Rationale**: gRPC communication enables clean separation of torch dependencies while maintaining equivalent functionality. gRPC is preferred over HTTP/REST for GPU services per project standards (efficient binary serialization, type safety, streaming support).

**Alternatives Considered**:

- HTTP/REST API: Violates project standard ("Use gRPC for inter-service communication (not REST)")
- Direct torch usage in main gateway: Creates deployment complexity
- Complete rewrite without GPU support: Loses GPU acceleration benefits
- Complex IPC mechanisms: Increases operational complexity

### 2. GPU Services Docker Container Architecture

**Decision**: Create dedicated GPU services Docker container with full torch ecosystem for GPU-accelerated processing, exposing gRPC API.

**Implementation Details**:

```dockerfile
# GPU services Docker container with torch
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Install full torch ecosystem
RUN pip install torch torchvision transformers accelerate sentence-transformers

# Copy GPU service code and proto definitions
COPY src/Medical_KG_rev/services/gpu/ /app/services/gpu/
COPY src/Medical_KG_rev/embeddings/utils/gpu.py /app/embeddings/utils/gpu.py
COPY src/Medical_KG_rev/services/vector_store/gpu.py /app/services/vector_store/gpu.py
COPY src/Medical_KG_rev/proto/ /app/proto/

# Health check using gRPC health protocol
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD grpc_health_probe -addr=:50051

# Expose gRPC port
EXPOSE 50051

CMD ["python", "-m", "gpu_services.grpc_server"]
```

**Rationale**: Dedicated GPU container isolates torch dependencies while providing gRPC API for GPU operations. gRPC health protocol provides standardized health checking.

**Alternatives Considered**:

- HTTP/REST API: Violates project standard for GPU service communication
- Multiple smaller containers: Increases orchestration complexity
- Single container with all services: Defeats isolation benefits
- No Docker isolation: Maintains torch dependencies in main codebase

### 3. Service Client Architecture with Circuit Breakers

**Decision**: Implement service client architecture with circuit breaker patterns for resilient gRPC service communication. Circuit breakers apply to **network/communication failures only**, not GPU unavailability (which must fail-fast).

**Implementation Details**:

```python
import grpc
from opentelemetry import trace

class ServiceClient:
    """gRPC service client with circuit breaker for network failures only.

    Note: Circuit breakers protect against network/communication issues.
    GPU unavailability must fail-fast without circuit breaker intervention.
    """

    def __init__(self, service_address: str, circuit_breaker: CircuitBreaker):
        self.service_address = service_address
        self.circuit_breaker = circuit_breaker
        self.channel = None
        self.tracer = trace.get_tracer(__name__)

    async def call_service(self, rpc_method, request):
        # Circuit breaker check for NETWORK failures only
        if not await self.circuit_breaker.can_execute():
            raise ServiceUnavailableError(
                f"Service {self.service_address} unavailable (network issues)"
            )

        try:
            # Make gRPC request to Docker service with tracing
            with self.tracer.start_as_current_span("grpc_call") as span:
                span.set_attribute("service", self.service_address)
                response = await rpc_method(request)
                await self.circuit_breaker.record_success()
                return response
        except grpc.RpcError as e:
            # Classify error type
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                # Network issue - record for circuit breaker
                await self.circuit_breaker.record_failure()
                raise ServiceError(f"Service communication failed: {e}")
            elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                # GPU unavailable - fail fast, no circuit breaker
                raise GpuNotAvailableError(f"GPU unavailable: {e.details()}")
            else:
                raise ServiceError(f"Service call failed: {e}")

# Circuit breaker configuration for network failures
class CircuitBreaker:
    """Protects against network/communication failures, not GPU unavailability."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 180):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
```

**Rationale**: Circuit breaker pattern provides resilient service communication with automatic failure detection and recovery for **network issues**. GPU unavailability bypasses circuit breaker for fail-fast behavior.

**Alternatives Considered**:

- No circuit breaker: Risk of cascading failures from network issues
- Circuit breaker for all errors including GPU: Violates fail-fast philosophy
- Simple retry logic: Insufficient for persistent service issues
- Complex service mesh: Overkill for this use case

### 4. Docling Integration for Torch-Free Chunking

**Decision**: Leverage Docling's built-in chunking capabilities to eliminate torch-based semantic chunking.

**Prerequisite**: This change assumes Docling VLM integration is complete (see `replace-mineru-with-docling-vlm` change proposal). Docling must be operational before torch isolation can proceed.

**Implementation Details**:

```python
# Docling provides superior chunking without torch
# Note: Docling itself is already integrated via replace-mineru-with-docling-vlm
class DoclingChunker:
    """Chunker that uses Docling's output without requiring torch in main gateway.

    Docling dependency (docling[vlm]>=2.0.0) is already present and runs in
    its own GPU-enabled container. This class consumes Docling's output.
    """

    def __init__(self, docling_output_store):
        self.docling_output_store = docling_output_store

    async def chunk_document(self, document_id: str) -> List[Chunk]:
        # Use Docling's pre-processed output (already GPU-accelerated)
        docling_result = await self.docling_output_store.get_result(document_id)

        # Docling provides structured chunks with semantic understanding
        chunks = []
        for element in docling_result.body.body:
            if element.label in ['TITLE', 'SECTION_HEADER', 'PARAGRAPH']:
                chunk = Chunk(
                    id=f"{document_id}-{element.id}",
                    text=element.text,
                    metadata={
                        "source": "docling",
                        "element_type": element.label,
                        "page": element.page_no,
                        "bbox": element.bbox,
                    }
                )
                chunks.append(chunk)
        return chunks

# Before: Torch-based semantic chunking in main gateway
semantic_chunker = SemanticChunker(gpu_model="bert-base-uncased")  # Requires torch
chunks = await semantic_chunker.chunk(document)

# After: Docling provides better chunking without torch in main gateway
docling_chunker = DoclingChunker(docling_output_store)  # No torch required
chunks = await docling_chunker.chunk_document(document_id)
```

**Rationale**: Docling provides superior chunking capabilities that eliminate the need for torch-based semantic chunking in the main codebase. Docling runs in its own GPU container (from `replace-mineru-with-docling-vlm` change).

**Alternatives Considered**:

- Keep torch-based chunking: Requires torch dependencies in main gateway
- Simple text splitting: Insufficient for medical document structure
- External chunking service: Already provided by Docling integration

### 5. Service Discovery and Load Balancing

**Decision**: Implement service registry with gRPC health checking and load balancing for GPU services.

**Implementation Details**:

```python
import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

class ServiceRegistry:
    def __init__(self, config: ServiceRegistryConfig):
        self.config = config
        self.service_endpoints = {}  # service_name -> list of healthy endpoints

    async def get_service_endpoint(self, service_name: str) -> str:
        # Health check all registered endpoints using gRPC health protocol
        healthy_endpoints = []
        for endpoint in self.service_endpoints.get(service_name, []):
            if await self._is_endpoint_healthy(endpoint):
                healthy_endpoints.append(endpoint)

        if not healthy_endpoints:
            # Service discovery for new endpoints
            await self._discover_service_endpoints(service_name)
            healthy_endpoints = self.service_endpoints.get(service_name, [])

        # Load balancing: round-robin across healthy endpoints
        return healthy_endpoints[0] if healthy_endpoints else self._get_default_endpoint(service_name)

    async def _is_endpoint_healthy(self, endpoint: str) -> bool:
        try:
            # Use gRPC health check protocol
            channel = grpc.aio.insecure_channel(endpoint)
            stub = health_pb2_grpc.HealthStub(channel)
            request = health_pb2.HealthCheckRequest(service="")
            response = await stub.Check(request, timeout=5)
            await channel.close()
            return response.status == health_pb2.HealthCheckResponse.SERVING
        except Exception:
            return False

    async def _discover_service_endpoints(self, service_name: str) -> None:
        # Service discovery logic (Docker DNS, Kubernetes service discovery)
        if service_name == "gpu-services":
            # Docker Compose: gpu-services:50051
            # Kubernetes: gpu-services.default.svc.cluster.local:50051
            endpoints = await self._discover_gpu_services()
        elif service_name == "embedding-services":
            endpoints = await self._discover_embedding_services()
        elif service_name == "reranking-services":
            endpoints = await self._discover_reranking_services()

        self.service_endpoints[service_name] = endpoints
```

**Rationale**: Service discovery with gRPC health protocol enables dynamic service location and standardized health checking across multiple service instances.

**Alternatives Considered**:

- Hard-coded service URLs: No scalability or failover
- HTTP health checks: Less efficient than gRPC health protocol
- External service mesh: Overkill for this use case
- No service discovery: Manual service management required

### 6. Service-to-Service Authentication

**Decision**: Implement mutual TLS (mTLS) for gRPC service-to-service authentication.

**Implementation Details**:

```python
import grpc
from grpc import ssl_channel_credentials, ssl_server_credentials

class AuthenticatedServiceClient:
    """gRPC client with mTLS authentication."""

    def __init__(self, service_address: str, cert_dir: str):
        self.service_address = service_address

        # Load client credentials
        with open(f"{cert_dir}/ca.crt", "rb") as f:
            ca_cert = f.read()
        with open(f"{cert_dir}/client.crt", "rb") as f:
            client_cert = f.read()
        with open(f"{cert_dir}/client.key", "rb") as f:
            client_key = f.read()

        # Create SSL credentials
        credentials = ssl_channel_credentials(
            root_certificates=ca_cert,
            private_key=client_key,
            certificate_chain=client_cert
        )

        # Create secure channel
        self.channel = grpc.aio.secure_channel(
            service_address,
            credentials,
            options=[
                ('grpc.ssl_target_name_override', 'gpu-services'),
            ]
        )

# GPU service server with mTLS
class GPUServiceServer:
    def __init__(self, cert_dir: str):
        # Load server credentials
        with open(f"{cert_dir}/ca.crt", "rb") as f:
            ca_cert = f.read()
        with open(f"{cert_dir}/server.crt", "rb") as f:
            server_cert = f.read()
        with open(f"{cert_dir}/server.key", "rb") as f:
            server_key = f.read()

        # Create SSL server credentials
        server_credentials = ssl_server_credentials(
            [(server_key, server_cert)],
            root_certificates=ca_cert,
            require_client_auth=True  # mTLS
        )

        self.server = grpc.aio.server()
        self.server.add_secure_port('[::]:50051', server_credentials)
```

**Rationale**: mTLS provides strong mutual authentication between services without requiring additional authentication infrastructure. Both client and server verify each other's identity.

**Alternatives Considered**:

- JWT tokens: Requires token service infrastructure
- API keys: Less secure, harder to rotate
- No authentication: Security risk for internal services
- Service mesh with built-in auth: Overkill for current scale

### 7. Dagster Integration

**Decision**: Integrate GPU services with Dagster pipeline orchestration using gRPC service clients in Dagster assets.

**Implementation Details**:

```python
from dagster import asset, OpExecutionContext
from Medical_KG_rev.services.clients import GPUClient, EmbeddingClient

@asset(
    compute_kind="gpu",
    required_resource_keys={"gpu_service"},
)
async def generate_embeddings(
    context: OpExecutionContext,
    chunked_documents: List[ChunkedDocument],
) -> List[EmbeddedDocument]:
    """Generate embeddings using GPU service via gRPC.

    This asset calls the embedding service which runs in a separate
    torch-enabled Docker container. The main Dagster worker remains
    torch-free.
    """

    # Get gRPC client from Dagster resources
    embedding_client: EmbeddingClient = context.resources.gpu_service.embedding_client

    embedded_docs = []
    for doc in chunked_documents:
        try:
            # Call embedding service via gRPC
            embeddings = await embedding_client.generate_embeddings(
                texts=[chunk.text for chunk in doc.chunks],
                model="Qwen/Qwen3-Embedding-8B",
            )

            embedded_doc = EmbeddedDocument(
                document_id=doc.document_id,
                embeddings=embeddings,
            )
            embedded_docs.append(embedded_doc)

        except GpuNotAvailableError as e:
            # GPU unavailable - fail fast, don't continue
            context.log.error(f"GPU unavailable for document {doc.document_id}: {e}")
            raise  # Fail the entire Dagster run

        except ServiceError as e:
            # Network issue - circuit breaker will handle
            context.log.warning(f"Service communication failed: {e}")
            # Retry or handle based on circuit breaker state

    return embedded_docs

# Dagster resource configuration
from dagster import resource

@resource(config_schema={
    "gpu_service_address": str,
    "embedding_service_address": str,
    "reranking_service_address": str,
})
def gpu_service_resource(init_context):
    """Dagster resource providing gRPC clients for GPU services."""
    config = init_context.resource_config

    return GPUServiceResource(
        gpu_client=GPUClient(config["gpu_service_address"]),
        embedding_client=EmbeddingClient(config["embedding_service_address"]),
        reranking_client=RerankingClient(config["reranking_service_address"]),
    )

# Dagster job configuration
from dagster import job

@job(
    resource_defs={
        "gpu_service": gpu_service_resource,
    },
    config={
        "resources": {
            "gpu_service": {
                "config": {
                    "gpu_service_address": "gpu-services:50051",
                    "embedding_service_address": "embedding-services:50051",
                    "reranking_service_address": "reranking-services:50051",
                }
            }
        }
    }
)
def document_processing_pipeline():
    """Two-phase pipeline using GPU services via gRPC."""
    # Auto-pipeline: metadata → chunk → embed → index
    # Manual pipeline: metadata → PDF → Docling → chunk → embed → index
    embedded_docs = generate_embeddings(chunked_documents)
    # ... rest of pipeline
```

**Rationale**: Dagster assets access GPU services via gRPC clients configured as Dagster resources. This keeps Dagster workers torch-free while enabling GPU-accelerated operations in the pipeline. Fail-fast semantics are preserved - GPU unavailability fails the entire Dagster run.

**Alternatives Considered**:

- Direct torch usage in Dagster workers: Requires torch dependencies
- External job queue for GPU operations: Adds complexity, breaks pipeline semantics
- Synchronous blocking calls: Doesn't leverage Dagster's async capabilities
- No Dagster integration: Defeats purpose of orchestration layer

## Risks / Trade-offs

### Risk: Service Communication Latency

**Risk**: gRPC calls to Docker services may introduce latency compared to direct torch usage.

**SLO**: Service call overhead < 10ms P95, total embedding generation latency < 500ms P95 (per project requirements).

**Mitigation**:

- Implement connection pooling and HTTP/2 multiplexing (built into gRPC)
- Add request batching for efficient service calls
- Implement caching for repeated service requests
- Add service call timeout optimization (default: 30s for embedding, 60s for reranking)
- Monitor service response times with Prometheus and optimize as needed
- Use gRPC streaming for large batch operations
- Enable gRPC keepalive to reduce connection overhead

### Risk: Service Dependency Management

**Risk**: Main gateway becomes dependent on external GPU services for core functionality.

**Mitigation**:

- Implement comprehensive health monitoring and alerting
- Add circuit breaker patterns for graceful degradation
- Provide fallback mechanisms for service failures
- Include service dependency health checks in deployment validation

### Risk: Docker Service Complexity

**Risk**: Additional Docker services increase operational complexity and resource requirements.

**Mitigation**:

- Implement automated deployment and scaling procedures
- Add comprehensive monitoring and alerting for all services
- Create operational runbooks for service management
- Include service health checks in CI/CD pipelines

### Trade-off: Performance vs Deployment Simplicity

**Trade-off**: gRPC communication may have slight performance overhead vs direct torch usage (estimated 5-15ms P95).

**Decision**: Accept minor performance overhead for significant deployment and maintenance benefits of torch-free main gateway. gRPC is more efficient than HTTP/REST for this use case.

### Trade-off: Service Architecture Complexity vs Operational Benefits

**Trade-off**: Service-oriented architecture increases complexity compared to monolithic design.

**Decision**: Accept architectural complexity for operational benefits including deployment flexibility, resource isolation, and maintenance simplicity.

## Migration Plan

### Phase 1: Torch-Free Chunking Implementation (Week 1)

1. Remove torch-based semantic checks from chunking modules
2. Integrate Docling's built-in chunking capabilities
3. Update chunking interfaces to use Docling output
4. Update chunking tests for torch-free operation
5. Validate chunking functionality matches or exceeds previous torch-based chunking

### Phase 2: GPU Services Docker Implementation (Week 2-3)

1. Create GPU services Docker container with torch ecosystem
2. Move GPU manager, embedding GPU utils, vector store GPU to Docker service
3. Implement HTTP API wrappers for GPU operations
4. Add service discovery and health monitoring
5. Update main gateway to use HTTP API calls instead of direct torch usage

### Phase 3: Embedding and Reranking Services (Week 4-5)

1. Create embedding services Docker container
2. Move embedding functionality to Docker service with HTTP API
3. Create reranking services Docker container
4. Move reranking pipeline to Docker service with HTTP API
5. Implement service client architecture with circuit breakers

### Phase 4: Torch-Free Main Gateway Deployment (Week 6)

1. Create torch-free main gateway Docker image
2. Remove all torch imports from main gateway code
3. Update configuration for service endpoints
4. Implement comprehensive service integration testing
5. Deploy torch-free main gateway alongside GPU services

### Phase 5: Validation and Optimization (Week 7)

1. Performance testing to ensure equivalent functionality
2. Load testing for service communication under high load
3. Security and compliance validation for service architecture
4. Operational procedures testing and documentation
5. Final deployment validation and monitoring setup

## Open Questions

1. **Service Scaling Strategy**: How should GPU services scale? Individual service scaling or shared GPU resource pools?
   - Recommendation: Start with individual service scaling, consolidate if resource utilization is low

2. **Model Version Management**: How should we handle model updates across multiple Docker services?
   - Recommendation: Use shared volume mounts for models with version tags in image names

3. **Monitoring Granularity**: What level of monitoring detail is needed for individual Docker services vs overall system?
   - Recommendation: Service-level metrics (latency, error rate, GPU utilization) + end-to-end tracing

4. **Rollback Strategy**: What procedures should be in place for rolling back to direct torch usage if service architecture has issues?
   - Recommendation: Feature flag (`USE_GPU_SERVICES=true/false`) with compatibility layer for gradual migration

5. **Service Consolidation**: Should embedding and reranking services be consolidated into single "ML Services" container?
   - Recommendation: Start separate for clear resource isolation, consolidate if operational overhead is high

6. **Multi-Tenancy in GPU Services**: How should GPU services handle tenant context propagation?
   - Recommendation: Include tenant_id in gRPC metadata for all service calls
