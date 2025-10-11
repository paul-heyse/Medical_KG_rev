# Service Architecture API Documentation

This document provides comprehensive API documentation for the GPU services architecture, including gRPC service endpoints, client usage examples, and best practices.

## Table of Contents

1. [Overview](#overview)
2. [GPU Management Service](#gpu-management-service)
3. [Embedding Service](#embedding-service)
4. [Reranking Service](#reranking-service)
5. [Docling VLM Service](#docling-vlm-service)
6. [Service Clients](#service-clients)
7. [Error Handling](#error-handling)
8. [Circuit Breaker Patterns](#circuit-breaker-patterns)
9. [Health Checks](#health-checks)
10. [Best Practices](#best-practices)

## Overview

The GPU services architecture isolates all PyTorch dependencies in dedicated Docker containers, communicating with the main gateway via gRPC. This provides:

- **Resource isolation**: GPU memory and compute resources are isolated per service
- **Independent scaling**: Services can scale independently based on workload
- **Fail-fast behavior**: Services fail immediately if GPU unavailable (no CPU fallback)
- **Simplified deployment**: Main gateway can run on CPU-only infrastructure

### Service Endpoints

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| GPU Management | 50051 | gRPC | GPU resource allocation and monitoring |
| Embedding | 50052 | gRPC | Transformer-based embedding generation |
| Reranking | 50053 | gRPC | Cross-encoder reranking models |
| Docling VLM | 50054 | gRPC | Document processing using vision-language models |

## GPU Management Service

### Service Definition

```protobuf
syntax = "proto3";

package medical_kg_rev.v1;

service GPUService {
  rpc GetStatus(GPUStatusRequest) returns (GPUStatusResponse);
  rpc ListDevices(ListDevicesRequest) returns (ListDevicesResponse);
  rpc AllocateGPU(AllocateGPURequest) returns (AllocateGPUResponse);
  rpc DeallocateGPU(DeallocateGPURequest) returns (DeallocateGPUResponse);
}

message GPUStatusRequest {}

message GPUStatusResponse {
  string status = 1;
  int32 available_gpus = 2;
  repeated GPUInfo gpus = 3;
}

message GPUInfo {
  int32 device_id = 1;
  string name = 2;
  int64 memory_total_mb = 3;
  int64 memory_used_mb = 4;
  float utilization_percent = 5;
  float temperature_celsius = 6;
  string status = 7;
}

message ListDevicesRequest {}

message ListDevicesResponse {
  repeated GPUDevice devices = 1;
}

message GPUDevice {
  int32 device_id = 1;
  string name = 2;
  int64 memory_total_mb = 3;
  string compute_capability = 4;
  bool available = 5;
}

message AllocateGPURequest {
  int32 device_id = 1;
  int64 memory_requested_mb = 2;
  string tenant_id = 3;
}

message AllocateGPUResponse {
  bool success = 1;
  string allocation_id = 2;
  GPUAllocation allocation = 3;
}

message GPUAllocation {
  string allocation_id = 1;
  int32 device_id = 2;
  int64 memory_allocated_mb = 3;
  string tenant_id = 4;
  int64 allocated_at = 5;
  int64 expires_at = 6;
}

message DeallocateGPURequest {
  string allocation_id = 1;
}

message DeallocateGPUResponse {
  bool success = 1;
}
```

### Client Usage

```python
import grpc
from Medical_KG_rev.proto import gpu_service_pb2_grpc, gpu_service_pb2
from Medical_KG_rev.services.clients.gpu_client import GPUClient

# Initialize client
gpu_client = GPUClient(service_url="localhost:50051")

# Get GPU status
status = await gpu_client.get_status()
print(f"Available GPUs: {status.available_gpus}")
for gpu in status.gpus:
    print(f"GPU {gpu.device_id}: {gpu.memory_used_mb}/{gpu.memory_total_mb} MB")

# List devices
devices = await gpu_client.list_devices()
for device in devices.devices:
    print(f"Device {device.device_id}: {device.name} ({device.memory_total_mb} MB)")

# Allocate GPU
allocation = await gpu_client.allocate_gpu(
    device_id=0,
    memory_requested_mb=4096,
    tenant_id="user123"
)
print(f"Allocation ID: {allocation.allocation_id}")

# Deallocate GPU
success = await gpu_client.deallocate_gpu(allocation.allocation_id)
print(f"Deallocation successful: {success}")
```

### Error Codes

| Code | Description | Action |
|------|-------------|--------|
| `UNAVAILABLE` | GPU service unavailable | Retry with backoff |
| `RESOURCE_EXHAUSTED` | No GPUs available | Wait and retry |
| `INVALID_ARGUMENT` | Invalid device ID or memory request | Fix request parameters |
| `PERMISSION_DENIED` | Insufficient permissions | Check tenant access |

## Embedding Service

### Service Definition

```protobuf
syntax = "proto3";

package medical_kg_rev.v1;

service EmbeddingService {
  rpc GenerateEmbeddings(GenerateEmbeddingsRequest) returns (GenerateEmbeddingsResponse);
  rpc GenerateEmbeddingsBatch(GenerateEmbeddingsBatchRequest) returns (GenerateEmbeddingsBatchResponse);
  rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
  rpc GetHealth(HealthRequest) returns (HealthResponse);
  rpc GetStats(StatsRequest) returns (StatsResponse);
}

message GenerateEmbeddingsRequest {
  repeated string texts = 1;
  string model_name = 2;
  EmbeddingConfig config = 3;
}

message GenerateEmbeddingsResponse {
  repeated EmbeddingResult embeddings = 1;
  ProcessingMetadata metadata = 2;
}

message EmbeddingResult {
  repeated float values = 1;
  int32 dimension = 2;
  float norm = 3;
}

message EmbeddingConfig {
  bool normalize = 1;
  int32 batch_size = 2;
  float timeout_seconds = 3;
}

message GenerateEmbeddingsBatchRequest {
  repeated GenerateEmbeddingsRequest requests = 1;
}

message GenerateEmbeddingsBatchResponse {
  repeated GenerateEmbeddingsResponse responses = 1;
  ProcessingMetadata metadata = 2;
}

message ListModelsRequest {}

message ListModelsResponse {
  repeated EmbeddingModelInfo models = 1;
}

message EmbeddingModelInfo {
  string name = 1;
  int32 dimension = 2;
  string description = 3;
  bool available = 4;
}

message ProcessingMetadata {
  string model_name = 1;
  float processing_time_ms = 2;
  int64 gpu_memory_used_mb = 3;
  int32 batch_size = 4;
  int32 total_requests = 5;
}
```

### Client Usage

```python
import grpc
from Medical_KG_rev.proto import embedding_service_pb2_grpc, embedding_service_pb2
from Medical_KG_rev.services.clients.embedding_client import EmbeddingClient

# Initialize client
embedding_client = EmbeddingClient(service_url="localhost:50052")

# Generate embeddings for single request
embeddings = await embedding_client.generate_embeddings(
    texts=["This is a test document", "Another test document"],
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    config=embedding_service_pb2.EmbeddingConfig(
        normalize=True,
        batch_size=32,
        timeout_seconds=30.0
    )
)

print(f"Generated {len(embeddings.embeddings)} embeddings")
for i, embedding in enumerate(embeddings.embeddings):
    print(f"Embedding {i}: dimension={embedding.dimension}, norm={embedding.norm}")

# Generate embeddings for batch request
batch_request = embedding_service_pb2.GenerateEmbeddingsBatchRequest(
    requests=[
        embedding_service_pb2.GenerateEmbeddingsRequest(
            texts=["Document 1", "Document 2"],
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
        embedding_service_pb2.GenerateEmbeddingsRequest(
            texts=["Document 3", "Document 4"],
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    ]
)

batch_response = await embedding_client.generate_embeddings_batch(batch_request)
print(f"Batch processing completed: {len(batch_response.responses)} responses")

# List available models
models = await embedding_client.list_models()
for model in models.models:
    print(f"Model: {model.name}, Dimension: {model.dimension}, Available: {model.available}")
```

### Error Codes

| Code | Description | Action |
|------|-------------|--------|
| `UNAVAILABLE` | Embedding service unavailable | Retry with backoff |
| `INVALID_ARGUMENT` | Invalid model name or text | Fix request parameters |
| `RESOURCE_EXHAUSTED` | GPU memory exhausted | Reduce batch size or wait |
| `DEADLINE_EXCEEDED` | Request timeout | Increase timeout or retry |

## Reranking Service

### Service Definition

```protobuf
syntax = "proto3";

package medical_kg_rev.v1;

service RerankingService {
  rpc RerankBatch(RerankBatchRequest) returns (RerankBatchResponse);
  rpc RerankMultipleBatches(RerankMultipleBatchesRequest) returns (RerankMultipleBatchesResponse);
  rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
  rpc GetHealth(HealthRequest) returns (HealthResponse);
  rpc GetStats(StatsRequest) returns (StatsResponse);
}

message RerankBatchRequest {
  string query = 1;
  repeated string documents = 2;
  RerankingConfig config = 3;
}

message RerankBatchResponse {
  repeated RerankingResult results = 1;
  ProcessingMetadata metadata = 2;
}

message RerankingResult {
  int32 document_index = 1;
  float score = 2;
  string document = 3;
}

message RerankingConfig {
  string model_name = 1;
  int32 batch_size = 2;
  float timeout_seconds = 3;
  bool return_documents = 4;
}

message RerankMultipleBatchesRequest {
  repeated RerankBatchRequest requests = 1;
}

message RerankMultipleBatchesResponse {
  repeated RerankBatchResponse responses = 1;
  ProcessingMetadata metadata = 2;
}

message RerankingModelInfo {
  string name = 1;
  string description = 2;
  bool available = 3;
  int32 max_sequence_length = 4;
}
```

### Client Usage

```python
import grpc
from Medical_KG_rev.proto import reranking_service_pb2_grpc, reranking_service_pb2
from Medical_KG_rev.services.clients.reranking_client import RerankingClient

# Initialize client
reranking_client = RerankingClient(service_url="localhost:50053")

# Rerank documents
query = "diabetes treatment"
documents = [
    "Metformin is commonly used to treat type 2 diabetes",
    "Insulin therapy is essential for type 1 diabetes",
    "Exercise and diet are important for diabetes management"
]

results = await reranking_client.rerank_batch(
    query=query,
    documents=documents,
    config=reranking_service_pb2.RerankingConfig(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=16,
        timeout_seconds=30.0,
        return_documents=True
    )
)

print(f"Reranking completed for {len(results.results)} documents")
for result in results.results:
    print(f"Document {result.document_index}: score={result.score:.4f}")
    print(f"Text: {result.document[:100]}...")

# Rerank multiple batches
batch_requests = [
    reranking_service_pb2.RerankBatchRequest(
        query="diabetes treatment",
        documents=["doc1", "doc2", "doc3"]
    ),
    reranking_service_pb2.RerankBatchRequest(
        query="hypertension management",
        documents=["doc4", "doc5", "doc6"]
    )
]

batch_response = await reranking_client.rerank_multiple_batches(batch_requests)
print(f"Multiple batch reranking completed: {len(batch_response.responses)} responses")

# List available models
models = await reranking_client.list_models()
for model in models.models:
    print(f"Model: {model.name}, Max Length: {model.max_sequence_length}, Available: {model.available}")
```

### Error Codes

| Code | Description | Action |
|------|-------------|--------|
| `UNAVAILABLE` | Reranking service unavailable | Retry with backoff |
| `INVALID_ARGUMENT` | Invalid model name or documents | Fix request parameters |
| `RESOURCE_EXHAUSTED` | GPU memory exhausted | Reduce batch size or wait |
| `DEADLINE_EXCEEDED` | Request timeout | Increase timeout or retry |

## Docling VLM Service

### Service Definition

```protobuf
syntax = "proto3";

package medical_kg_rev.v1;

service DoclingVLMService {
  rpc ProcessPDF(ProcessPDFRequest) returns (ProcessPDFResponse);
  rpc ProcessPDFBatch(ProcessPDFBatchRequest) returns (ProcessPDFBatchResponse);
  rpc GetHealth(HealthRequest) returns (HealthResponse);
  rpc GetStats(StatsRequest) returns (StatsResponse);
}

message ProcessPDFRequest {
  bytes pdf_content = 1;
  string filename = 2;
  DoclingConfig config = 3;
}

message ProcessPDFResponse {
  DocTagsResult result = 1;
  ProcessingMetadata metadata = 2;
}

message DocTagsResult {
  DocumentStructure document = 1;
  repeated Table tables = 2;
  repeated Figure figures = 3;
  repeated TextBlock text_blocks = 4;
  DocumentMetadata metadata = 5;
}

message DocumentStructure {
  string title = 1;
  repeated Section sections = 2;
  repeated Paragraph paragraphs = 3;
}

message Section {
  string title = 1;
  int32 level = 2;
  repeated Paragraph paragraphs = 3;
  repeated Table tables = 4;
  repeated Figure figures = 5;
}

message Paragraph {
  string text = 1;
  int32 page_number = 2;
  BoundingBox bbox = 3;
}

message Table {
  string caption = 1;
  repeated TableRow rows = 2;
  int32 page_number = 3;
  BoundingBox bbox = 4;
}

message TableRow {
  repeated TableCell cells = 1;
}

message TableCell {
  string text = 1;
  int32 row_span = 2;
  int32 col_span = 3;
}

message Figure {
  string caption = 1;
  string description = 2;
  int32 page_number = 3;
  BoundingBox bbox = 4;
}

message TextBlock {
  string text = 1;
  string type = 2;
  int32 page_number = 3;
  BoundingBox bbox = 4;
}

message BoundingBox {
  float x = 1;
  float y = 2;
  float width = 3;
  float height = 4;
}

message DocumentMetadata {
  string filename = 1;
  int32 page_count = 2;
  string processing_model = 3;
  int64 processing_time_ms = 4;
  string checksum = 5;
}

message DoclingConfig {
  string model_name = 1;
  int32 batch_size = 2;
  float timeout_seconds = 3;
  bool extract_tables = 4;
  bool extract_figures = 5;
  bool extract_text_blocks = 6;
}

message ProcessPDFBatchRequest {
  repeated ProcessPDFRequest requests = 1;
}

message ProcessPDFBatchResponse {
  repeated ProcessPDFResponse responses = 1;
  ProcessingMetadata metadata = 2;
}
```

### Client Usage

```python
import grpc
from Medical_KG_rev.proto import docling_vlm_service_pb2_grpc, docling_vlm_service_pb2
from Medical_KG_rev.services.clients.docling_vlm_client import DoclingVLMClient

# Initialize client
docling_client = DoclingVLMClient(service_url="localhost:50054")

# Process single PDF
with open("document.pdf", "rb") as f:
    pdf_content = f.read()

result = await docling_client.process_pdf(
    pdf_content=pdf_content,
    filename="document.pdf",
    config=docling_vlm_service_pb2.DoclingConfig(
        model_name="gemma3-12b",
        batch_size=1,
        timeout_seconds=300.0,
        extract_tables=True,
        extract_figures=True,
        extract_text_blocks=True
    )
)

print(f"Document processed: {result.result.metadata.filename}")
print(f"Pages: {result.result.metadata.page_count}")
print(f"Processing time: {result.result.metadata.processing_time_ms}ms")

# Process tables
for table in result.result.tables:
    print(f"Table on page {table.page_number}: {table.caption}")
    for row in table.rows:
        row_text = " | ".join([cell.text for cell in row.cells])
        print(f"  {row_text}")

# Process figures
for figure in result.result.figures:
    print(f"Figure on page {figure.page_number}: {figure.caption}")
    print(f"  Description: {figure.description}")

# Process text blocks
for text_block in result.result.text_blocks:
    print(f"Text block ({text_block.type}): {text_block.text[:100]}...")

# Process batch of PDFs
pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
batch_requests = []

for pdf_file in pdf_files:
    with open(pdf_file, "rb") as f:
        pdf_content = f.read()

    batch_requests.append(
        docling_vlm_service_pb2.ProcessPDFRequest(
            pdf_content=pdf_content,
            filename=pdf_file,
            config=docling_vlm_service_pb2.DoclingConfig(
                model_name="gemma3-12b",
                batch_size=3,
                timeout_seconds=600.0
            )
        )
    )

batch_response = await docling_client.process_pdf_batch(
    docling_vlm_service_pb2.ProcessPDFBatchRequest(requests=batch_requests)
)

print(f"Batch processing completed: {len(batch_response.responses)} responses")
for i, response in enumerate(batch_response.responses):
    print(f"PDF {i+1}: {response.result.metadata.filename} - {response.result.metadata.page_count} pages")
```

### Error Codes

| Code | Description | Action |
|------|-------------|--------|
| `UNAVAILABLE` | Docling VLM service unavailable | Retry with backoff |
| `INVALID_ARGUMENT` | Invalid PDF content or model name | Fix request parameters |
| `RESOURCE_EXHAUSTED` | GPU memory exhausted | Reduce batch size or wait |
| `DEADLINE_EXCEEDED` | Request timeout | Increase timeout or retry |
| `FAILED_PRECONDITION` | Model not loaded | Wait for model loading |

## Service Clients

### Base Client Interface

All service clients implement a common interface:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseServiceClient(ABC):
    """Base class for all service clients."""

    def __init__(self, service_url: str, timeout: float = 30.0):
        self.service_url = service_url
        self.timeout = timeout

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        pass
```

### Client Configuration

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ServiceClientConfig:
    """Configuration for service clients."""

    service_url: str
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    max_retry_delay: float = 60.0
    exponential_base: float = 2.0

    # Authentication
    use_tls: bool = False
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None
    tls_ca_path: Optional[str] = None

    # Headers
    tenant_id: Optional[str] = None
    correlation_id: Optional[str] = None
```

### Client Factory

```python
from Medical_KG_rev.services.clients.gpu_client import GPUClient
from Medical_KG_rev.services.clients.embedding_client import EmbeddingClient
from Medical_KG_rev.services.clients.reranking_client import RerankingClient
from Medical_KG_rev.services.clients.docling_vlm_client import DoclingVLMClient

class ServiceClientFactory:
    """Factory for creating service clients."""

    @staticmethod
    def create_gpu_client(config: ServiceClientConfig) -> GPUClient:
        """Create GPU service client."""
        return GPUClient(
            service_url=config.service_url,
            timeout=config.timeout
        )

    @staticmethod
    def create_embedding_client(config: ServiceClientConfig) -> EmbeddingClient:
        """Create embedding service client."""
        return EmbeddingClient(
            service_url=config.service_url,
            timeout=config.timeout
        )

    @staticmethod
    def create_reranking_client(config: ServiceClientConfig) -> RerankingClient:
        """Create reranking service client."""
        return RerankingClient(
            service_url=config.service_url,
            timeout=config.timeout
        )

    @staticmethod
    def create_docling_vlm_client(config: ServiceClientConfig) -> DoclingVLMClient:
        """Create Docling VLM service client."""
        return DoclingVLMClient(
            service_url=config.service_url,
            timeout=config.timeout
        )
```

## Error Handling

### Error Classification

```python
from enum import Enum
from typing import Dict, Any

class ErrorCategory(Enum):
    """Error categories for service communication."""

    NETWORK = "network"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    INVALID_ARGUMENT = "invalid_argument"
    PERMISSION_DENIED = "permission_denied"
    INTERNAL = "internal"
    UNAVAILABLE = "unavailable"

class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RetryStrategy(Enum):
    """Retry strategies for different error types."""

    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIXED = "fixed"
```

### Error Handler

```python
import grpc
import asyncio
import logging
from typing import Callable, Any, Dict
from Medical_KG_rev.services.clients.error_handler import ServiceErrorHandler

logger = logging.getLogger(__name__)

class ServiceErrorHandler:
    """Comprehensive error handling for service calls."""

    def __init__(self):
        self.error_classifications = {
            grpc.StatusCode.UNAVAILABLE: {
                "category": ErrorCategory.NETWORK,
                "severity": ErrorSeverity.HIGH,
                "retry_strategy": RetryStrategy.EXPONENTIAL
            },
            grpc.StatusCode.DEADLINE_EXCEEDED: {
                "category": ErrorCategory.TIMEOUT,
                "severity": ErrorSeverity.MEDIUM,
                "retry_strategy": RetryStrategy.LINEAR
            },
            grpc.StatusCode.RESOURCE_EXHAUSTED: {
                "category": ErrorCategory.RESOURCE_EXHAUSTED,
                "severity": ErrorSeverity.HIGH,
                "retry_strategy": RetryStrategy.FIXED
            },
            grpc.StatusCode.INVALID_ARGUMENT: {
                "category": ErrorCategory.INVALID_ARGUMENT,
                "severity": ErrorSeverity.LOW,
                "retry_strategy": RetryStrategy.NONE
            },
            grpc.StatusCode.PERMISSION_DENIED: {
                "category": ErrorCategory.PERMISSION_DENIED,
                "severity": ErrorSeverity.HIGH,
                "retry_strategy": RetryStrategy.NONE
            },
            grpc.StatusCode.INTERNAL: {
                "category": ErrorCategory.INTERNAL,
                "severity": ErrorSeverity.CRITICAL,
                "retry_strategy": RetryStrategy.EXPONENTIAL
            }
        }

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""

        last_exception = None

        for attempt in range(max_attempts):
            try:
                return await func(*args, **kwargs)
            except grpc.RpcError as e:
                last_exception = e
                error_info = self.error_classifications.get(
                    e.code(),
                    {
                        "category": ErrorCategory.INTERNAL,
                        "severity": ErrorSeverity.MEDIUM,
                        "retry_strategy": RetryStrategy.EXPONENTIAL
                    }
                )

                if error_info["retry_strategy"] == RetryStrategy.NONE:
                    logger.error(f"Non-retryable error: {e}")
                    raise

                if attempt == max_attempts - 1:
                    logger.error(f"Max retry attempts reached: {e}")
                    raise

                delay = self._calculate_delay(
                    attempt,
                    error_info["retry_strategy"],
                    base_delay,
                    max_delay,
                    exponential_base
                )

                logger.warning(
                    f"Retry attempt {attempt + 1}/{max_attempts} "
                    f"after {delay:.2f}s delay: {e}"
                )

                await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

        if last_exception:
            raise last_exception

    def _calculate_delay(
        self,
        attempt: int,
        strategy: RetryStrategy,
        base_delay: float,
        max_delay: float,
        exponential_base: float
    ) -> float:
        """Calculate delay for retry attempt."""

        if strategy == RetryStrategy.LINEAR:
            delay = base_delay * (attempt + 1)
        elif strategy == RetryStrategy.EXPONENTIAL:
            delay = base_delay * (exponential_base ** attempt)
        elif strategy == RetryStrategy.FIXED:
            delay = base_delay
        else:
            delay = base_delay

        return min(delay, max_delay)
```

## Circuit Breaker Patterns

### Circuit Breaker Implementation

```python
import asyncio
import time
from enum import Enum
from typing import Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker implementation for service calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""

        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True

        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker reset to CLOSED")

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self.state

    def get_failure_count(self) -> int:
        """Get current failure count."""
        return self.failure_count
```

### Circuit Breaker Usage

```python
from Medical_KG_rev.services.clients.circuit_breaker import CircuitBreaker
import grpc

# Initialize circuit breaker
circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exception=grpc.RpcError
)

# Use with service calls
async def safe_service_call():
    """Make service call with circuit breaker protection."""
    try:
        result = await circuit_breaker.call(
            embedding_client.generate_embeddings,
            texts=["test"],
            model_name="test-model"
        )
        return result
    except Exception as e:
        logger.error(f"Service call failed: {e}")
        raise

# Monitor circuit breaker state
def monitor_circuit_breaker():
    """Monitor circuit breaker state."""
    state = circuit_breaker.get_state()
    failure_count = circuit_breaker.get_failure_count()

    logger.info(f"Circuit breaker state: {state}, failures: {failure_count}")

    if state == CircuitState.OPEN:
        logger.warning("Circuit breaker is OPEN - service calls will fail")
    elif state == CircuitState.HALF_OPEN:
        logger.info("Circuit breaker is HALF_OPEN - testing service recovery")
```

## Health Checks

### Health Check Implementation

```python
import grpc
import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class HealthStatus:
    """Health status information."""

    service_name: str
    status: str
    timestamp: datetime
    response_time_ms: float
    details: Dict[str, Any]
    error: Optional[str] = None

class HealthChecker:
    """Health checker for GPU services."""

    def __init__(self, services: Dict[str, str]):
        self.services = services
        self.health_status: Dict[str, HealthStatus] = {}

    async def check_all_services(self) -> Dict[str, HealthStatus]:
        """Check health of all services."""
        tasks = []
        for service_name, service_url in self.services.items():
            task = asyncio.create_task(
                self._check_service_health(service_name, service_url)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            service_name = list(self.services.keys())[i]
            if isinstance(result, Exception):
                self.health_status[service_name] = HealthStatus(
                    service_name=service_name,
                    status="unhealthy",
                    timestamp=datetime.utcnow(),
                    response_time_ms=0.0,
                    details={},
                    error=str(result)
                )
            else:
                self.health_status[service_name] = result

        return self.health_status

    async def _check_service_health(
        self,
        service_name: str,
        service_url: str
    ) -> HealthStatus:
        """Check health of a single service."""
        start_time = time.time()

        try:
            async with grpc.aio.insecure_channel(service_url) as channel:
                # Use gRPC health check protocol
                from grpc_health.v1 import health_pb2_grpc, health_pb2

                stub = health_pb2_grpc.HealthStub(channel)
                request = health_pb2.HealthCheckRequest(service="")

                response = await stub.Check(request, timeout=5.0)

                response_time = (time.time() - start_time) * 1000

                return HealthStatus(
                    service_name=service_name,
                    status="healthy" if response.status == health_pb2.HealthCheckResponse.SERVING else "unhealthy",
                    timestamp=datetime.utcnow(),
                    response_time_ms=response_time,
                    details={
                        "grpc_status": response.status,
                        "service_url": service_url
                    }
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            return HealthStatus(
                service_name=service_name,
                status="unhealthy",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={"service_url": service_url},
                error=str(e)
            )

    def get_service_health(self, service_name: str) -> Optional[HealthStatus]:
        """Get health status for specific service."""
        return self.health_status.get(service_name)

    def is_service_healthy(self, service_name: str) -> bool:
        """Check if specific service is healthy."""
        status = self.get_service_health(service_name)
        return status is not None and status.status == "healthy"

    def get_unhealthy_services(self) -> List[str]:
        """Get list of unhealthy services."""
        return [
            service_name for service_name, status in self.health_status.items()
            if status.status != "healthy"
        ]
```

### Health Check Usage

```python
# Initialize health checker
health_checker = HealthChecker({
    "gpu-management": "localhost:50051",
    "embedding": "localhost:50052",
    "reranking": "localhost:50053",
    "docling-vlm": "localhost:50054"
})

# Check all services
health_status = await health_checker.check_all_services()

# Print health status
for service_name, status in health_status.items():
    print(f"{service_name}: {status.status} ({status.response_time_ms:.2f}ms)")
    if status.error:
        print(f"  Error: {status.error}")

# Check specific service
if health_checker.is_service_healthy("embedding"):
    print("Embedding service is healthy")
else:
    print("Embedding service is unhealthy")

# Get unhealthy services
unhealthy_services = health_checker.get_unhealthy_services()
if unhealthy_services:
    print(f"Unhealthy services: {unhealthy_services}")
```

## Best Practices

### 1. Service Client Initialization

```python
# Good: Initialize clients once and reuse
class ServiceManager:
    def __init__(self):
        self.gpu_client = GPUClient("localhost:50051")
        self.embedding_client = EmbeddingClient("localhost:50052")
        self.reranking_client = RerankingClient("localhost:50053")
        self.docling_client = DoclingVLMClient("localhost:50054")

    async def process_document(self, pdf_content: bytes):
        # Use pre-initialized clients
        pass

# Bad: Initialize clients for each request
async def process_document(pdf_content: bytes):
    gpu_client = GPUClient("localhost:50051")  # Inefficient
    embedding_client = EmbeddingClient("localhost:50052")  # Inefficient
    # ...
```

### 2. Error Handling

```python
# Good: Comprehensive error handling
async def robust_service_call():
    try:
        result = await embedding_client.generate_embeddings(
            texts=["test"],
            model_name="test-model"
        )
        return result
    except ServiceTimeoutError:
        logger.warning("Service timeout, retrying with longer timeout")
        # Retry with longer timeout
        result = await embedding_client.generate_embeddings(
            texts=["test"],
            model_name="test-model",
            timeout=60.0
        )
        return result
    except ServiceUnavailableError:
        logger.error("Service unavailable, failing fast")
        raise
    except ServiceError as e:
        logger.error(f"Service error: {e}")
        raise

# Bad: Generic error handling
async def poor_service_call():
    try:
        result = await embedding_client.generate_embeddings(
            texts=["test"],
            model_name="test-model"
        )
        return result
    except Exception as e:  # Too generic
        logger.error(f"Error: {e}")
        raise
```

### 3. Circuit Breaker Usage

```python
# Good: Use circuit breaker for resilience
circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exception=grpc.RpcError
)

async def resilient_service_call():
    return await circuit_breaker.call(
        embedding_client.generate_embeddings,
        texts=["test"],
        model_name="test-model"
    )

# Bad: No circuit breaker protection
async def fragile_service_call():
    return await embedding_client.generate_embeddings(
        texts=["test"],
        model_name="test-model"
    )
```

### 4. Batch Processing

```python
# Good: Use batch processing for efficiency
async def efficient_batch_processing():
    texts = ["doc1", "doc2", "doc3", "doc4", "doc5"]

    # Process in batches
    batch_size = 2
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_result = await embedding_client.generate_embeddings(
            texts=batch,
            model_name="test-model"
        )
        results.extend(batch_result.embeddings)

    return results

# Bad: Process one by one
async def inefficient_processing():
    texts = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    results = []

    for text in texts:
        result = await embedding_client.generate_embeddings(
            texts=[text],
            model_name="test-model"
        )
        results.extend(result.embeddings)

    return results
```

### 5. Monitoring and Observability

```python
# Good: Add monitoring and metrics
import time
from Medical_KG_rev.observability.metrics import MetricsCollector

metrics = MetricsCollector()

async def monitored_service_call():
    start_time = time.time()

    try:
        result = await embedding_client.generate_embeddings(
            texts=["test"],
            model_name="test-model"
        )

        # Record success metrics
        duration = time.time() - start_time
        metrics.record_service_call(
            service="embedding",
            duration=duration,
            success=True
        )

        return result
    except Exception as e:
        # Record failure metrics
        duration = time.time() - start_time
        metrics.record_service_call(
            service="embedding",
            duration=duration,
            success=False,
            error=str(e)
        )
        raise

# Bad: No monitoring
async def unmonitored_service_call():
    return await embedding_client.generate_embeddings(
        texts=["test"],
        model_name="test-model"
    )
```

### 6. Configuration Management

```python
# Good: Use configuration management
from dataclasses import dataclass
from typing import Optional

@dataclass
class ServiceConfig:
    gpu_service_url: str = "localhost:50051"
    embedding_service_url: str = "localhost:50052"
    reranking_service_url: str = "localhost:50053"
    docling_vlm_service_url: str = "localhost:50054"

    # Timeouts
    gpu_service_timeout: float = 30.0
    embedding_service_timeout: float = 60.0
    reranking_service_timeout: float = 45.0
    docling_vlm_service_timeout: float = 300.0

    # Circuit breaker settings
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0

    # Retry settings
    max_retry_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0

# Load configuration from environment
def load_service_config() -> ServiceConfig:
    import os

    return ServiceConfig(
        gpu_service_url=os.getenv("GPU_SERVICE_URL", "localhost:50051"),
        embedding_service_url=os.getenv("EMBEDDING_SERVICE_URL", "localhost:50052"),
        reranking_service_url=os.getenv("RERANKING_SERVICE_URL", "localhost:50053"),
        docling_vlm_service_url=os.getenv("DOCLING_VLM_SERVICE_URL", "localhost:50054"),
        gpu_service_timeout=float(os.getenv("GPU_SERVICE_TIMEOUT", "30.0")),
        embedding_service_timeout=float(os.getenv("EMBEDDING_SERVICE_TIMEOUT", "60.0")),
        reranking_service_timeout=float(os.getenv("RERANKING_SERVICE_TIMEOUT", "45.0")),
        docling_vlm_service_timeout=float(os.getenv("DOCLING_VLM_SERVICE_TIMEOUT", "300.0"))
    )

# Bad: Hardcoded configuration
class ServiceManager:
    def __init__(self):
        self.gpu_client = GPUClient("localhost:50051")  # Hardcoded
        self.embedding_client = EmbeddingClient("localhost:50052")  # Hardcoded
        # ...
```

This comprehensive API documentation provides all the information needed to work with the GPU services architecture, including service definitions, client usage examples, error handling patterns, and best practices.
