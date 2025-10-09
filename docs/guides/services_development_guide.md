# Services Development Guide

## Overview

The Services layer contains the core business logic for the Medical Knowledge Graph system. It orchestrates operations across multiple domains including embeddings, chunking, retrieval, evaluation, and document processing.

## Architecture

Services are organized by domain:

- **Embedding Services**: Text embedding generation and management
- **Chunking Services**: Text segmentation and processing
- **Retrieval Services**: Hybrid search and retrieval operations
- **Evaluation Services**: Performance evaluation and metrics
- **MinerU Services**: Document processing and extraction
- **Health Services**: System health monitoring

## Key Components

### Embedding Service

The embedding service (`src/Medical_KG_rev/services/embedding/service.py`) provides:

```python
from dagster import job, op
from Medical_KG_rev.services.embedding.service import EmbeddingService

@op
def generate_embeddings(text: str, model: str) -> List[float]:
    """Generate embeddings for input text."""
    service = EmbeddingService()
    return service.embed(text, model)

@job
def embedding_pipeline():
    """Complete embedding pipeline."""
    embeddings = generate_embeddings()
    return embeddings
```

### Chunking Service

The chunking service (`src/Medical_KG_rev/services/chunking/runtime.py`) handles:

```python
from Medical_KG_rev.services.chunking.runtime import ChunkingRuntime

def chunk_text(text: str, profile: str) -> List[Chunk]:
    """Chunk text using specified profile."""
    runtime = ChunkingRuntime()
    return runtime.chunk(text, profile)
```

### Retrieval Service

The retrieval service (`src/Medical_KG_rev/services/retrieval/retrieval_service.py`) implements:

```python
from Medical_KG_rev.services.retrieval.retrieval_service import RetrievalService

def hybrid_search(query: str, filters: Dict) -> List[SearchResult]:
    """Perform hybrid search across multiple indices."""
    service = RetrievalService()
    return service.search(query, filters)
```

## Development Standards

### Section Headers

All Service modules must follow the `SERVICE_SECTIONS` standard:

```python
# ==============================================================================
# IMPORTS
# ==============================================================================

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

# ==============================================================================
# DATA MODELS
# ==============================================================================

# ==============================================================================
# ERROR HANDLING
# ==============================================================================

# ==============================================================================
# SERVICE IMPLEMENTATION
# ==============================================================================

# ==============================================================================
# PIPELINE ORCHESTRATION
# ==============================================================================

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
```

### Service Interface Pattern

All services should implement a consistent interface:

```python
from abc import ABC, abstractmethod
from typing import Protocol

class ServiceInterface(Protocol):
    """Base interface for all services."""

    def initialize(self) -> None:
        """Initialize the service."""
        ...

    def process(self, data: Any) -> Any:
        """Process input data."""
        ...

    def cleanup(self) -> None:
        """Cleanup service resources."""
        ...

class EmbeddingService:
    """Embedding service implementation."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None

    def initialize(self) -> None:
        """Initialize the embedding model."""
        self.model = load_model(self.config.model_path)

    def process(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        if not self.model:
            raise ServiceError("Service not initialized")
        return self.model.embed(text)

    def cleanup(self) -> None:
        """Cleanup model resources."""
        if self.model:
            self.model.cleanup()
            self.model = None
```

### Error Handling

Services must implement comprehensive error handling:

```python
from enum import Enum
from typing import Optional

class ServiceErrorType(Enum):
    """Service error types."""
    INITIALIZATION_ERROR = "initialization_error"
    PROCESSING_ERROR = "processing_error"
    CONFIGURATION_ERROR = "configuration_error"
    RESOURCE_ERROR = "resource_error"

class ServiceError(Exception):
    """Base service error."""

    def __init__(
        self,
        error_type: ServiceErrorType,
        message: str,
        details: Optional[Dict] = None
    ):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        super().__init__(message)

def handle_service_error(func):
    """Decorator for service error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if isinstance(e, ServiceError):
                raise
            else:
                raise ServiceError(
                    ServiceErrorType.PROCESSING_ERROR,
                    f"Unexpected error in {func.__name__}: {str(e)}"
                )
    return wrapper
```

## Configuration Management

### Service Configuration

Services should use structured configuration:

```python
from pydantic import BaseModel, Field
from typing import Optional

class EmbeddingConfig(BaseModel):
    """Embedding service configuration."""

    model_path: str = Field(..., description="Path to embedding model")
    batch_size: int = Field(default=32, description="Batch size for processing")
    max_length: int = Field(default=512, description="Maximum sequence length")
    device: str = Field(default="cpu", description="Device for model execution")

    class Config:
        env_prefix = "EMBEDDING_"

class ChunkingConfig(BaseModel):
    """Chunking service configuration."""

    profile_path: str = Field(..., description="Path to chunking profiles")
    default_profile: str = Field(default="medical", description="Default chunking profile")
    overlap_size: int = Field(default=50, description="Overlap between chunks")

    class Config:
        env_prefix = "CHUNKING_"
```

### Configuration Loading

Load configuration from multiple sources:

```python
from pathlib import Path
import yaml
from Medical_KG_rev.services.config import load_config

def load_service_config(config_path: Path) -> Dict:
    """Load service configuration from file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def get_service_config(service_name: str) -> Dict:
    """Get configuration for a specific service."""
    config_path = Path(f"config/services/{service_name}.yaml")
    if config_path.exists():
        return load_service_config(config_path)
    else:
        return load_config(service_name)
```

## Testing

### Unit Tests

Services should have comprehensive unit tests:

```python
import pytest
from unittest.mock import Mock, patch
from Medical_KG_rev.services.embedding.service import EmbeddingService

class TestEmbeddingService:
    """Test cases for embedding service."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = EmbeddingConfig(
            model_path="test_model",
            batch_size=16
        )
        self.service = EmbeddingService(self.config)

    def test_initialization(self):
        """Test service initialization."""
        with patch('Medical_KG_rev.services.embedding.service.load_model') as mock_load:
            self.service.initialize()
            mock_load.assert_called_once_with("test_model")

    def test_embedding_generation(self):
        """Test embedding generation."""
        self.service.model = Mock()
        self.service.model.embed.return_value = [0.1, 0.2, 0.3]

        result = self.service.process("test text")
        assert result == [0.1, 0.2, 0.3]
        self.service.model.embed.assert_called_once_with("test text")

    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ServiceError) as exc_info:
            self.service.process("test text")
        assert exc_info.value.error_type == ServiceErrorType.PROCESSING_ERROR
```

### Integration Tests

Integration tests should verify service interactions:

```python
def test_embedding_pipeline():
    """Test complete embedding pipeline."""
    # Setup
    config = EmbeddingConfig(model_path="test_model")
    service = EmbeddingService(config)

    # Test
    service.initialize()
    embeddings = service.process("Sample medical text")

    # Verify
    assert len(embeddings) > 0
    assert all(isinstance(x, float) for x in embeddings)

    # Cleanup
    service.cleanup()
```

## Performance Optimization

### Caching

Implement caching for expensive operations:

```python
from functools import lru_cache
from typing import Dict, Any

class CachedEmbeddingService(EmbeddingService):
    """Embedding service with caching."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._cache: Dict[str, List[float]] = {}

    @lru_cache(maxsize=1000)
    def get_cached_embedding(self, text: str) -> List[float]:
        """Get cached embedding if available."""
        if text in self._cache:
            return self._cache[text]

        embedding = self.model.embed(text)
        self._cache[text] = embedding
        return embedding
```

### Batch Processing

Implement batch processing for efficiency:

```python
from typing import List, Iterator

def process_batch(
    texts: List[str],
    batch_size: int = 32
) -> Iterator[List[float]]:
    """Process texts in batches."""
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        yield [embed_text(text) for text in batch]
```

### Async Processing

Use async processing for I/O operations:

```python
import asyncio
from typing import List

async def async_embed_text(text: str) -> List[float]:
    """Asynchronously embed text."""
    # Simulate async model loading
    await asyncio.sleep(0.1)
    return embed_text(text)

async def process_texts_async(texts: List[str]) -> List[List[float]]:
    """Process multiple texts asynchronously."""
    tasks = [async_embed_text(text) for text in texts]
    return await asyncio.gather(*tasks)
```

## Monitoring and Observability

### Metrics Collection

Collect service-specific metrics:

```python
from prometheus_client import Counter, Histogram, Gauge

# Service metrics
SERVICE_REQUESTS = Counter('service_requests_total', 'Total requests', ['service', 'operation'])
SERVICE_DURATION = Histogram('service_duration_seconds', 'Service duration', ['service', 'operation'])
SERVICE_ERRORS = Counter('service_errors_total', 'Total errors', ['service', 'error_type'])
ACTIVE_CONNECTIONS = Gauge('service_active_connections', 'Active connections', ['service'])

def track_service_operation(service_name: str, operation: str):
    """Track service operation metrics."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            SERVICE_REQUESTS.labels(service=service_name, operation=operation).inc()
            with SERVICE_DURATION.labels(service=service_name, operation=operation).time():
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    SERVICE_ERRORS.labels(service=service_name, error_type=type(e).__name__).inc()
                    raise
        return wrapper
    return decorator
```

### Health Checks

Implement health checks for services:

```python
from enum import Enum
from typing import Dict, Any

class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class ServiceHealth:
    """Service health checker."""

    def __init__(self, service: ServiceInterface):
        self.service = service

    def check_health(self) -> Dict[str, Any]:
        """Check service health."""
        try:
            # Test basic functionality
            self.service.initialize()
            result = self.service.process("health_check")
            self.service.cleanup()

            return {
                "status": HealthStatus.HEALTHY.value,
                "details": {"test_result": "passed"}
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "details": {"error": str(e)}
            }
```

## Deployment

### Docker Configuration

Services can be deployed using Docker:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

# Service-specific configuration
ENV SERVICE_NAME=embedding
ENV MODEL_PATH=/models/embedding_model

CMD ["python", "-m", "Medical_KG_rev.services.embedding.service"]
```

### Kubernetes Deployment

Deploy services using Kubernetes:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedding-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: embedding-service
  template:
    metadata:
      labels:
        app: embedding-service
    spec:
      containers:
      - name: embedding-service
        image: medical-kg/embedding-service:latest
        ports:
        - containerPort: 8001
        env:
        - name: MODEL_PATH
          value: "/models/embedding_model"
        - name: BATCH_SIZE
          value: "32"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## Troubleshooting

### Common Issues

1. **Model Loading Failures**: Check model path and permissions
2. **Memory Issues**: Monitor memory usage and implement batching
3. **Configuration Errors**: Validate configuration files
4. **Service Dependencies**: Ensure all dependencies are available

### Debugging

Enable debug mode for development:

```python
import logging
import os

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Enable service debug mode
os.environ["SERVICE_DEBUG"] = "true"

# Add debug middleware
def debug_middleware(func):
    def wrapper(*args, **kwargs):
        logging.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logging.debug(f"{func.__name__} returned: {result}")
        return result
    return wrapper
```

## Best Practices

1. **Error Handling**: Implement comprehensive error handling with specific error types
2. **Configuration**: Use structured configuration with validation
3. **Testing**: Write comprehensive unit and integration tests
4. **Performance**: Implement caching, batching, and async processing
5. **Monitoring**: Collect metrics and implement health checks
6. **Documentation**: Maintain up-to-date service documentation
7. **Security**: Validate inputs and implement proper authentication
8. **Resource Management**: Properly manage resources and cleanup
