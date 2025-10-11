# Torch Isolation Architecture Migration Guide

## Overview

This guide covers the migration from the monolithic torch-based architecture to the new torch isolation architecture. The new architecture separates GPU-intensive operations into dedicated Docker services while keeping the main API gateway torch-free.

## Architecture Changes

### Before (Monolithic)

```
┌─────────────────────────────────────┐
│           Main Gateway              │
│  ┌─────────┐ ┌─────────┐ ┌──────┐ │
│  │ Chunking│ │Embedding│ │GPU Mgmt│ │
│  │ (torch) │ │ (torch) │ │(torch)│ │
│  └─────────┘ └─────────┘ └──────┘ │
└─────────────────────────────────────┘
```

### After (Torch Isolation)

```
┌─────────────────┐    gRPC    ┌─────────────────┐
│   Main Gateway  │ ──────────▶│  GPU Services   │
│   (torch-free)  │            │    (torch)     │
└─────────────────┘            └─────────────────┘
         │                              │
         │ gRPC                         │ gRPC
         ▼                              ▼
┌─────────────────┐            ┌─────────────────┐
│Embedding Services│            │Reranking Services│
│    (torch)      │            │    (torch)     │
└─────────────────┘            └─────────────────┘
```

## Key Changes

### 1. Chunking Module Changes

**Before:**

```python
from Medical_KG_rev.chunking.chunkers.semantic import SemanticSplitterChunker

chunker = SemanticSplitterChunker(
    gpu_semantic_checks=True,  # Required torch
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

**After:**

```python
from Medical_KG_rev.chunking.chunkers.docling import DoclingChunker

chunker = DoclingChunker(
    docling_output_store=docling_store
)
```

### 2. Embedding Generation Changes

**Before:**

```python
from Medical_KG_rev.services.embedding import EmbeddingService

service = EmbeddingService()
embeddings = service.generate_embeddings(texts)
```

**After:**

```python
from Medical_KG_rev.services.embedding.grpc_client import EmbeddingServiceClient

client = EmbeddingServiceClient("embedding-services:50051")
embeddings = client.generate_embeddings(texts)
```

### 3. Reranking Changes

**Before:**

```python
from Medical_KG_rev.services.reranking import RerankingService

service = RerankingService()
scores = service.rerank(query, documents)
```

**After:**

```python
from Medical_KG_rev.services.reranking.grpc_client import RerankingServiceClient

client = RerankingServiceClient("reranking-services:50051")
scores = client.rerank(query, documents)
```

### 4. GPU Management Changes

**Before:**

```python
from Medical_KG_rev.services.gpu import GPUManager

manager = GPUManager()
status = manager.get_gpu_status()
```

**After:**

```python
from Medical_KG_rev.services.gpu.grpc_client import GPUServiceClient

client = GPUServiceClient("gpu-services:50051")
status = client.get_gpu_status()
```

## Configuration Changes

### 1. Gateway Configuration

**Before:**

```yaml
# config/gateway.yaml
gateway:
  chunking:
    semantic_chunker:
      gpu_enabled: true
      model_name: "sentence-transformers/all-MiniLM-L6-v2"

  embedding:
    gpu_enabled: true
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
```

**After:**

```yaml
# config/gateway.yaml
gateway:
  services:
    gpu:
      url: "gpu-services:50051"
    embedding:
      url: "embedding-services:50051"
    reranking:
      url: "reranking-services:50051"

  chunking:
    default_chunker: "docling"
    chunkers:
      docling:
        enabled: true
```

### 2. Docker Configuration

**Before:**

```yaml
# docker-compose.yml
services:
  gateway:
    build: .
    environment:
      - GPU_ENABLED=true
```

**After:**

```yaml
# docker-compose.torch-isolation.yml
services:
  gateway:
    build:
      dockerfile: ops/docker/gateway/Dockerfile
    environment:
      - GPU_SERVICE_URL=gpu-services:50051
      - EMBEDDING_SERVICE_URL=embedding-services:50051
      - RERANKING_SERVICE_URL=reranking-services:50051

  gpu-services:
    build:
      dockerfile: ops/docker/gpu-services/Dockerfile
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Migration Steps

### Step 1: Update Dependencies

Remove torch dependencies from the main gateway:

```bash
# Remove torch from requirements.txt
sed -i '/torch/d' requirements.txt
sed -i '/torchvision/d' requirements.txt
sed -i '/torchaudio/d' requirements.txt

# Add gRPC dependencies
echo "grpcio>=1.60.0" >> requirements.txt
echo "grpcio-tools>=1.60.0" >> requirements.txt
echo "grpc-health-checking>=1.60.0" >> requirements.txt
```

### Step 2: Update Code References

Replace direct service calls with gRPC clients:

```bash
# Find and replace embedding service calls
find src/ -name "*.py" -exec sed -i 's/EmbeddingService/EmbeddingServiceClient/g' {} \;

# Find and replace reranking service calls
find src/ -name "*.py" -exec sed -i 's/RerankingService/RerankingServiceClient/g' {} \;

# Find and replace GPU manager calls
find src/ -name "*.py" -exec sed -i 's/GPUManager/GPUServiceClient/g' {} \;
```

### Step 3: Update Configuration

Update configuration files to use service endpoints:

```bash
# Update gateway configuration
cp src/Medical_KG_rev/config/gateway.yaml config/gateway.yaml.backup
# Edit config/gateway.yaml to use new service endpoints
```

### Step 4: Update Docker Configuration

Switch to the new Docker Compose configuration:

```bash
# Stop existing services
docker-compose down

# Start new architecture
./scripts/start_torch_isolation.sh
```

### Step 5: Test Migration

Run integration tests to verify the migration:

```bash
# Run torch isolation tests
pytest tests/integration/test_torch_isolation.py -v

# Check service health
./scripts/check_torch_isolation_health.sh
```

## Breaking Changes

### 1. Chunking API Changes

**Before:**

```python
# Semantic chunking with GPU
chunker = SemanticSplitterChunker(gpu_semantic_checks=True)
```

**After:**

```python
# Docling chunking (torch-free)
chunker = DoclingChunker(docling_output_store=store)
```

### 2. Service Initialization Changes

**Before:**

```python
# Direct service initialization
embedding_service = EmbeddingService()
```

**After:**

```python
# gRPC client initialization
embedding_client = EmbeddingServiceClient("embedding-services:50051")
```

### 3. Configuration Changes

**Before:**

```yaml
# Direct GPU configuration
gpu:
  enabled: true
  device_id: 0
```

**After:**

```yaml
# Service endpoint configuration
services:
  gpu:
    url: "gpu-services:50051"
```

## Performance Considerations

### 1. Network Latency

The new architecture introduces network latency for GPU operations. Consider:

- Use localhost for development
- Use service mesh for production
- Implement connection pooling
- Use batch processing when possible

### 2. Resource Allocation

GPU services now run in separate containers:

- Allocate sufficient GPU memory
- Monitor GPU utilization
- Implement proper resource limits
- Use GPU sharing when appropriate

### 3. Fault Tolerance

Implement circuit breakers and retries:

```python
from Medical_KG_rev.services.base import CircuitBreaker

client = EmbeddingServiceClient(
    "embedding-services:50051",
    circuit_breaker=CircuitBreaker(
        failure_threshold=5,
        timeout=60
    )
)
```

## Troubleshooting

### Common Issues

1. **Service Discovery Failures**
   - Check Docker network configuration
   - Verify service names and ports
   - Check DNS resolution

2. **GPU Service Failures**
   - Verify NVIDIA Docker runtime
   - Check GPU availability
   - Monitor GPU memory usage

3. **Performance Degradation**
   - Check network latency
   - Monitor service health
   - Verify resource allocation

### Debug Commands

```bash
# Check service status
docker-compose -f ops/docker/docker-compose.torch-isolation.yml ps

# View service logs
docker-compose -f ops/docker/docker-compose.torch-isolation.yml logs -f

# Check GPU status
docker exec gpu-services python -c "import torch; print(torch.cuda.is_available())"

# Test gRPC connectivity
grpcurl -plaintext localhost:50051 list
```

## Rollback Plan

If issues arise, rollback to the previous architecture:

1. **Stop New Services**

   ```bash
   ./scripts/stop_torch_isolation.sh
   ```

2. **Restore Previous Configuration**

   ```bash
   cp config/gateway.yaml.backup config/gateway.yaml
   ```

3. **Start Previous Services**

   ```bash
   docker-compose up -d
   ```

4. **Verify Rollback**

   ```bash
   pytest tests/integration/test_previous_architecture.py -v
   ```

## Support

For migration support:

1. Check the troubleshooting section
2. Review service logs
3. Test individual components
4. Contact the development team

## Conclusion

The torch isolation architecture provides better scalability, maintainability, and resource utilization. While it introduces some complexity, the benefits outweigh the costs for production deployments.

Key benefits:

- **Scalability**: Independent scaling of GPU services
- **Maintainability**: Clear separation of concerns
- **Resource Efficiency**: Better GPU utilization
- **Fault Tolerance**: Isolated failure domains
- **Development**: Faster iteration on torch-free components
