# Storage API Reference

The Storage layer provides abstract interfaces and implementations for object storage, caching, and vector storage backends.

## Core Storage Components

### Base Storage Interfaces

::: Medical_KG_rev.storage.base
    options:
      show_root_heading: true
      members:
        - ObjectStore
        - LedgerStore
        - CacheBackend
        - ObjectMetadata
        - StorageError

## Vector Store Services

### Vector Store Service

::: Medical_KG_rev.services.vector_store.service
    options:
      show_root_heading: true
      members:
        - VectorStoreService
        - VectorStoreConfig
        - VectorStoreError

### Vector Store Registry

::: Medical_KG_rev.services.vector_store.registry
    options:
      show_root_heading: true
      members:
        - VectorStoreRegistry
        - register_vector_store
        - get_vector_store

### Vector Store Factory

::: Medical_KG_rev.services.vector_store.factory
    options:
      show_root_heading: true
      members:
        - VectorStoreFactory
        - create_vector_store
        - VectorStoreType

### Vector Store Monitoring

::: Medical_KG_rev.services.vector_store.monitoring
    options:
      show_root_heading: true
      members:
        - VectorStoreMonitor
        - IndexHealthCheck
        - PerformanceMetrics

### Vector Store Types

::: Medical_KG_rev.services.vector_store.types
    options:
      show_root_heading: true
      members:
        - VectorStoreType
        - IndexConfig
        - SearchConfig
        - VectorMetadata

### Vector Store Models

::: Medical_KG_rev.services.vector_store.models
    options:
      show_root_heading: true
      members:
        - VectorDocument
        - VectorQuery
        - VectorResult
        - VectorIndex

### Vector Store Errors

::: Medical_KG_rev.services.vector_store.errors
    options:
      show_root_heading: true
      members:
        - VectorStoreError
        - IndexError
        - SearchError
        - ConfigurationError

### GPU Vector Store

::: Medical_KG_rev.services.vector_store.gpu
    options:
      show_root_heading: true
      members:
        - GPUVectorStore
        - GPUConfig
        - GPUError

### Vector Store Compression

::: Medical_KG_rev.services.vector_store.compression
    options:
      show_root_heading: true
      members:
        - VectorCompressor
        - CompressionConfig
        - CompressionError

### Vector Store Evaluation

::: Medical_KG_rev.services.vector_store.evaluation
    options:
      show_root_heading: true
      members:
        - VectorStoreEvaluator
        - EvaluationConfig
        - EvaluationMetrics

## Vector Store Implementations

### FAISS Store

::: Medical_KG_rev.services.vector_store.stores.faiss
    options:
      show_root_heading: true
      members:
        - FAISSStore
        - FAISSConfig
        - FAISSError

### Qdrant Store

::: Medical_KG_rev.services.vector_store.stores.qdrant
    options:
      show_root_heading: true
      members:
        - QdrantStore
        - QdrantConfig
        - QdrantError

### Weaviate Store

::: Medical_KG_rev.services.vector_store.stores.weaviate
    options:
      show_root_heading: true
      members:
        - WeaviateStore
        - WeaviateConfig
        - WeaviateError

### Pinecone Store

::: Medical_KG_rev.services.vector_store.stores.pinecone
    options:
      show_root_heading: true
      members:
        - PineconeStore
        - PineconeConfig
        - PineconeError

## Usage Examples

### Basic Vector Store Usage

```python
from Medical_KG_rev.services.vector_store.service import VectorStoreService

# Initialize vector store service
service = VectorStoreService(
    store_type="faiss",
    config={
        "index_type": "HNSW",
        "dimension": 384,
        "metric": "cosine"
    }
)

# Create index
await service.create_index("medical-documents")

# Add documents
documents = [
    {
        "id": "doc-1",
        "vector": [0.1, 0.2, 0.3, ...],  # 384-dimensional vector
        "metadata": {"title": "Medical Paper 1", "doi": "10.1234/paper1"}
    },
    {
        "id": "doc-2",
        "vector": [0.4, 0.5, 0.6, ...],  # 384-dimensional vector
        "metadata": {"title": "Medical Paper 2", "doi": "10.1234/paper2"}
    }
]

await service.add_documents("medical-documents", documents)

# Search for similar documents
query_vector = [0.2, 0.3, 0.4, ...]  # 384-dimensional query vector
results = await service.search(
    index_name="medical-documents",
    query_vector=query_vector,
    top_k=10,
    filter={"doi": {"$exists": True}}
)

for result in results:
    print(f"Document: {result.metadata['title']}")
    print(f"Score: {result.score}")
    print(f"DOI: {result.metadata['doi']}")
```

### FAISS Vector Store Usage

```python
from Medical_KG_rev.services.vector_store.stores.faiss import FAISSStore

# Initialize FAISS store
faiss_store = FAISSStore(
    config={
        "index_type": "HNSW",
        "dimension": 384,
        "metric": "cosine",
        "nlist": 1000,
        "nprobe": 10
    }
)

# Create index
await faiss_store.create_index("medical-documents")

# Add documents
documents = [
    {
        "id": "doc-1",
        "vector": [0.1, 0.2, 0.3, ...],
        "metadata": {"title": "Medical Paper 1"}
    }
]

await faiss_store.add_documents("medical-documents", documents)

# Search
query_vector = [0.2, 0.3, 0.4, ...]
results = await faiss_store.search(
    index_name="medical-documents",
    query_vector=query_vector,
    top_k=10
)

# Save index to disk
await faiss_store.save_index("medical-documents", "/path/to/index.faiss")

# Load index from disk
await faiss_store.load_index("medical-documents", "/path/to/index.faiss")
```

### Qdrant Vector Store Usage

```python
from Medical_KG_rev.services.vector_store.stores.qdrant import QdrantStore

# Initialize Qdrant store
qdrant_store = QdrantStore(
    config={
        "host": "localhost",
        "port": 6333,
        "collection_name": "medical-documents",
        "vector_size": 384,
        "distance": "Cosine"
    }
)

# Create collection
await qdrant_store.create_collection("medical-documents")

# Add points
points = [
    {
        "id": "doc-1",
        "vector": [0.1, 0.2, 0.3, ...],
        "payload": {"title": "Medical Paper 1", "doi": "10.1234/paper1"}
    }
]

await qdrant_store.add_points("medical-documents", points)

# Search
query_vector = [0.2, 0.3, 0.4, ...]
results = await qdrant_store.search(
    collection_name="medical-documents",
    query_vector=query_vector,
    top_k=10,
    filter={"doi": {"$exists": True}}
)
```

### GPU Vector Store Usage

```python
from Medical_KG_rev.services.vector_store.gpu import GPUVectorStore

# Initialize GPU store
gpu_store = GPUVectorStore(
    config={
        "device": "cuda",
        "index_type": "HNSW",
        "dimension": 384,
        "metric": "cosine",
        "gpu_memory_fraction": 0.8
    }
)

# Create index on GPU
await gpu_store.create_index("medical-documents")

# Add documents
documents = [
    {
        "id": "doc-1",
        "vector": [0.1, 0.2, 0.3, ...],
        "metadata": {"title": "Medical Paper 1"}
    }
]

await gpu_store.add_documents("medical-documents", documents)

# Search on GPU
query_vector = [0.2, 0.3, 0.4, ...]
results = await gpu_store.search(
    index_name="medical-documents",
    query_vector=query_vector,
    top_k=10
)

# Move index to CPU for persistence
await gpu_store.move_to_cpu("medical-documents")
```

### Vector Store Compression Usage

```python
from Medical_KG_rev.services.vector_store.compression import VectorCompressor

# Initialize compressor
compressor = VectorCompressor(
    config={
        "compression_type": "quantization",
        "quantization_bits": 8,
        "compression_ratio": 0.5
    }
)

# Compress vectors
original_vectors = [
    [0.1, 0.2, 0.3, ...],
    [0.4, 0.5, 0.6, ...]
]

compressed_vectors = await compressor.compress(original_vectors)

# Decompress vectors
decompressed_vectors = await compressor.decompress(compressed_vectors)

# Check compression ratio
compression_ratio = len(compressed_vectors) / len(original_vectors)
print(f"Compression ratio: {compression_ratio}")
```

### Vector Store Evaluation Usage

```python
from Medical_KG_rev.services.vector_store.evaluation import VectorStoreEvaluator

# Initialize evaluator
evaluator = VectorStoreEvaluator(
    config={
        "metrics": ["precision", "recall", "f1_score", "ndcg"],
        "ground_truth_file": "data/ground_truth.json",
        "query_file": "data/queries.json"
    }
)

# Run evaluation
results = await evaluator.evaluate(
    vector_store=service,
    index_name="medical-documents"
)

# Print results
for metric, value in results.items():
    print(f"{metric}: {value}")

# Generate evaluation report
report = await evaluator.generate_report(results)
print(report)
```

## Configuration

### Vector Store Configuration

```python
# FAISS configuration
FAISS_CONFIG = {
    "index_type": "HNSW",
    "dimension": 384,
    "metric": "cosine",
    "nlist": 1000,
    "nprobe": 10,
    "ef_construction": 200,
    "ef_search": 50
}

# Qdrant configuration
QDRANT_CONFIG = {
    "host": "localhost",
    "port": 6333,
    "collection_name": "medical-documents",
    "vector_size": 384,
    "distance": "Cosine",
    "replication_factor": 1,
    "write_consistency_factor": 1
}

# GPU configuration
GPU_CONFIG = {
    "device": "cuda",
    "index_type": "HNSW",
    "dimension": 384,
    "metric": "cosine",
    "gpu_memory_fraction": 0.8,
    "max_batch_size": 1000
}

# Compression configuration
COMPRESSION_CONFIG = {
    "compression_type": "quantization",
    "quantization_bits": 8,
    "compression_ratio": 0.5,
    "preserve_quality": True
}
```

### Environment Variables

- `VECTOR_STORE_TYPE`: Default vector store type (faiss, qdrant, etc.)
- `VECTOR_STORE_HOST`: Vector store host
- `VECTOR_STORE_PORT`: Vector store port
- `VECTOR_STORE_DIMENSION`: Vector dimension
- `VECTOR_STORE_METRIC`: Distance metric (cosine, euclidean, etc.)
- `GPU_ENABLED`: Enable GPU acceleration
- `GPU_MEMORY_FRACTION`: GPU memory fraction to use

## Error Handling

### Storage Error Hierarchy

```python
# Base storage error
class StorageError(Exception):
    """Base exception for storage errors."""
    pass

# Vector store errors
class VectorStoreError(StorageError):
    """Vector store-specific errors."""
    pass

# Index errors
class IndexError(VectorStoreError):
    """Index-related errors."""
    pass

# Search errors
class SearchError(VectorStoreError):
    """Search-related errors."""
    pass

# Configuration errors
class ConfigurationError(VectorStoreError):
    """Configuration-related errors."""
    pass
```

### Error Handling Patterns

```python
from Medical_KG_rev.services.vector_store.errors import VectorStoreError

try:
    # Execute vector store operation
    result = await service.search(index_name, query_vector, top_k=10)

    if not result:
        # Handle empty result
        logger.warning("Search returned no results")
        return []

except VectorStoreError as e:
    # Handle vector store errors
    if "Index not found" in str(e):
        # Handle missing index
        logger.error("Index not found, creating new index")
        await service.create_index(index_name)
        # Retry operation
        result = await service.search(index_name, query_vector, top_k=10)
    elif "GPU" in str(e):
        # Handle GPU errors
        logger.error("GPU error, falling back to CPU")
        await service.move_to_cpu(index_name)
        result = await service.search(index_name, query_vector, top_k=10)
    else:
        # Handle other errors
        logger.error(f"Vector store error: {e}")
        raise
```

## Performance Considerations

- **GPU Acceleration**: FAISS and other stores support GPU acceleration
- **Batch Operations**: Support for batch operations to improve throughput
- **Index Optimization**: Proper index configuration for performance
- **Memory Management**: Efficient memory usage for large indices
- **Caching**: Query result caching to improve response times

## Monitoring and Observability

- **Index Health**: Monitor index health and performance
- **Search Performance**: Track search latency and throughput
- **Memory Usage**: Monitor memory usage for indices
- **GPU Utilization**: Track GPU usage for GPU-accelerated stores
- **Distributed Tracing**: OpenTelemetry spans for storage operations
- **Structured Logging**: Comprehensive logging with correlation IDs
- **Health Checks**: Vector store health check endpoints

## Testing

### Mock Vector Store

```python
from Medical_KG_rev.services.vector_store.service import VectorStoreService

class MockVectorStore(VectorStoreService):
    """Mock vector store for testing."""

    def __init__(self):
        self.indices = {}
        self.documents = {}

    async def create_index(self, index_name: str) -> None:
        """Mock index creation."""
        self.indices[index_name] = {"created_at": datetime.now()}

    async def add_documents(self, index_name: str, documents: list) -> None:
        """Mock document addition."""
        if index_name not in self.documents:
            self.documents[index_name] = []
        self.documents[index_name].extend(documents)

    async def search(self, index_name: str, query_vector: list, top_k: int = 10) -> list:
        """Mock search."""
        if index_name not in self.documents:
            return []

        # Simple mock search - return first few documents
        docs = self.documents[index_name][:top_k]
        return [{"id": doc["id"], "score": 0.9, "metadata": doc["metadata"]} for doc in docs]
```

### Integration Tests

```python
import pytest
from Medical_KG_rev.services.vector_store.service import VectorStoreService

@pytest.mark.asyncio
async def test_vector_store():
    """Test vector store functionality."""
    service = VectorStoreService(store_type="faiss")

    # Test index creation
    await service.create_index("test-index")

    # Test document addition
    documents = [
        {
            "id": "doc-1",
            "vector": [0.1, 0.2, 0.3],
            "metadata": {"title": "Test Document"}
        }
    ]

    await service.add_documents("test-index", documents)

    # Test search
    query_vector = [0.2, 0.3, 0.4]
    results = await service.search("test-index", query_vector, top_k=10)

    assert len(results) == 1
    assert results[0]["id"] == "doc-1"
    assert results[0]["metadata"]["title"] == "Test Document"

    # Cleanup
    await service.delete_index("test-index")
```
