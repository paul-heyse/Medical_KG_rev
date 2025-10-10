# Services API Reference

The Services layer provides core business logic for embedding, chunking, retrieval, evaluation, and GPU-accelerated processing.

## Embedding Services

### Core Embedding Service

::: Medical_KG_rev.services.embedding.service
    options:
      show_root_heading: true
      members:
        - EmbeddingService
        - EmbeddingRequest
        - EmbeddingResponse
        - EmbeddingError

### Embedding Policy Management

::: Medical_KG_rev.services.embedding.policy
    options:
      show_root_heading: true
      members:
        - EmbeddingPolicy
        - PolicyConfig
        - validate_policy

### Embedding Registry

::: Medical_KG_rev.services.embedding.registry
    options:
      show_root_heading: true
      members:
        - EmbeddingRegistry
        - register_embedding
        - get_embedding

### Embedding Persistence

::: Medical_KG_rev.services.embedding.persister
    options:
      show_root_heading: true
      members:
        - EmbeddingPersister
        - save_embeddings
        - load_embeddings

### Embedding Telemetry

::: Medical_KG_rev.services.embedding.telemetry
    options:
      show_root_heading: true
      members:
        - EmbeddingTelemetry
        - record_embedding_time
        - record_embedding_error

## Chunking Services

### Chunking Runtime

::: Medical_KG_rev.services.chunking.runtime
    options:
      show_root_heading: true
      members:
        - ChunkingRuntime
        - ChunkingProfile
        - ChunkingResult
        - create_chunker

### Chunking Profiles

::: Medical_KG_rev.services.chunking.profiles
    options:
      show_root_heading: true
      members:
        - ProfileManager
        - load_profile
        - validate_profile

## Retrieval Services

### Hybrid Retrieval Service

::: Medical_KG_rev.services.retrieval.retrieval_service
    options:
      show_root_heading: true
      members:
        - RetrievalService
        - RetrievalRequest
        - RetrievalResponse
        - RetrievalStrategy

### Vector Store Integration

::: Medical_KG_rev.services.vector_store.service
    options:
      show_root_heading: true
      members:
        - VectorStoreService
        - VectorStoreConfig
        - IndexManager

### Vector Store Monitoring

::: Medical_KG_rev.services.vector_store.monitoring
    options:
      show_root_heading: true
      members:
        - VectorStoreMonitor
        - IndexHealthCheck
        - PerformanceMetrics

## Evaluation Services

### Test Set Management

::: Medical_KG_rev.services.evaluation.test_sets
    options:
      show_root_heading: true
      members:
        - TestSetManager
        - TestSet
        - load_test_set
        - validate_test_set

### Evaluation Metrics

::: Medical_KG_rev.services.evaluation.metrics
    options:
      show_root_heading: true
      members:
        - EvaluationMetrics
        - calculate_precision
        - calculate_recall
        - calculate_f1_score
        - calculate_ndcg

### CI Integration

::: Medical_KG_rev.services.evaluation.ci
    options:
      show_root_heading: true
      members:
        - CIEvaluator
        - run_evaluation
        - generate_report

## Docling VLM Service

### Docling Processor

::: Medical_KG_rev.services.parsing.docling_vlm_service
    options:
      show_root_heading: true
      members:
        - DoclingVLMService
        - DoclingVLMResult

### Docling Metrics

::: Medical_KG_rev.services.parsing.metrics
    options:
      show_root_heading: false
      members:
        - DOCLING_PROCESSING_SECONDS
        - DOCLING_GPU_MEMORY_MB
        - DOCLING_MODEL_LOAD_SECONDS
        - DOCLING_BATCH_SIZE
        - DOCLING_RETRIES_TOTAL
        - DOCLING_SUCCESS_TOTAL

### Docling Exceptions

::: Medical_KG_rev.services.parsing.exceptions
    options:
      show_root_heading: true
      members:
        - DoclingVLMError
        - DoclingModelLoadError
        - DoclingProcessingError
        - DoclingProcessingTimeoutError
        - DoclingModelUnavailableError
        - DoclingOutOfMemoryError
        - TimeoutContext

### Docling Output Parsing

::: Medical_KG_rev.services.parsing.docling
    options:
      show_root_heading: true
      members:
        - DoclingParser
        - DoclingVLMOutputParser

## Health Services

### Health Checks

::: Medical_KG_rev.services.health
    options:
      show_root_heading: true
      members:
        - HealthChecker
        - HealthStatus
        - check_health

## Usage Examples

### Embedding Service Usage

```python
from Medical_KG_rev.services.embedding.service import EmbeddingService

# Initialize service
service = EmbeddingService(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=32
)

# Embed text
texts = ["Medical research paper", "Clinical trial results"]
embeddings = await service.embed(texts)

# Get embedding dimensions
dimensions = service.get_embedding_dimensions()
```

### Chunking Service Usage

```python
from Medical_KG_rev.services.chunking.runtime import ChunkingRuntime

# Initialize runtime
runtime = ChunkingRuntime(profile_name="medical_text")

# Chunk document
document = "Long medical document text..."
chunks = await runtime.chunk(document)

# Get chunk metadata
for chunk in chunks:
    print(f"Chunk: {chunk.text[:100]}...")
    print(f"Start: {chunk.start}, End: {chunk.end}")
```

### Retrieval Service Usage

```python
from Medical_KG_rev.services.retrieval.retrieval_service import RetrievalService

# Initialize service
service = RetrievalService(
    vector_store="faiss",
    retrieval_strategies=["dense", "sparse", "hybrid"]
)

# Search documents
query = "diabetes treatment"
results = await service.search(
    query=query,
    top_k=10,
    namespace="medical"
)

# Process results
for result in results:
    print(f"Score: {result.score}")
    print(f"Document: {result.document.title}")
    print(f"Content: {result.content[:200]}...")
```

### Docling VLM Service Usage

```python
from Medical_KG_rev.services.parsing.docling_vlm_service import DoclingVLMService

# Initialize service (auto-loads configuration from Settings)
service = DoclingVLMService(eager=True)

# Process PDF
pdf_path = "/path/to/document.pdf"
result = service.process_pdf(pdf_path, document_id="demo-doc")

# Get structured output
print(result.metadata["docling_vlm"])
print(f"Extracted {len(result.tables)} tables and {len(result.figures)} figures")
```

## Configuration

### Service Configuration

```python
# Embedding service configuration
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 32,
    "max_length": 512,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Chunking configuration
CHUNKING_CONFIG = {
    "profile": "medical_text",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "preserve_sentences": True
}

# Retrieval configuration
RETRIEVAL_CONFIG = {
    "vector_store": "faiss",
    "index_type": "HNSW",
    "retrieval_strategies": ["dense", "sparse"],
    "fusion_method": "rrf"
}

# Docling VLM configuration
DOCLING_VLM_CONFIG = {
    "model_path": "/models/gemma3-12b",
    "batch_size": 8,
    "timeout_seconds": 300,
    "retry_attempts": 3,
    "gpu_memory_fraction": 0.95,
}
```

### Environment Variables

- `EMBEDDING_MODEL_NAME`: Default embedding model
- `EMBEDDING_BATCH_SIZE`: Batch size for embedding processing
- `CHUNKING_PROFILE`: Default chunking profile
- `VECTOR_STORE_TYPE`: Vector store backend (faiss, qdrant, etc.)
- `DOCLING_VLM_MODEL_PATH`: Location of Gemma3 weights
- `DOCLING_VLM_BATCH_SIZE`: Batch size for Docling inference
- `DOCLING_VLM_TIMEOUT_SECONDS`: Processing timeout per document
- `PDF_PROCESSING_BACKEND`: Feature flag toggling `docling_vlm` vs archived `mineru`

## Performance Considerations

- **Batch Processing**: Services support batch processing for efficiency
- **GPU Acceleration**: Docling VLM and embedding services utilize GPU when available
- **Caching**: Embedding and retrieval results are cached to improve performance
- **Async Operations**: All I/O operations are asynchronous
- **Connection Pooling**: External service connections are pooled

## Error Handling

- **Service-Specific Exceptions**: Each service defines its own exception hierarchy
- **Graceful Degradation**: Services degrade gracefully when dependencies are unavailable
- **Circuit Breakers**: External service calls are protected with circuit breakers
- **Retry Logic**: Failed operations are retried with exponential backoff
- **Health Checks**: Services expose health check endpoints for monitoring

## Monitoring and Observability

- **Metrics Collection**: All services emit Prometheus metrics
- **Distributed Tracing**: OpenTelemetry spans for request tracing
- **Structured Logging**: Comprehensive logging with correlation IDs
- **Health Endpoints**: Health check endpoints for service status
- **Performance Monitoring**: Latency and throughput monitoring
