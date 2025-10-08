# Standardized Embeddings & Representation

## Quick Reference

This proposal replaces fragmented embedding architecture (3+ models, scattered storage, CPU fallbacks) with a unified, library-based system using vLLM, Pyserini, and FAISS.

### Key Changes

- **vLLM Dense Embeddings**: Qwen3-Embedding-8B via OpenAI-compatible API (5x faster: 1000+ emb/sec)
- **Pyserini Sparse Signals**: SPLADE-v3 document-side expansion with OpenSearch `rank_features`
- **Multi-Namespace Registry**: A/B test models, gradual migration, explicit versioning
- **GPU Fail-Fast**: 100% enforcement, no CPU fallbacks
- **FAISS Primary Storage**: GPU-accelerated KNN (<50ms P95 for 10M vectors)

### Breaking Changes

- ❌ Embedding API requires `namespace` parameter (e.g., `namespace="single_vector.qwen3.4096.v1"`)
- ❌ GPU fail-fast: Embedding jobs fail immediately if GPU unavailable (no CPU fallback)
- ❌ FAISS primary for dense vectors (Neo4j vector index opt-in for graph queries only)
- ❌ OpenSearch `rank_features` field required for sparse signals

---

## vLLM Dense Embeddings

### Setup

**Start vLLM Server**:

```bash
# Docker Compose (development)
docker-compose up -d vllm-qwen3

# Kubernetes (production)
kubectl apply -f ops/k8s/base/deployment-vllm-qwen3.yaml
```

**Verify Health**:

```bash
curl http://vllm-qwen3:8001/health
# Expected: {"status": "healthy", "gpu": "available"}
```

### Usage

**Python Client**:

```python
from Medical_KG_rev.services.embedding.vllm import VLLMClient

# Initialize client
client = VLLMClient(base_url="http://vllm-qwen3:8001")

# Embed texts (batch size up to 64)
texts = [
    "Significant reduction in HbA1c levels",
    "Patient reported improved glycemic control"
]

# Call vLLM via OpenAI-compatible API
embeddings = await client.embed(
    texts=texts,
    model="Qwen/Qwen2.5-Coder-1.5B"
)

# Result: np.ndarray with shape (2, 4096)
print(f"Embeddings shape: {embeddings.shape}")
# Output: Embeddings shape: (2, 4096)
```

**Gateway REST API**:

```bash
curl -X POST http://gateway:8000/v1/embed \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["diabetes treatment", "insulin therapy"],
    "namespace": "single_vector.qwen3.4096.v1"
  }'
```

**Response**:

```json
{
  "data": [
    {
      "embedding": [0.023, -0.045, 0.012, ...],  // 4096 dimensions
      "namespace": "single_vector.qwen3.4096.v1"
    },
    {
      "embedding": [-0.012, 0.056, -0.023, ...],
      "namespace": "single_vector.qwen3.4096.v1"
    }
  ],
  "meta": {
    "model": "Qwen/Qwen2.5-Coder-1.5B",
    "duration_ms": 45,
    "gpu_utilized": true
  }
}
```

### GPU Fail-Fast

**GPU Unavailable Handling**:

```python
from Medical_KG_rev.utils.errors import GpuNotAvailableError

try:
    embeddings = await client.embed(texts)
except GpuNotAvailableError as e:
    # Error: "vLLM service reports GPU unavailable"
    logger.error(f"GPU unavailable: {e}")
    # Job ledger updated: status="embed_failed", error="gpu_unavailable"
    # NO CPU fallback attempted
```

**Health Check**:

```bash
curl http://vllm-qwen3:8001/health

# GPU available:
# {"status": "healthy", "gpu": "available"}

# GPU unavailable:
# HTTP 503 Service Unavailable
# {"status": "unhealthy", "gpu": "unavailable"}
```

---

## Pyserini Sparse Embeddings

### Setup

**Install Pyserini**:

```bash
pip install pyserini>=0.22.0
```

**Download SPLADE-v3 Model**:

```python
# Automatic download on first use
from pyserini.encode import SpladeQueryEncoder

encoder = SpladeQueryEncoder("naver/splade-v3")
# Model downloaded to ~/.cache/huggingface/
```

### Usage

**Python Wrapper**:

```python
from Medical_KG_rev.services.embedding.pyserini import PyseriniSPLADEWrapper

# Initialize wrapper
wrapper = PyseriniSPLADEWrapper(model_name="naver/splade-v3")

# Document-side expansion (default, top_k=400)
text = "Significant reduction in HbA1c levels after treatment"
term_weights = wrapper.expand_document(text, top_k=400)

# Result: {term: weight, ...} sorted by weight descending
print(term_weights)
# Output: {
#   "hba1c": 2.8,
#   "reduction": 2.1,
#   "significant": 1.9,
#   "levels": 1.5,
#   "treatment": 1.3,
#   ...  // up to 400 terms
# }
```

**Query-Side Expansion (Opt-In)**:

```python
# Query-side expansion with smaller top_k=100
query = "diabetes treatment"
query_weights = wrapper.expand_query(query, top_k=100)

# Result includes semantic neighbors
print(query_weights)
# Output: {
#   "diabetes": 3.2,
#   "treatment": 2.5,
#   "glucose": 1.8,  // Semantic neighbor (not in original query)
#   "insulin": 1.5,  // Semantic neighbor
#   ...  // up to 100 terms
# }
```

### OpenSearch Integration

**Update Mapping for rank_features**:

```python
# scripts/embedding/update_opensearch_mapping.py
from opensearchpy import AsyncOpenSearch

client = AsyncOpenSearch(hosts=["http://opensearch:9200"])

await client.indices.put_mapping(
    index="chunks",
    body={
        "properties": {
            "splade_terms": {
                "type": "rank_features"  # Enables BM25+SPLADE fusion
            }
        }
    }
)
```

**Write Sparse Embeddings**:

```python
# Write SPLADE terms to OpenSearch
await opensearch.update(
    index="chunks",
    id=chunk_id,
    body={
        "doc": {
            "text": "Significant reduction in HbA1c...",
            "splade_terms": term_weights  # {term: weight}
        }
    }
)
```

**BM25 + SPLADE Fusion Query**:

```json
POST /chunks/_search
{
  "query": {
    "bool": {
      "should": [
        {
          "match": {
            "text": {
              "query": "diabetes treatment",
              "boost": 1.0  // BM25 baseline
            }
          }
        },
        {
          "rank_feature": {
            "field": "splade_terms",
            "boost": 2.0,  // SPLADE weighted 2x
            "saturation": {"pivot": 10}
          }
        }
      ]
    }
  }
}
```

---

## Multi-Namespace Registry

### Namespace Configuration

**Dense Embedding Namespace**:

```yaml
# config/embedding/namespaces/single_vector.qwen3.4096.v1.yaml
name: qwen3-embedding-8b
kind: single_vector
model_id: Qwen/Qwen2.5-Coder-1.5B
model_version: v1
dim: 4096
provider: vllm
endpoint: http://vllm-qwen3:8001/v1/embeddings
parameters:
  batch_size: 64
  normalize: true
  max_tokens: 8192
  gpu_memory_utilization: 0.9
```

**Sparse Embedding Namespace**:

```yaml
# config/embedding/namespaces/sparse.splade_v3.400.v1.yaml
name: splade-v3
kind: sparse
model_id: naver/splade-v3
model_version: v3
dim: 400  # top_k terms
provider: pyserini
parameters:
  top_k: 400
  expand_query_side: false  # Default: doc-side only
```

### Registry Usage

**Load Namespaces**:

```python
from Medical_KG_rev.services.embedding.namespace import load_namespaces

# Load all namespaces from config/embedding/namespaces/*.yaml
registry = load_namespaces()

# List available namespaces
namespaces = registry.list_namespaces()
print(namespaces)
# Output: [
#   "single_vector.qwen3.4096.v1",
#   "sparse.splade_v3.400.v1",
#   "multi_vector.colbert_v2.128.v1"
# ]

# Get namespace config
config = registry.get("single_vector.qwen3.4096.v1")
print(f"Provider: {config.provider}")  # Output: Provider: vllm
print(f"Endpoint: {config.endpoint}")  # Output: Endpoint: http://vllm-qwen3:8001/v1/embeddings
```

**Embed with Namespace**:

```python
from Medical_KG_rev.services.embedding import EmbeddingService

service = EmbeddingService(registry=registry)

# Dense embeddings (vLLM)
dense_embeds = await service.embed(
    texts=["diabetes treatment"],
    namespace="single_vector.qwen3.4096.v1"
)

# Sparse embeddings (Pyserini)
sparse_embeds = await service.embed(
    texts=["diabetes treatment"],
    namespace="sparse.splade_v3.400.v1"
)
```

### A/B Testing Workflow

**Experiment Setup**:

```python
import random

# Define experiment
namespaces = {
    "control": "single_vector.qwen3.4096.v1",
    "treatment": "single_vector.qwen3.4096.v2"
}

# Route 10% traffic to treatment
if random.random() < 0.1:
    namespace = namespaces["treatment"]
else:
    namespace = namespaces["control"]

# Embed with selected namespace
embeddings = await service.embed(texts, namespace=namespace)

# Track metrics by namespace
track_metric(
    "embedding.recall_at_10",
    value=recall,
    tags={"namespace": namespace}
)
```

**Compare Results**:

```python
# Query Neo4j for retrieval metrics by namespace
query = """
MATCH (c:Chunk)-[:HAS_EMBEDDING]->(e:Embedding)
WHERE e.namespace IN $namespaces
RETURN e.namespace, avg(e.retrieval_recall) as avg_recall
"""

results = await neo4j.execute(query, namespaces=list(namespaces.values()))
# Output:
# [
#   {"namespace": "single_vector.qwen3.4096.v1", "avg_recall": 0.82},
#   {"namespace": "single_vector.qwen3.4096.v2", "avg_recall": 0.85}
# ]
```

---

## FAISS Storage

### Index Creation

**Create FAISS HNSW Index**:

```python
from Medical_KG_rev.services.storage.faiss import FAISSIndex
import numpy as np

# Create index (HNSW, GPU-accelerated)
index = FAISSIndex(dim=4096, index_type="HNSW", use_gpu=True)

# Add vectors
vectors = np.array([emb.vector for emb in embeddings], dtype=np.float32)
ids = np.array([int(emb.chunk_id.split(":")[-1]) for emb in embeddings], dtype=np.int64)

index.add(vectors, ids)

# Save to disk
index.save("/data/faiss/chunks_qwen3_v1.bin")
print("FAISS index saved")
```

### Search

**KNN Search**:

```python
# Load index
index = FAISSIndex.load("/data/faiss/chunks_qwen3_v1.bin", use_gpu=True)

# Query vector
query_vector = np.array([query_emb.vector], dtype=np.float32)

# Search for k=10 nearest neighbors
distances, indices = index.search(query_vector, k=10)

# Result: distances and indices of nearest neighbors
print(f"Top 10 neighbors: {indices[0]}")
print(f"Distances: {distances[0]}")

# Performance: <50ms P95 for 10M vectors (GPU)
```

**Incremental Indexing**:

```python
# Append new vectors to existing index
new_vectors = np.array([...], dtype=np.float32)
new_ids = np.array([...], dtype=np.int64)

index.add(new_vectors, new_ids)

# Save updated index
index.save("/data/faiss/chunks_qwen3_v1.bin")
```

---

## Model-Aligned Tokenizers

### Exact Token Counting

**Qwen3 Tokenizer**:

```python
from transformers import AutoTokenizer

# Load tokenizer (cached on first load)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")

# Count tokens (exact, not approximate)
text = "Significant reduction in HbA1c levels after 12 weeks of treatment"
token_count = len(tokenizer.encode(text))

print(f"Token count: {token_count}")  # Output: Token count: 18

# Check against limit
MAX_TOKENS = 8192
if token_count > MAX_TOKENS:
    raise TokenLimitExceededError(f"Text has {token_count} tokens, max {MAX_TOKENS}")
```

**Approximate vs Exact Comparison**:

```python
# Approximate counting (OLD, WRONG)
approximate_count = len(text) / 4
print(f"Approximate: {approximate_count}")  # Output: Approximate: 16.75

# Exact counting (NEW, CORRECT)
exact_count = len(tokenizer.encode(text))
print(f"Exact: {exact_count}")  # Output: Exact: 18

# Difference: Approximate misses overflows
if approximate_count <= MAX_TOKENS and exact_count > MAX_TOKENS:
    print("Approximate would MISS this overflow!")
```

---

## Monitoring & Observability

### Prometheus Metrics

**Embedding Throughput**:

```bash
# Query Prometheus
rate(medicalkg_embeddings_generated_total[5m])

# Expected: ≥1000 emb/sec for dense (vLLM)
#          ≥500 docs/sec for sparse (Pyserini)
```

**GPU Utilization**:

```bash
medicalkg_gpu_utilization_percent

# Healthy range: 60-80%
# Alert if >95% for >5 minutes
```

**Embedding Duration**:

```bash
histogram_quantile(0.95, rate(medicalkg_embedding_duration_seconds_bucket[5m]))

# Expected: P95 <3 seconds for batch of 64
```

**FAISS Search Latency**:

```bash
histogram_quantile(0.95, rate(medicalkg_faiss_search_duration_seconds_bucket[5m]))

# Expected: P95 <50ms for 10M vectors
```

### Grafana Dashboard

**Panels**:

1. **Embedding Throughput**: Line chart (emb/sec) by namespace
   - Query: `rate(medicalkg_embeddings_generated_total{namespace=~".*"}[5m])`

2. **GPU Utilization**: Gauge (%) for each GPU device
   - Query: `medicalkg_gpu_utilization_percent`

3. **Embedding Failures**: Bar chart by error type
   - Query: `increase(medicalkg_embedding_failures_total[1h])`

4. **FAISS Search Latency**: Percentiles (P50, P95, P99)
   - Query: `histogram_quantile(0.95, rate(medicalkg_faiss_search_duration_seconds_bucket[5m]))`

5. **OpenSearch Sparse Search Latency**: Percentiles
   - Query: `histogram_quantile(0.95, rate(medicalkg_opensearch_sparse_search_duration_seconds_bucket[5m]))`

### CloudEvents

**Embedding Completed Event**:

```json
{
  "specversion": "1.0",
  "type": "com.medical-kg.embedding.completed",
  "source": "/embedding-service",
  "id": "embedding-job-abc123",
  "time": "2025-10-07T14:30:02Z",
  "data": {
    "job_id": "job-abc123",
    "namespace": "single_vector.qwen3.4096.v1",
    "embeddings_count": 150,
    "duration_seconds": 2.5,
    "throughput_emb_per_sec": 60,
    "gpu_utilized": true
  }
}
```

**Embedding Failed Event**:

```json
{
  "specversion": "1.0",
  "type": "com.medical-kg.embedding.failed",
  "source": "/embedding-service",
  "id": "embedding-job-abc123",
  "time": "2025-10-07T14:30:01Z",
  "data": {
    "job_id": "job-abc123",
    "namespace": "single_vector.qwen3.4096.v1",
    "error_type": "gpu_unavailable",
    "error_message": "vLLM service reports GPU unavailable",
    "retry_allowed": false
  }
}
```

---

## Migration Guide

### Pre-Migration Checklist

- [ ] vLLM service deployed and healthy (staging)
- [ ] Pyserini wrapper tested with sample data
- [ ] FAISS index created (empty, ready for population)
- [ ] OpenSearch mapping updated for `rank_features` (staging)
- [ ] All tests passing (unit, integration, performance)
- [ ] No legacy imports remain
- [ ] Monitoring dashboards deployed
- [ ] Runbook reviewed by ops team

### Migration Steps

**Step 1: Deploy vLLM Service**:

```bash
# Kubernetes
kubectl apply -f ops/k8s/base/deployment-vllm-qwen3.yaml

# Verify health
curl http://vllm-qwen3:8001/health
```

**Step 2: Update Gateway + Orchestration**:

```bash
kubectl apply -f ops/k8s/deployments/gateway.yaml
kubectl apply -f ops/k8s/deployments/orchestration.yaml
```

**Step 3: Create FAISS Index**:

```bash
python scripts/embedding/create_faiss_index.py
```

**Step 4: Update OpenSearch Mapping**:

```bash
python scripts/embedding/update_opensearch_mapping.py
```

**Step 5: Re-Embed Existing Chunks (Background)**:

```bash
# Run in background (non-blocking)
python scripts/embedding/reembed_chunks.py &
```

**Step 6: Validate for 48 Hours**:

```bash
# Monitor metrics
curl http://gateway:8000/metrics | grep embedding

# Check GPU utilization
nvidia-smi

# Validate retrieval quality
python scripts/evaluate_retrieval.py --before-migration --after-migration
```

---

## Troubleshooting

### vLLM Service Won't Start

**Symptom**: `RuntimeError: No GPU available`

**Solution**:

```bash
# Check GPU availability
nvidia-smi

# Verify CUDA version
nvcc --version

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

### FAISS Search Latency High (>100ms P95)

**Symptom**: Slow KNN search

**Solution**:

```python
# 1. Verify GPU usage
if not index.use_gpu:
    index = FAISSIndex.load(path, use_gpu=True)

# 2. Check index size
print(f"Index size: {index.index.ntotal} vectors")

# 3. Consider index optimization
# If >10M vectors, use IVF index for better scaling
```

### OpenSearch rank_features Not Working

**Symptom**: Sparse search returns no results

**Solution**:

```bash
# Verify mapping
curl http://opensearch:9200/chunks/_mapping | jq '.chunks.mappings.properties.splade_terms'

# Should return: {"type": "rank_features"}

# If not, update mapping
python scripts/embedding/update_opensearch_mapping.py
```

### GPU Out of Memory (OOM)

**Symptom**: `CUDA out of memory` errors

**Solution**:

```bash
# 1. Reduce vLLM GPU memory utilization via container env override
docker compose run --rm \
  -e GPU_MEMORY_UTILIZATION=0.8 \
  vllm-qwen3 --help  # Compose will respect override on next up

# 2. Reduce batch size
# In namespace config: batch_size: 32  # Reduce from 64 to 32

# 3. Monitor GPU memory
watch -n 1 nvidia-smi
```

---

## Dependencies

### New Libraries

```txt
pyserini>=0.22.0       # SPLADE-v3 wrapper with document-side expansion
faiss-gpu>=1.7.4       # GPU-accelerated dense vector search
redis[hiredis]>=5.0.0  # Embedding cache backend
```

vLLM itself ships exclusively as the Docker image
`ghcr.io/example/vllm-qwen3-embedding:latest`; no Python package is imported by the
application code.

### Updated Libraries

```txt
transformers>=4.38.0  # Qwen3 tokenizer support
torch>=2.1.0  # CUDA 12.1+ for FAISS GPU helpers and health checks
```

---

## Performance Benchmarks

| Metric | Legacy | New (vLLM/Pyserini/FAISS) | Improvement |
|--------|--------|---------------------------|-------------|
| **Dense Embedding Throughput** | 200 emb/sec | 1000+ emb/sec | **5x faster** |
| **Sparse Embedding Throughput** | 200 docs/sec | 500+ docs/sec | **2.5x faster** |
| **FAISS Search Latency (P95)** | 200ms (ad-hoc) | <50ms | **4x faster** |
| **GPU Utilization** | Variable (30-90%) | Stable (60-80%) | **Predictable** |
| **Codebase Size** | 530 lines | 400 lines | **25% reduction** |

---

## Status

**Created**: 2025-10-07
**Status**: Ready for implementation
**Timeline**: 6 weeks (2 build, 2 test, 2 deploy)
**Breaking Changes**: 4 (API signature, GPU fail-fast, FAISS primary, rank_features)
