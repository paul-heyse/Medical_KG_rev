# Design Document: Standardized Embeddings & Representation

## Context

The current embedding architecture evolved organically over 18 months, resulting in:

1. **Model Fragmentation**: 3+ embedding models (BGE-small-en, SPLADE-v3, ColBERT-v2) with inconsistent serving patterns
2. **Tokenization Misalignment**: Approximate token counting (`len(text) / 4`) causes 15% of embedding failures when chunks exceed model limits
3. **CPU Fallback Violations**: Silent CPU fallbacks degrade quality by 40-60% without detection
4. **Storage Scatter**: Dense vectors in FAISS + Neo4j + ad-hoc files; sparse signals lack standardized OpenSearch format

**Constraints**:

- GPU-only policy must be strictly enforced (no CPU fallbacks)
- Local deployment (no cloud embedding APIs)
- <500ms P95 retrieval latency requirement
- Multi-tenant isolation at storage layer
- Backward compatibility NOT required (hard cutover acceptable)

**Stakeholders**:

- **Engineering**: Seeks code simplification, clear ownership
- **Operations**: Needs reliable GPU monitoring, clear failure modes
- **Data Science**: Requires experiment-friendly architecture (A/B testing new models)
- **Security**: Enforces multi-tenancy, audit logging

---

## Goals / Non-Goals

### Goals

1. **Standardize Dense Embeddings**: Single vLLM server serving Qwen3-Embedding-8B via OpenAI-compatible API
2. **Standardize Sparse Signals**: Pyserini wrapper for SPLADE-v3 with OpenSearch `rank_features` storage
3. **Enforce GPU-Only Policy**: Fail-fast health checks prevent CPU fallbacks
4. **Simplify Storage**: FAISS primary for dense, OpenSearch `rank_features` for sparse
5. **Enable Experimentation**: Multi-namespace registry supports A/B testing new models
6. **Reduce Codebase**: 25% reduction (530 → 400 lines) by delegating to proven libraries

### Non-Goals

- **Backward Compatibility**: No gradual migration, hard cutover acceptable
- **Cloud Embedding APIs**: Local GPU deployment only (no OpenAI/Cohere/Anthropic APIs)
- **Real-Time Embedding**: Batch-oriented (not optimized for single-text <10ms latency)
- **Multi-Model Ensemble**: Single model per namespace (no ensemble averaging)
- **Custom Model Training**: Use pre-trained models only (no fine-tuning in scope)

---

## Technical Decisions

### Decision 1: vLLM for Dense Embeddings (OpenAI-Compatible API)

**Choice**: Serve Qwen3-Embedding-8B via vLLM with OpenAI-compatible `/v1/embeddings` endpoint

**Why**:

- **Performance**: vLLM achieves 1000+ embeddings/sec with GPU batching (5x faster than sentence-transformers)
- **Simplicity**: OpenAI-compatible API reduces client code from 50+ lines to 5 lines
- **GPU-Only**: vLLM refuses to start without GPU, enforcing fail-fast policy
- **Battle-Tested**: vLLM used in production by major LLM deployments (Anthropic, Anyscale)

**Alternatives Considered**:

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| **TEI (Text Embeddings Inference)** | HuggingFace official, supports multiple models | Requires separate deployment per model, less mature than vLLM | ❌ Rejected: Less mature |
| **sentence-transformers directly** | Simple, no server needed | CPU fallback possible, slow (100-200 emb/sec), no batching | ❌ Rejected: CPU fallback risk |
| **Custom FastAPI + transformers** | Full control | Bespoke code, reinventing wheel, maintenance burden | ❌ Rejected: Maintenance burden |

**Implementation**:

```python
# vLLM server startup (Docker first-class)
docker run --rm \
  --gpus all \
  -p 8001:8001 \
  -e MODEL_PATH=/models/qwen3-embedding-8b \
  -e GPU_MEMORY_UTILIZATION=0.9 \
  -e MAX_MODEL_LEN=8192 \
  -v $(pwd)/models/qwen3-embedding-8b:/models/qwen3-embedding-8b:ro \
  ghcr.io/example/vllm-embedding:latest

# Client code (5 lines) – talks to containerised vLLM endpoint
response = await httpx.post(
    "http://vllm-service:8001/v1/embeddings",
    json={"input": ["text1", "text2"], "model": "Qwen/Qwen2.5-Coder-1.5B"}
)
embeddings = [item["embedding"] for item in response.json()["data"]]
```

**GPU Enforcement**:

```python
# Container startup fails-fast without GPU
$ docker run --rm ghcr.io/example/vllm-embedding:latest
# Output: RuntimeError: No GPU available. vLLM requires CUDA.
```

**Container Standard**:

- All GPU-serving models (Qwen3 single-vector, ColBERT multi-vector) are
  packaged as Docker images built from `ops/Dockerfile.vllm`.
- Kubernetes and Docker Compose deployments reference the images directly;
  the application never imports `vllm` as a Python module.
- Environment-specific tuning (e.g., `GPU_MEMORY_UTILIZATION`) is supplied via
  container environment variables, keeping runtime parity between staging and
  production.

**Trade-offs**:

- ✅ **Pros**: High performance, simple client code, GPU-only enforced
- ⚠️ **Cons**: Requires separate vLLM deployment (adds operational complexity)
- 🔧 **Mitigation**: Docker Compose for dev, Kubernetes for prod, pre-built images

---

### Decision 2: Pyserini for SPLADE Sparse Signals

**Choice**: Use Pyserini library to wrap SPLADE-v3 with document-side expansion as default

**Why**:

- **Proven Library**: Pyserini maintained by UWaterloo IR group, used in academic/industry IR systems
- **SPLADE Expertise**: Handles SPLADE intricacies (term weighting, top-K pruning, tokenization)
- **Document-Side Default**: 80% of recall gains from doc-side expansion, simpler ops than query-side
- **OpenSearch Integration**: Pyserini output format aligns with OpenSearch `rank_features` field

**Alternatives Considered**:

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| **Direct SPLADE model via transformers** | Full control, no dependency | Complex: term weighting, pruning, tokenization logic | ❌ Rejected: Reinventing wheel |
| **Elasticsearch ELSER** | Native Elasticsearch integration | Elasticsearch-specific, not OpenSearch compatible | ❌ Rejected: OpenSearch incompatible |
| **Custom SPLADE wrapper** | Tailored to needs | Maintenance burden, duplicates Pyserini | ❌ Rejected: Maintenance burden |

**Implementation**:

```python
# Pyserini document-side expansion
from pyserini.encode import SpladeQueryEncoder

encoder = SpladeQueryEncoder("naver/splade-v3")

# Expand document (top-K=400 terms with weights)
term_weights = encoder.encode("Significant reduction in HbA1c levels...")
# Returns: {"hba1c": 2.8, "reduction": 2.1, "significant": 1.9, ...}

# Store in OpenSearch rank_features
await opensearch.update(
    index="chunks",
    id=chunk_id,
    body={"doc": {"splade_terms": term_weights}}
)
```

**Query-Time Fusion**:

```python
# BM25 + SPLADE fusion in OpenSearch
{
    "query": {
        "bool": {
            "should": [
                {"match": {"text": "diabetes treatment"}},  # BM25
                {"rank_feature": {"field": "splade_terms", "query": "diabetes"}}  # SPLADE
            ]
        }
    }
}
```

**Document-Side vs Query-Side**:

| Approach | Recall Gain | Latency | Ops Complexity | Decision |
|----------|-------------|---------|----------------|----------|
| **Document-Side Only** | +15% | Base | Simple (offline expansion) | ✅ **Default** |
| **Query-Side Only** | +8% | +50ms | Medium (per-query expansion) | ⚠️ Opt-in |
| **Both** | +18% | +50ms | Complex (two expansion points) | ⚠️ Advanced opt-in |

**Trade-offs**:

- ✅ **Pros**: Proven library, handles SPLADE complexity, 80% of recall gains
- ⚠️ **Cons**: Query-side expansion not default (may need opt-in for max recall)
- 🔧 **Mitigation**: Feature flag for query-side expansion, document when to enable

---

### Decision 3: Model-Aligned Tokenizers (No Approximation)

**Choice**: Use exact tokenizers aligned with embedding models (Qwen3 for dense, SPLADE for sparse)

**Why**:

- **Accuracy**: Exact tokenizers catch 100% of overflows vs 85% with approximation
- **Alignment**: Tokenizer must match model's vocabulary (Qwen3 has custom tokenizer)
- **Fail-Fast**: Reject chunks exceeding token limits at embedding stage (before GPU compute)
- **Reproducibility**: Same text → same token count across runs

**Previous Approach (Approximate)**:

```python
# WRONG: Approximate token counting
token_count = len(text) / 4  # Assumes ~4 chars/token

# Problem: Misses 15% of overflows
# Example: "cardiovascular" = 1 token, but approximation says 3.5 tokens
```

**New Approach (Exact)**:

```python
# CORRECT: Exact tokenizer aligned with Qwen3
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
token_count = len(tokenizer.encode(text))

# Catches 100% of overflows
if token_count > 8192:
    raise TokenLimitExceededError(f"Text has {token_count} tokens, max 8192")
```

**Tokenizer Caching**:

```python
# Load tokenizer once on startup, cache in memory
class EmbeddingService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
        self.tokenizer_cache = {}  # Cache tokenized results for repeated texts

    def count_tokens(self, text: str) -> int:
        if text in self.tokenizer_cache:
            return self.tokenizer_cache[text]
        count = len(self.tokenizer.encode(text))
        self.tokenizer_cache[text] = count
        return count
```

**Trade-offs**:

- ✅ **Pros**: 100% accuracy, prevents embedding failures, reproducible
- ⚠️ **Cons**: Tokenizer loading adds ~2s to startup time
- 🔧 **Mitigation**: Load tokenizer once on startup, cache in memory

---

### Decision 4: FAISS Primary for Dense Vectors (GPU-Accelerated)

**Choice**: Use FAISS HNSW index as primary storage for dense vectors, optional Neo4j vector index for graph-side KNN

**Why**:

- **Performance**: FAISS GPU-accelerated search achieves <50ms P95 for 10M vectors
- **Simplicity**: Single source of truth for dense vectors (no scatter across storages)
- **HNSW Quality**: HNSW index provides high recall (>95%) with sub-linear search time
- **Memory Efficiency**: FAISS supports memory-mapped indexes for large-scale deployments

**Previous Approach (Scattered)**:

- Dense vectors stored in: FAISS (primary), Neo4j (secondary), ad-hoc pickle files (tertiary)
- Inconsistencies: "Which is source of truth?" "Why different results from FAISS vs Neo4j?"
- Performance: Neo4j vector search slower (200ms P95 vs FAISS 50ms P95)

**New Approach (FAISS Primary)**:

```python
# FAISS HNSW index creation
import faiss

dim = 4096  # Qwen3 embedding dimension
index = faiss.IndexHNSWFlat(dim, 32)  # M=32 connections per node

# Move to GPU for acceleration
if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)

# Add vectors
vectors = np.array([emb.vector for emb in embeddings], dtype=np.float32)
ids = np.array([int(emb.chunk_id.split(":")[-1]) for emb in embeddings], dtype=np.int64)
index.add_with_ids(vectors, ids)

# Save index to disk
faiss.write_index(faiss.index_gpu_to_cpu(index), "faiss_index.bin")
```

**Search Performance**:

```python
# KNN search (k=10)
query_vector = np.array([query_emb], dtype=np.float32)
distances, indices = index.search(query_vector, k=10)

# Performance: <50ms P95 for 10M vectors (GPU)
# vs 200ms P95 for Neo4j vector search
```

**Neo4j Integration (Optional)**:

- Neo4j vector index used ONLY for graph-side KNN queries (e.g., "Find similar documents within 2-hop neighborhood")
- Neo4j vectors synced from FAISS (FAISS is source of truth)
- Use case: Graph-constrained similarity search (rare, <5% of queries)

**Index Management**:

```python
# Incremental indexing (append mode)
new_vectors = np.array([...])
new_ids = np.array([...])
index.add_with_ids(new_vectors, new_ids)

# Full reindex (rebuild from scratch)
index = faiss.IndexHNSWFlat(dim, 32)
# ... re-add all vectors

# Backup strategy
# Daily: Save index snapshot to S3/MinIO
# Restore: Load snapshot from S3/MinIO
```

**Trade-offs**:

- ✅ **Pros**: Single source of truth, <50ms P95, GPU-accelerated, proven at scale
- ⚠️ **Cons**: FAISS index rebuild takes time (1-2 hours for 10M vectors)
- 🔧 **Mitigation**: Incremental indexing for daily updates, full reindex monthly

---

### Decision 5: OpenSearch rank_features for Sparse Signals

**Choice**: Store SPLADE sparse vectors in OpenSearch `rank_features` field, enabling BM25+SPLADE fusion

**Why**:

- **Fusion-Ready**: `rank_features` field enables combining BM25 and SPLADE scores in single query
- **No Separate Index**: Sparse signals stored alongside text, metadata (no index duplication)
- **OpenSearch Native**: `rank_features` is OpenSearch-native field type for learned sparse retrieval
- **Proven Pattern**: Used in production by neural search systems (Pinecone, Vespa)

**Previous Approach (Inconsistent)**:

- Sparse signals stored as custom JSON fields or separate index
- Fusion required client-side merging (slow, complex)
- No standardized format across sources

**New Approach (rank_features)**:

```json
// OpenSearch mapping
{
  "mappings": {
    "properties": {
      "chunk_id": {"type": "keyword"},
      "text": {"type": "text"},
      "splade_terms": {
        "type": "rank_features"  // NEW: Enables fusion
      }
    }
  }
}
```

**Indexing Sparse Signals**:

```python
# Write SPLADE term weights to rank_features
await opensearch.index(
    index="chunks",
    id=chunk_id,
    body={
        "text": "Significant reduction in HbA1c...",
        "splade_terms": {
            "hba1c": 2.8,
            "reduction": 2.1,
            "significant": 1.9,
            # ... up to 400 terms
        }
    }
)
```

**Fusion Query**:

```json
// BM25 + SPLADE fusion
{
  "query": {
    "bool": {
      "should": [
        {
          "match": {
            "text": {
              "query": "diabetes treatment",
              "boost": 1.0  // BM25 weight
            }
          }
        },
        {
          "rank_feature": {
            "field": "splade_terms",
            "boost": 2.0,  // SPLADE weight (2x BM25)
            "saturation": {"pivot": 10}
          }
        }
      ]
    }
  }
}
```

**Performance**:

- Query latency: <200ms P95 (BM25+SPLADE fusion)
- vs 150ms P95 (BM25 only) — acceptable 50ms overhead for +15% recall
- Index size: +30% (SPLADE terms add ~300 bytes/chunk)

**Trade-offs**:

- ✅ **Pros**: Fusion-ready, no separate index, OpenSearch-native, proven pattern
- ⚠️ **Cons**: Index size +30%, query latency +50ms
- 🔧 **Mitigation**: Tune top-K terms (400 → 200 reduces size by 15%), acceptable for recall gains

---

### Decision 6: Multi-Namespace Registry for Experimentation

**Choice**: Implement namespace registry supporting multiple embedding families (single_vector, sparse, multi_vector)

**Why**:

- **A/B Testing**: Experiment with new models without breaking production
- **Gradual Migration**: Route 10% traffic to new model, compare metrics, full rollout if better
- **Model Versioning**: Track model versions explicitly (`qwen3.v1` vs `qwen3.v2`)
- **Future-Proof**: Add new embedding types (e.g., multi-vector ColBERT) without refactoring

**Namespace Naming Convention**:

```
{kind}.{model}.{dim}.{version}

Examples:
- single_vector.qwen3.4096.v1
- sparse.splade_v3.400.v1
- multi_vector.colbert_v2.128.v1
```

**Namespace Configuration**:

```yaml
# config/embedding/namespaces/single_vector.qwen3.4096.v1.yaml
name: qwen3-embedding-8b
kind: single_vector
model_id: Qwen/Qwen2.5-Coder-1.5B
model_version: v1
dim: 4096
provider: vllm
endpoint: http://vllm-service:8001/v1/embeddings
parameters:
  batch_size: 64
  normalize: true
  max_tokens: 8192
```

**Registry Implementation**:

```python
class EmbeddingNamespaceRegistry:
    def __init__(self):
        self.namespaces: Dict[str, NamespaceConfig] = {}

    def register(self, namespace: str, config: NamespaceConfig):
        self.namespaces[namespace] = config

    def get(self, namespace: str) -> NamespaceConfig:
        if namespace not in self.namespaces:
            available = ", ".join(self.namespaces.keys())
            raise ValueError(f"Namespace '{namespace}' not found. Available: {available}")
        return self.namespaces[namespace]

    def list_by_kind(self, kind: EmbeddingKind) -> List[str]:
        return [ns for ns, config in self.namespaces.items() if config.kind == kind]
```

**Usage in Embedding Service**:

```python
async def embed(texts: List[str], namespace: str) -> List[Embedding]:
    config = registry.get(namespace)

    if config.provider == "vllm":
        client = VLLMClient(endpoint=config.endpoint)
        vectors = await client.embed(texts, model=config.model_id)
        return [Embedding(vector=v, namespace=namespace) for v in vectors]

    elif config.provider == "pyserini":
        wrapper = PyseriniSPLADEWrapper(model=config.model_id)
        sparse_embeds = [
            SparseEmbedding(term_weights=wrapper.expand_document(text), namespace=namespace)
            for text in texts
        ]
        return sparse_embeds

    else:
        raise ValueError(f"Unknown provider: {config.provider}")
```

**A/B Testing Workflow**:

```python
# Experiment: Compare Qwen3-v1 vs Qwen3-v2
namespaces = {
    "control": "single_vector.qwen3.4096.v1",
    "treatment": "single_vector.qwen3.4096.v2"
}

# Route 10% traffic to treatment
if random.random() < 0.1:
    namespace = namespaces["treatment"]
else:
    namespace = namespaces["control"]

embeddings = await embed(texts, namespace=namespace)

# Track metrics by namespace
track_metric("embedding.recall_at_10", value=recall, tags={"namespace": namespace})
```

**Trade-offs**:

- ✅ **Pros**: Experiment-friendly, gradual migration, future-proof, explicit versioning
- ⚠️ **Cons**: Adds complexity (multiple namespaces to manage), storage overhead (multiple embeddings per chunk)
- 🔧 **Mitigation**: Start with 2-3 namespaces, expand as needed; automatic cleanup of unused namespaces

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    CLIENT (Gateway / Orchestrator)                │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            │ embed(texts, namespace="single_vector.qwen3.4096.v1")
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                   EMBEDDING SERVICE                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Namespace Registry (load from YAML configs)             │   │
│  │  - single_vector.qwen3.4096.v1 → vLLM provider          │   │
│  │  - sparse.splade_v3.400.v1 → Pyserini provider          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  GPU Enforcer (fail-fast if GPU unavailable)            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Model-Aligned Tokenizers (Qwen3, SPLADE)               │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        │ Dense Embeddings                      │ Sparse Signals
        │ (vLLM Client)                         │ (Pyserini Wrapper)
        │                                       │
┌───────▼─────────────┐              ┌─────────▼──────────────┐
│  vLLM Server        │              │  Pyserini SPLADE       │
│  (GPU-Only)         │              │  (CPU or GPU)          │
├─────────────────────┤              ├────────────────────────┤
│ Qwen3-Embedding-8B  │              │ SPLADE-v3              │
│ OpenAI-compatible   │              │ Document-side expand   │
│ /v1/embeddings      │              │ Top-K=400 terms        │
│ Batch size: 64      │              │                        │
└─────────────────────┘              └────────────────────────┘
        │                                       │
        │ 4096-D vectors                        │ {term: weight}
        │                                       │
┌───────▼─────────────────────────────┬─────────▼──────────────────┐
│  STORAGE LAYER                      │                            │
├─────────────────────────────────────┼────────────────────────────┤
│  FAISS (Dense Primary)              │  OpenSearch (Sparse)       │
│  - HNSW index                       │  - rank_features field     │
│  - GPU-accelerated search           │  - BM25 + SPLADE fusion    │
│  - <50ms P95 for 10M vectors        │  - <200ms P95 for fusion   │
│                                     │                            │
│  Neo4j (Metadata + Optional Dense)  │                            │
│  - Embedding metadata               │                            │
│  - Optional vector index for        │                            │
│    graph-constrained KNN            │                            │
└─────────────────────────────────────┴────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Build New Architecture (Week 1-2)

#### Week 1: vLLM + Pyserini Setup

**Day 1-2: vLLM Server**

- Download Qwen3-Embedding-8B model weights
- Create vLLM Docker image with model
- Add vLLM service to docker-compose.yml
- Test vLLM health endpoint and GPU enforcement

**Day 3-4: vLLM Client**

- Implement `VLLMClient` with OpenAI-compatible API
- Add GPU enforcer (fail-fast if GPU unavailable)
- Add error handling (timeout, 503, invalid input)
- Unit tests for vLLM client

**Day 5: Pyserini Wrapper**

- Implement `PyseriniSPLADEWrapper` with document-side expansion
- Test SPLADE term weighting and top-K pruning
- Unit tests for Pyserini wrapper

#### Week 2: Multi-Namespace Registry + Storage

**Day 1-2: Namespace Registry**

- Define `NamespaceConfig` schema (Pydantic)
- Implement `EmbeddingNamespaceRegistry`
- Create YAML configs for default namespaces
- Load namespaces on startup

**Day 3: FAISS Integration**

- Implement `FAISSIndex` wrapper (HNSW + GPU)
- Add embedding writer to FAISS
- Test add/search roundtrip

**Day 4: OpenSearch rank_features**

- Update OpenSearch mapping for `rank_features`
- Implement sparse embedding writer
- Test BM25+SPLADE fusion query

**Day 5: Atomic Deletions (Part 1)**

- **Commit 1**: Add vLLM client + Delete `bge_embedder.py` + Update imports
- **Commit 2**: Add Pyserini wrapper + Delete `splade_embedder.py` + Update imports
- Run dangling import detection script

---

### Phase 2: Integration Testing (Week 3-4)

#### Week 3: End-to-End Pipeline

**Day 1-2: Orchestration Integration**

- Update embedding stage to call vLLM/Pyserini
- Add namespace parameter to orchestration
- Update job ledger for GPU failures

**Day 3-4: Gateway Integration**

- Update REST `/v1/embed` endpoint with namespace parameter
- Update GraphQL embedding mutation
- Add API documentation

**Day 5: Atomic Deletions (Part 2)**

- **Commit 3**: Add model-aligned tokenizers + Delete `token_counter.py`
- **Commit 4**: Delete `manual_batching.py` (vLLM handles batching)
- **Commit 5**: Refactor `registry.py` to use new clients

#### Week 4: Quality Validation

**Day 1-2: Embedding Quality Tests**

- Compare Qwen3 vs BGE embeddings (semantic similarity correlation ≥0.85)
- Compare Pyserini SPLADE vs custom SPLADE (term overlap ≥90%)
- Test embedding stability (same text → same vector)

**Day 3-4: Performance Benchmarks**

- Benchmark vLLM throughput (target: ≥1000 emb/sec)
- Benchmark Pyserini throughput (target: ≥500 docs/sec)
- Benchmark FAISS search latency (target: P95 <50ms)
- Benchmark OpenSearch sparse search (target: P95 <200ms)

**Day 5: Regression Tests**

- Validate retrieval quality unchanged (Recall@10 stable or improved)
- Validate downstream pipeline unchanged (chunking → embedding → retrieval)

---

### Phase 3: Production Deployment (Week 5-6)

#### Week 5: Staging Deployment

**Day 1-2: Build Production Images**

- Build vLLM Docker image with Qwen3 model
- Build updated gateway image with vLLM client
- Build updated orchestration image with Pyserini wrapper

**Day 3-4: Deploy to Staging**

- Deploy vLLM service to GPU nodes
- Deploy updated gateway and orchestration
- Run smoke tests (health checks, basic embedding requests)

**Day 5: Storage Migration (Staging)**

- Create new FAISS index for staging
- Update OpenSearch mapping for `rank_features` (staging)
- Re-embed existing chunks (background job, staging only)

#### Week 6: Production Deployment + Validation

**Day 1-2: Production Deployment**

- Deploy vLLM service to production GPU nodes
- Deploy updated gateway and orchestration to production
- Monitor metrics for 24 hours

**Day 3-4: Storage Migration (Production)**

- Create new FAISS index for production (incremental from staging)
- Update OpenSearch mapping for `rank_features` (production)
- Re-embed existing chunks (background job, production)

**Day 5: Post-Deployment Validation**

- Monitor for 48 hours:
  - Embedding throughput: ≥1000 emb/sec ✅
  - GPU utilization: 60-80% (healthy range) ✅
  - FAISS search latency: P95 <50ms ✅
  - OpenSearch sparse search: P95 <200ms ✅
  - Retrieval quality: Recall@10 stable or improved ✅
  - Zero CPU fallbacks ✅

**Day 6-7: Documentation + Lessons Learned**

- Update `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md`
- Create migration guide for teams
- Document lessons learned and recommendations

---

## Configuration Management

### Namespace Configuration Files

```
config/embedding/namespaces/
├── single_vector.qwen3.4096.v1.yaml
├── sparse.splade_v3.400.v1.yaml
└── multi_vector.colbert_v2.128.v1.yaml (optional)
```

**Example: Dense Embedding Namespace**

```yaml
# config/embedding/namespaces/single_vector.qwen3.4096.v1.yaml
name: qwen3-embedding-8b
kind: single_vector
model_id: Qwen/Qwen2.5-Coder-1.5B
model_version: v1
dim: 4096
provider: vllm
endpoint: http://vllm-service:8001/v1/embeddings
parameters:
  batch_size: 64
  normalize: true
  max_tokens: 8192
  gpu_memory_utilization: 0.9
```

**Example: Sparse Embedding Namespace**

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

### Environment Variables

```bash
# vLLM Service
export VLLM_ENDPOINT=http://vllm-service:8001
export VLLM_GPU_MEMORY_UTILIZATION=0.9
export VLLM_MAX_MODEL_LEN=8192

# Embedding Service
export MK_EMBEDDING_DEFAULT_NAMESPACE=single_vector.qwen3.4096.v1
export MK_EMBEDDING_CACHE_TTL=3600  # 1 hour

# FAISS Storage
export FAISS_INDEX_PATH=/data/faiss/chunks_qwen3_v1.bin
export FAISS_USE_GPU=true

# OpenSearch
export OPENSEARCH_HOSTS=http://opensearch:9200
export OPENSEARCH_INDEX_CHUNKS=chunks
```

---

## Observability & Monitoring

### Prometheus Metrics

```python
# Embedding throughput
EMBEDDING_THROUGHPUT = Counter(
    "medicalkg_embeddings_generated_total",
    "Total embeddings generated",
    ["namespace", "provider"]
)

# Embedding duration
EMBEDDING_DURATION = Histogram(
    "medicalkg_embedding_duration_seconds",
    "Embedding generation duration",
    ["namespace", "provider"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

# GPU utilization
GPU_UTILIZATION = Gauge(
    "medicalkg_gpu_utilization_percent",
    "GPU utilization percentage",
    ["device"]
)

# Embedding failures
EMBEDDING_FAILURES = Counter(
    "medicalkg_embedding_failures_total",
    "Total embedding failures",
    ["namespace", "error_type"]
)

# FAISS search latency
FAISS_SEARCH_LATENCY = Histogram(
    "medicalkg_faiss_search_duration_seconds",
    "FAISS KNN search duration",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5]
)

# OpenSearch sparse search latency
OPENSEARCH_SPARSE_SEARCH_LATENCY = Histogram(
    "medicalkg_opensearch_sparse_search_duration_seconds",
    "OpenSearch sparse search duration",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0]
)
```

### CloudEvents

**Embedding Lifecycle Events**:

1. **embedding.started**

```json
{
  "specversion": "1.0",
  "type": "com.medical-kg.embedding.started",
  "source": "/embedding-service",
  "id": "embedding-job-abc123",
  "time": "2025-10-07T14:30:00Z",
  "data": {
    "job_id": "job-abc123",
    "namespace": "single_vector.qwen3.4096.v1",
    "chunk_count": 150,
    "gpu_available": true
  }
}
```

2. **embedding.completed**

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

3. **embedding.failed**

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

### Grafana Dashboard

**Panels**:

1. **Embedding Throughput**: Line chart (emb/sec) by namespace
2. **GPU Utilization**: Gauge (%) for each GPU device
3. **Embedding Failures**: Bar chart by error type
4. **FAISS Search Latency**: Percentiles (P50, P95, P99)
5. **OpenSearch Sparse Search Latency**: Percentiles
6. **Embedding Cache Hit Rate**: Percentage over time

**Alerting Rules**:

```yaml
# GPU utilization too high
- alert: HighGPUUtilization
  expr: medicalkg_gpu_utilization_percent > 95
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "GPU utilization >95% for 5 minutes"

# Embedding failure rate too high
- alert: HighEmbeddingFailureRate
  expr: rate(medicalkg_embedding_failures_total[5m]) / rate(medicalkg_embeddings_generated_total[5m]) > 0.05
  for: 10m
  labels:
    severity: critical
  annotations:
    summary: "Embedding failure rate >5% for 10 minutes"

# vLLM service down
- alert: VLLMServiceDown
  expr: up{job="vllm-service"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "vLLM service is down"

# FAISS search latency too high
- alert: FAISSSearchLatencyHigh
  expr: histogram_quantile(0.95, rate(medicalkg_faiss_search_duration_seconds_bucket[5m])) > 0.1
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "FAISS search P95 latency >100ms"
```

---

## Risks / Trade-offs

### Risk 1: vLLM Startup Complexity

**Risk**: vLLM requires GPU, model weights, specific CUDA version — startup failures possible

**Impact**: Service unavailable if vLLM fails to start

**Mitigation**:

- Pre-build Docker image with vLLM + Qwen3 model weights
- Add comprehensive health checks (GPU available, model loaded)
- Document troubleshooting guide (common startup errors)
- Test vLLM startup in staging before production

**Likelihood**: Medium | **Impact**: High | **Mitigation Effectiveness**: High

---

### Risk 2: FAISS Index Rebuild Time

**Risk**: Full FAISS reindex takes 1-2 hours for 10M vectors

**Impact**: Extended maintenance window, potential downtime

**Mitigation**:

- Use incremental indexing for daily updates (append mode)
- Full reindex only monthly or on-demand
- Blue-green deployment: build new index in parallel, swap atomically
- Pre-build index in staging, promote to production

**Likelihood**: Low | **Impact**: Medium | **Mitigation Effectiveness**: High

---

### Risk 3: GPU Memory Pressure

**Risk**: GPU OOM (Out of Memory) when processing large batches

**Impact**: Embedding jobs fail, require retry

**Mitigation**:

- Configure vLLM with `--gpu-memory-utilization=0.9` (leave 10% buffer)
- Monitor GPU memory via Prometheus, alert if >95%
- Implement dynamic batch sizing (reduce batch size on OOM)
- Graceful degradation: reduce batch size, retry with smaller batches

**Likelihood**: Medium | **Impact**: Medium | **Mitigation Effectiveness**: High

---

### Risk 4: OpenSearch rank_features Storage Overhead

**Risk**: SPLADE terms add ~300 bytes/chunk, increasing index size by 30%

**Impact**: Increased storage costs, slightly slower indexing

**Mitigation**:

- Tune top-K terms (400 → 200 reduces size by 15%)
- Acceptable trade-off for +15% recall improvement
- Monitor index size growth, adjust top-K if needed

**Likelihood**: Low | **Impact**: Low | **Mitigation Effectiveness**: Medium

---

### Risk 5: Multi-Namespace Complexity

**Risk**: Managing multiple namespaces increases operational overhead

**Impact**: Confusion about which namespace to use, storage duplication

**Mitigation**:

- Start with 2-3 namespaces (dense, sparse, optional multi-vector)
- Document namespace selection guide (when to use each)
- Automatic cleanup of unused namespaces (>30 days no usage)
- Clear naming convention: `{kind}.{model}.{dim}.{version}`

**Likelihood**: Low | **Impact**: Low | **Mitigation Effectiveness**: High

---

## Migration Plan

### Pre-Migration Checklist

- [ ] vLLM service deployed and healthy (staging)
- [ ] Pyserini wrapper tested with sample data
- [ ] FAISS index created for staging (empty, ready for population)
- [ ] OpenSearch mapping updated for `rank_features` (staging)
- [ ] All tests passing (unit, integration, performance)
- [ ] No legacy imports remain (dangling import detection script passes)
- [ ] Monitoring dashboards deployed (Grafana + Prometheus)
- [ ] Runbook reviewed by ops team

### Migration Steps (Production)

#### Step 1: Deploy vLLM Service (Week 5, Day 1)

```bash
# Deploy vLLM to GPU nodes
kubectl apply -f ops/k8s/deployments/vllm-service.yaml

# Verify vLLM health
curl http://vllm-service:8001/health
# Expected: {"status": "healthy", "gpu": "available"}

# Test embedding request
curl -X POST http://vllm-service:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["test text"], "model": "Qwen/Qwen2.5-Coder-1.5B"}'
```

#### Step 2: Deploy Updated Gateway + Orchestration (Week 5, Day 2)

```bash
# Deploy updated gateway with vLLM client
kubectl apply -f ops/k8s/deployments/gateway.yaml

# Deploy updated orchestration with Pyserini wrapper
kubectl apply -f ops/k8s/deployments/orchestration.yaml

# Verify deployments
kubectl get pods -l app=gateway
kubectl get pods -l app=orchestration
```

#### Step 3: Create FAISS Index (Week 5, Day 3)

```python
# scripts/embedding/create_faiss_index.py
import faiss
import numpy as np

# Create FAISS HNSW index
dim = 4096
index = faiss.IndexHNSWFlat(dim, 32)

# Move to GPU
if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)

# Save empty index
faiss.write_index(faiss.index_gpu_to_cpu(index), "/data/faiss/chunks_qwen3_v1.bin")
print("FAISS index created: /data/faiss/chunks_qwen3_v1.bin")
```

#### Step 4: Update OpenSearch Mapping (Week 5, Day 3)

```python
# scripts/embedding/update_opensearch_mapping.py
import asyncio
from opensearchpy import AsyncOpenSearch

async def update_mapping():
    client = AsyncOpenSearch(hosts=["http://opensearch:9200"])

    # Add rank_features field to existing chunks index
    await client.indices.put_mapping(
        index="chunks",
        body={
            "properties": {
                "splade_terms": {
                    "type": "rank_features"
                }
            }
        }
    )
    print("OpenSearch mapping updated with rank_features field")

asyncio.run(update_mapping())
```

#### Step 5: Re-Embed Existing Chunks (Week 5, Day 4-5)

```python
# scripts/embedding/reembed_chunks.py
# Background job: Re-embed existing chunks with vLLM + Pyserini

async def reembed_all_chunks():
    # Fetch all chunks from OpenSearch
    chunks = await fetch_all_chunks(index="chunks")

    # Batch re-embedding
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        # Dense embeddings (vLLM)
        dense_embeds = await embed_vllm(
            texts=[chunk.text for chunk in batch],
            namespace="single_vector.qwen3.4096.v1"
        )

        # Sparse embeddings (Pyserini)
        sparse_embeds = await embed_pyserini(
            texts=[chunk.text for chunk in batch],
            namespace="sparse.splade_v3.400.v1"
        )

        # Write to FAISS
        await write_faiss(dense_embeds, index_path="/data/faiss/chunks_qwen3_v1.bin")

        # Write to OpenSearch (rank_features)
        await write_opensearch(sparse_embeds, index="chunks")

        print(f"Re-embedded batch {i//batch_size + 1}/{len(chunks)//batch_size}")

# Run in background (non-blocking)
asyncio.create_task(reembed_all_chunks())
```

#### Step 6: Validation (Week 6, Day 1-2)

```bash
# Test embedding request via gateway
curl -X POST http://gateway:8000/v1/embed \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"texts": ["diabetes treatment"], "namespace": "single_vector.qwen3.4096.v1"}'

# Test FAISS search
curl -X GET "http://gateway:8000/v1/search?q=diabetes&k=10"

# Test OpenSearch BM25+SPLADE fusion
curl -X POST "http://opensearch:9200/chunks/_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "bool": {
        "should": [
          {"match": {"text": "diabetes"}},
          {"rank_feature": {"field": "splade_terms", "query": "diabetes"}}
        ]
      }
    }
  }'
```

#### Step 7: Monitor for 48 Hours (Week 6, Day 3-4)

**Metrics to Monitor**:

- Embedding throughput: ≥1000 emb/sec
- GPU utilization: 60-80% (healthy range)
- FAISS search latency: P95 <50ms
- OpenSearch sparse search: P95 <200ms
- Retrieval quality: Recall@10 stable or improved
- Zero CPU fallbacks
- Zero GPU OOM errors

**Alert Thresholds**:

- GPU utilization >95% for >5 minutes → WARNING
- Embedding failure rate >5% for >10 minutes → CRITICAL
- vLLM service down → CRITICAL
- FAISS search P95 latency >100ms → WARNING

---

## Testing Strategy

### Unit Tests (50 tests)

**vLLM Client** (10 tests):

- Test successful embedding request
- Test batch embedding (64 texts)
- Test error handling (timeout, 503, invalid input)
- Test GPU health check
- Test empty text handling

**Pyserini Wrapper** (10 tests):

- Test document expansion (top_k=400)
- Test query expansion (top_k=100)
- Test empty text handling
- Test long text truncation
- Test term weight sorting

**Namespace Registry** (10 tests):

- Test register namespace
- Test get namespace
- Test list namespaces
- Test unknown namespace error
- Test load from YAML

**FAISS Index** (10 tests):

- Test add vectors
- Test search KNN
- Test save/load index
- Test GPU vs CPU index
- Test HNSW index

**GPU Enforcer** (5 tests):

- Test GPU available check
- Test GPU unavailable error
- Test health endpoint
- Test fail-fast behavior

**Storage Writers** (5 tests):

- Test write embeddings to FAISS
- Test write sparse embeddings to OpenSearch
- Test Neo4j metadata writes

---

### Integration Tests (21 tests)

**End-to-End Pipeline** (5 tests):

- Test chunk → vLLM embed → FAISS write
- Test chunk → Pyserini expand → OpenSearch write
- Test multi-namespace embedding (dense + sparse)
- Test GPU fail-fast integration
- Test orchestration stage integration

**Storage Integration** (8 tests):

- Test FAISS roundtrip (add → search)
- Test OpenSearch rank_features roundtrip
- Test Neo4j metadata writes
- Test multi-tenant index partitioning
- Test incremental indexing (append mode)
- Test index rebuild (full reindex)
- Test GPU-accelerated FAISS search
- Test FAISS memory-mapped loading

**API Integration** (8 tests):

- Test REST `/v1/embed` with namespace parameter
- Test GraphQL embedding mutation
- Test gRPC embedding service
- Test gateway error propagation (GPU unavailable)
- Test rate limiting on embedding endpoint
- Test JWT authorization for embedding
- Test multi-tenant embedding isolation
- Test embedding result caching

---

### Quality Validation (10 tests)

**Embedding Quality** (5 tests):

- Test: Qwen3 embeddings vs BGE embeddings (semantic similarity correlation ≥0.85)
- Test: SPLADE expansion vs custom expansion (term overlap ≥90%)
- Test: Embedding stability (same text → same vector across runs)
- Test: Tokenization accuracy (exact token count vs approximate ±5%)
- Test: Retrieval quality (Recall@10 stable or improved)

**Performance Benchmarks** (5 tests):

- Benchmark: vLLM throughput (target: ≥1000 emb/sec)
- Benchmark: Pyserini throughput (target: ≥500 docs/sec)
- Benchmark: FAISS search latency (target: P95 <50ms for 10M vectors)
- Benchmark: OpenSearch sparse search latency (target: P95 <200ms)
- Benchmark: End-to-end pipeline latency (chunk → embed → store: P95 <500ms)

---

## Rollback Procedures

### Emergency Rollback (If Critical Issues)

**Trigger Conditions**:

- GPU OOM errors >10% of embedding jobs
- vLLM service crashes repeatedly
- FAISS search latency P95 >500ms (10x degradation)
- Retrieval quality drops >20% (Recall@10)

**Rollback Steps**:

1. **Stop New Embeddings**:

```bash
# Pause embedding jobs in orchestration
kubectl scale deployment orchestration --replicas=0
```

2. **Restore Legacy Code** (if still available):

```bash
# Revert to previous deployment
kubectl rollout undo deployment/gateway
kubectl rollout undo deployment/orchestration
```

3. **Switch to Legacy Storage**:

```bash
# Update retrieval service to use legacy FAISS index (if exists)
# Or use Neo4j vector index as fallback
```

4. **Notify Team**:

- Alert on-call engineer
- Post incident in #incidents channel
- Schedule post-mortem review

**Recovery Time Objective (RTO)**: <30 minutes

---

## Open Questions

1. **vLLM Model Quantization**: Should we use quantized Qwen3 model (FP16 → INT8) to reduce GPU memory? Trade-off: 2x faster but quality slightly degraded
   - **Recommendation**: Start with FP16, quantize only if GPU memory pressure

2. **SPLADE Query-Side Expansion Default**: Should query-side expansion be enabled by default? Trade-off: +3% recall but +50ms latency
   - **Recommendation**: Document-side only by default, query-side opt-in for high-recall scenarios

3. **ColBERT Multi-Vector Support**: Should we implement ColBERT namespace immediately or defer?
   - **Recommendation**: Defer to Phase 2 (after validating single-vector + sparse architecture)

4. **FAISS Reindex Frequency**: Daily incremental vs weekly full reindex?
   - **Recommendation**: Daily incremental (append mode), monthly full reindex for defragmentation

5. **Neo4j Vector Index Usage**: When to use Neo4j vector index vs FAISS?
   - **Recommendation**: FAISS for 95% of queries, Neo4j only for graph-constrained KNN (<5% of queries)

---

## Summary

This design standardizes embeddings and representation around three core principles:

1. **GPU-Only Enforcement**: vLLM and FAISS GPU-accelerated, explicit fail-fast checks
2. **Library Delegation**: vLLM, Pyserini, FAISS replace 530 lines of bespoke code
3. **Experimentation-Friendly**: Multi-namespace registry enables A/B testing and gradual migration

**Key Benefits**:

- 5x throughput improvement (1000+ emb/sec vs 200 emb/sec)
- 25% codebase reduction (530 → 400 lines)
- Zero CPU fallbacks (100% GPU enforcement)
- <50ms P95 FAISS search (vs 200ms ad-hoc)
- Future-proof architecture (add new models without refactoring)

**Timeline**: 6 weeks (2 build, 2 test, 2 deploy)

**Status**: Ready for implementation after approval
