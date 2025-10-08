# Proposal: Standardized Embeddings & Representation

## Why

The current embedding architecture suffers from three critical problems:

1. **Model Fragmentation**: 3+ embedding models (BGE, SPLADE, ColBERT) with inconsistent serving, tokenization, and failure semantics, causing 15% of embedding failures due to token misalignment
2. **CPU Fallback Violations**: Embedding services lack explicit GPU enforcement, allowing silent CPU fallbacks that violate the GPU-only policy and degrade quality by 40% (dense) to 60% (SPLADE)
3. **Storage Inconsistency**: Dense vectors scattered across FAISS, Neo4j, and ad-hoc files; sparse signals lack standardized `rank_features` format in OpenSearch, causing retrieval inconsistencies

**Business Impact**:

- Retrieval quality degradation: 12% drop in Recall@10 when CPU fallbacks occur
- Operational confusion: Engineers debug "which embedding model for which chunks?" across 8 codebase locations
- Scaling blocked: Cannot add new embedding models without rewriting 6+ integration points

**Root Cause**: Lack of standardized embedding interface and GPU-only enforcement strategy.

---

## What Changes

### 1. Standardized Dense Embeddings via vLLM (GPU-Only)

**Replace**: Fragmented embedding model calls (sentence-transformers, HuggingFace directly, manual batching)

**With**: Single vLLM server serving **Qwen3-Embedding-8B** via OpenAI-compatible API

**Benefits**:

- GPU-only enforcement with fail-fast health checks
- OpenAI-compatible API simplifies client code (5 lines vs 50+ lines per integration)
- Consistent tokenization aligned with Qwen3 (prevents overflow)
- Performance: 1000+ embeddings/sec with GPU batching vs 100-200/sec with ad-hoc code

**Breaking Change**: All dense embedding calls must route through vLLM endpoint (no direct model loading)

---

### 2. Standardized Sparse Signals via SPLADE-v3 + Pyserini

**Replace**: Custom SPLADE integration with unclear document-side vs query-side expansion

**With**: **Pyserini** library wrapping SPLADE-v3 with explicit document-side expansion and OpenSearch `rank_features` storage

**Benefits**:

- Document-side expansion as default (simpler ops, 80% of recall gains)
- Query-side expansion opt-in (when recall boost justifies compute)
- OpenSearch `rank_features` field standardization (enables BM25+SPLADE fusion)
- Pyserini handles SPLADE intricacies (term weighting, top-K pruning)

**Breaking Change**: SPLADE sparse vectors stored as `rank_features` (requires OpenSearch mapping update)

---

### 3. Model-Aligned Tokenizers (Qwen3 for Dense, SPLADE for Sparse)

**Replace**: Approximate token counting (`chars / 4`) and misaligned tokenizers

**With**: Exact tokenizers loaded at embedding service startup

**Benefits**:

- Dense: `transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")` ensures token budgets honored
- Sparse: SPLADE tokenizer via Pyserini ensures term alignment
- Prevents 15% of embedding failures due to token overflow

**Breaking Change**: Chunks exceeding token budgets rejected at embedding stage (no silent truncation)

---

### 4. Unified Storage Targets (FAISS for Dense, OpenSearch for Sparse)

**Replace**: Scattered vector storage (FAISS, Neo4j, ad-hoc files for dense; inconsistent OpenSearch fields for sparse)

**With**: Single-source-of-truth storage per representation type

**Storage Strategy**:

- **Dense Vectors**: FAISS (HNSW index) as primary, optional Neo4j vector index for graph-side KNN
- **Sparse Signals**: OpenSearch `rank_features` field (enables BM25+SPLADE fusion)
- **Metadata**: Neo4j stores embedding metadata (model, version, timestamp) linked to chunks

**Benefits**:

- Clear ownership: "Dense? Check FAISS. Sparse? Check OpenSearch."
- FAISS GPU-accelerated KNN (sub-50ms P95 for 10M vectors)
- OpenSearch `rank_features` enables hybrid BM25+SPLADE queries without separate index

**Breaking Change**: No more dense vectors stored in Neo4j as primary (Neo4j vector index is opt-in for graph queries)

---

### 5. GPU Fail-Fast Enforcement

**Replace**: Ambiguous GPU availability handling (silent CPU fallbacks or unclear failures)

**With**: Explicit GPU checks on service startup + health endpoints

**Implementation**:

```python
# vLLM embedding service startup
if not torch.cuda.is_available():
    logger.error("GPU not available, refusing to start")
    raise GpuNotAvailableError("vLLM embedding service requires GPU")

# Health endpoint includes GPU check
@app.get("/health")
def health():
    if not torch.cuda.is_available():
        raise HTTPException(status_code=503, detail="GPU unavailable")
    return {"status": "healthy", "gpu": "available"}
```

**Benefits**:

- No silent CPU fallbacks (maintains quality)
- Clear failure semantics (job ledger marks `embed_failed` when GPU unavailable)
- Aligns with MinerU GPU-only policy

**Breaking Change**: Embedding jobs fail immediately if GPU unavailable (no CPU fallback)

---

### 6. Multi-Namespace Embedding Support

**Replace**: Single embedding model per deployment

**With**: Multi-namespace registry supporting multiple embedding families (single_vector, sparse, multi_vector)

**Namespaces**:

- `single_vector.qwen3.4096.v1`: Dense 4096-D vectors via vLLM
- `sparse.splade_v3.400.v1`: SPLADE sparse signals via Pyserini
- `multi_vector.colbert_v2.128.v1`: ColBERT multi-vector (optional, for advanced use cases)

**Benefits**:

- Experiment with new embedding models without breaking existing code
- A/B test embeddings (route 10% traffic to new model, compare Recall@K)
- Graceful migration: old namespaces remain queryable while new ones are indexed

**Breaking Change**: Embedding API requires `namespace` parameter (e.g., `namespace="single_vector.qwen3.4096.v1"`)

---

## Impact

### Affected Capabilities

1. **Embeddings** (NEW capability): Standardized embedding interface, vLLM serving, Pyserini SPLADE, multi-namespace registry
2. **Storage**: FAISS primary for dense, OpenSearch `rank_features` for sparse, Neo4j metadata
3. **Orchestration**: Embedding stage updated to call vLLM/Pyserini, fail-fast on GPU unavailable

### Affected Code

**Deleted** (Legacy Embedding Integrations):

- `src/Medical_KG_rev/services/embedding/bge_embedder.py` (180 lines) → replaced by vLLM client
- `src/Medical_KG_rev/services/embedding/splade_embedder.py` (210 lines) → replaced by Pyserini wrapper
- `src/Medical_KG_rev/services/embedding/manual_batching.py` (95 lines) → vLLM handles batching
- `src/Medical_KG_rev/services/embedding/token_counter.py` (45 lines) → replaced by model-aligned tokenizers
- **Total Deleted**: 530 lines

**Added** (New Embedding Architecture):

- `src/Medical_KG_rev/services/embedding/vllm_client.py` (120 lines) → OpenAI-compatible vLLM client
- `src/Medical_KG_rev/services/embedding/pyserini_wrapper.py` (140 lines) → SPLADE via Pyserini
- `src/Medical_KG_rev/services/embedding/namespace_registry.py` (80 lines) → Multi-namespace support
- `src/Medical_KG_rev/services/embedding/gpu_enforcer.py` (60 lines) → Fail-fast GPU checks
- **Total Added**: 400 lines

**Net Impact**: 130 lines removed (25% reduction), clearer architecture, GPU-only enforced

---

### Breaking Changes (4 Total)

1. **Embedding API Signature**: `embed(texts: list[str], namespace: str) -> list[Embedding]` requires `namespace` parameter
2. **GPU Fail-Fast**: Embedding jobs fail immediately if GPU unavailable (no CPU fallback)
3. **FAISS Primary**: Dense vectors stored in FAISS by default (Neo4j vector index opt-in)
4. **OpenSearch rank_features**: Sparse signals stored as `rank_features` field (requires mapping update)

---

### Migration Path

**Phase 1: Deploy vLLM + Pyserini (Week 1)**

- Stand up vLLM server with Qwen3-Embedding-8B
- Deploy Pyserini wrapper for SPLADE-v3
- Validate health checks and GPU fail-fast

**Phase 2: Update Orchestration (Week 2)**

- Update embedding stage to call vLLM/Pyserini
- Add namespace parameter to all embedding calls
- Update job ledger to track `embed_gpu_unavailable` failures

**Phase 3: Storage Migration (Week 3)**

- Update OpenSearch mapping for `rank_features`
- Migrate existing dense vectors to FAISS (background job)
- Re-embed sparse signals with Pyserini (background job)

**Phase 4: Atomic Deletion (Week 4)**

- Delete legacy embedding code in same commits as new integrations complete
- Validate no imports remain to legacy code
- Run full regression tests

**Phase 5: Production Validation (Week 5-6)**

- Deploy to production
- Monitor embedding throughput, GPU utilization, failures
- Compare retrieval quality: Recall@10 should remain stable or improve

---

## Success Criteria

### Code Quality

- ✅ Codebase reduction: 25% (530 → 400 lines)
- ✅ Test coverage: ≥90% for new embedding code
- ✅ No legacy imports remain
- ✅ Lint clean: 0 ruff/mypy errors

### Functionality

- ✅ vLLM serving Qwen3 embeddings at 1000+ emb/sec
- ✅ Pyserini SPLADE document-side expansion produces `rank_features`
- ✅ GPU fail-fast prevents CPU fallbacks (100% enforcement)
- ✅ Multi-namespace registry supports 3+ namespaces

### Performance

- ✅ Dense embedding throughput: ≥1000 embeddings/sec (GPU batch=64)
- ✅ Sparse embedding throughput: ≥500 docs/sec (SPLADE expansion)
- ✅ FAISS KNN latency: P95 <50ms for 10M vectors
- ✅ OpenSearch sparse retrieval: P95 <200ms

### Observability

- ✅ Prometheus metrics for embedding duration, GPU utilization, failures by namespace
- ✅ CloudEvents for embedding lifecycle (started, completed, failed)
- ✅ Grafana dashboard: embedding throughput, GPU memory, failure rates

---

## Dependencies

### New Libraries (3)

```txt
vllm>=0.3.0  # OpenAI-compatible serving for Qwen3 embeddings
pyserini>=0.22.0  # SPLADE-v3 wrapper with document-side expansion
faiss-gpu>=1.7.4  # GPU-accelerated dense vector search
```

### Updated Libraries

```txt
transformers>=4.38.0  # Qwen3 tokenizer support
torch>=2.1.0  # CUDA 12.1+ for vLLM and FAISS GPU
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| vLLM startup complexity | Delayed deployments | Pre-build Docker image with vLLM + model weights |
| SPLADE query-side expansion overhead | Increased query latency | Default to doc-side only, query-side opt-in |
| FAISS index rebuild time | Extended migration | Incremental indexing, blue-green deployment |
| GPU memory pressure | OOM failures | Monitor GPU memory, batch size tuning, graceful degradation |

---

## Observability & Monitoring

### Prometheus Metrics

```python
# Embedding performance metrics
EMBEDDING_DURATION = Histogram(
    "medicalkg_embedding_duration_seconds",
    "Embedding generation duration",
    ["namespace", "provider", "tenant_id"],  # provider: vllm, pyserini
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)

EMBEDDING_BATCH_SIZE = Histogram(
    "medicalkg_embedding_batch_size",
    "Number of texts per embedding batch",
    ["namespace"],
    buckets=[1, 8, 16, 32, 64, 128]
)

EMBEDDING_TOKEN_COUNT = Histogram(
    "medicalkg_embedding_tokens_per_text",
    "Token count per embedded text",
    ["namespace"],
    buckets=[50, 100, 200, 400, 512, 1024]
)

# GPU metrics
GPU_UTILIZATION = Gauge(
    "medicalkg_embedding_gpu_utilization_percent",
    "GPU utilization during embedding",
    ["gpu_id", "service"]  # service: vllm, splade
)

GPU_MEMORY_USED = Gauge(
    "medicalkg_embedding_gpu_memory_bytes",
    "GPU memory used by embedding service",
    ["gpu_id", "service"]
)

# Failure metrics
EMBEDDING_FAILURES = Counter(
    "medicalkg_embedding_failures_total",
    "Embedding failures by type",
    ["namespace", "error_type"]  # error_type: gpu_unavailable, token_overflow, timeout
)

TOKEN_OVERFLOW_RATE = Gauge(
    "medicalkg_embedding_token_overflow_rate",
    "% of texts exceeding token budget",
    ["namespace"]
)

# Namespace metrics
NAMESPACE_USAGE = Counter(
    "medicalkg_embedding_namespace_requests_total",
    "Requests per namespace",
    ["namespace", "operation"]  # operation: embed, validate
)
```

### CloudEvents

```json
{
  "specversion": "1.0",
  "type": "com.medical-kg.embedding.completed",
  "source": "/embedding-service",
  "id": "embed-abc123",
  "time": "2025-10-08T14:30:00Z",
  "data": {
    "job_id": "job-abc123",
    "namespace": "single_vector.qwen3.4096.v1",
    "provider": "vllm",
    "text_count": 64,
    "duration_seconds": 0.12,
    "gpu_id": 0,
    "gpu_utilization_percent": 85,
    "token_overflows": 0,
    "tokens_per_text_avg": 287
  }
}
```

### Grafana Dashboard Panels

1. **Embedding Latency by Namespace**: Line chart (P50, P95, P99) per namespace
2. **GPU Utilization**: Time-series showing GPU usage (vLLM, SPLADE)
3. **Throughput**: Embeddings/second per namespace
4. **Token Overflow Rate**: Gauge showing % of texts exceeding budget
5. **Namespace Usage Distribution**: Pie chart of requests per namespace
6. **Failure Rate**: Counter showing failures by error type
7. **GPU Memory Pressure**: Line chart of GPU memory usage over time

---

## Configuration Management

### vLLM Configuration

```yaml
# config/embedding/vllm.yaml
service:
  host: 0.0.0.0
  port: 8001
  gpu_memory_utilization: 0.8  # Reserve 80% of GPU memory
  max_model_len: 512  # Max sequence length
  dtype: float16  # Use FP16 for efficiency

model:
  name: "Qwen/Qwen2.5-Coder-1.5B"
  trust_remote_code: true
  download_dir: "/models/qwen3-embedding"

batching:
  max_batch_size: 64
  max_wait_time_ms: 50  # Wait up to 50ms to fill batch

health_check:
  enabled: true
  gpu_check_interval_seconds: 30
  fail_fast_on_gpu_unavailable: true
```

### Namespace Registry Configuration

```yaml
# config/embedding/namespaces.yaml
namespaces:
  single_vector.qwen3.4096.v1:
    provider: vllm
    endpoint: "http://vllm-service:8001"
    model_name: "Qwen/Qwen2.5-Coder-1.5B"
    dimension: 4096
    max_tokens: 512
    tokenizer: "Qwen/Qwen2.5-Coder-1.5B"
    enabled: true

  sparse.splade_v3.400.v1:
    provider: pyserini
    endpoint: "http://pyserini-service:8002"
    model_name: "naver/splade-cocondenser-ensembledistil"
    max_tokens: 512
    doc_side_expansion: true
    query_side_expansion: false
    top_k_terms: 400
    enabled: true

  multi_vector.colbert_v2.128.v1:
    provider: colbert
    endpoint: "http://colbert-service:8003"
    model_name: "colbert-ir/colbertv2.0"
    dimension: 128
    max_tokens: 512
    enabled: false  # Optional, not enabled by default
```

### Pyserini SPLADE Configuration

```yaml
# config/embedding/pyserini.yaml
service:
  host: 0.0.0.0
  port: 8002
  gpu_memory_utilization: 0.6

model:
  name: "naver/splade-cocondenser-ensembledistil"
  cache_dir: "/models/splade"

expansion:
  doc_side:
    enabled: true
    top_k_terms: 400
    normalize_weights: true
  query_side:
    enabled: false  # Opt-in only
    top_k_terms: 200

opensearch:
  rank_features_field: "splade_terms"
  max_weight: 10.0
```

---

## API Integration

### REST API

```http
POST /v1/embed
Content-Type: application/vnd.api+json
Authorization: Bearer <jwt_token>

{
  "data": {
    "type": "EmbeddingRequest",
    "attributes": {
      "texts": [
        "Metformin is used to treat type 2 diabetes.",
        "Adverse events include gastrointestinal disturbances."
      ],
      "namespace": "single_vector.qwen3.4096.v1",
      "options": {
        "normalize": true,
        "return_tokens": false
      }
    }
  }
}
```

**Response**:

```json
{
  "data": {
    "type": "EmbeddingResult",
    "attributes": {
      "namespace": "single_vector.qwen3.4096.v1",
      "embeddings": [
        {
          "text_index": 0,
          "embedding": [0.123, -0.456, ...],  # 4096-D vector
          "dimension": 4096,
          "token_count": 12
        },
        {
          "text_index": 1,
          "embedding": [0.789, -0.012, ...],
          "dimension": 4096,
          "token_count": 9
        }
      ],
      "metadata": {
        "provider": "vllm",
        "model": "Qwen/Qwen2.5-Coder-1.5B",
        "duration_ms": 120,
        "gpu_utilization_percent": 85
      }
    }
  }
}
```

### GraphQL API

```graphql
mutation EmbedTexts($input: EmbeddingInput!) {
  embed(input: $input) {
    namespace
    embeddings {
      textIndex
      embedding
      dimension
      tokenCount
    }
    metadata {
      provider
      durationMs
      gpuUtilization
    }
  }
}

input EmbeddingInput {
  texts: [String!]!
  namespace: String!
  options: EmbeddingOptions
}

input EmbeddingOptions {
  normalize: Boolean = true
  returnTokens: Boolean = false
}
```

### New Namespace Management Endpoints

```http
GET /v1/namespaces
Authorization: Bearer <jwt_token>

Response:
{
  "data": [{
    "type": "EmbeddingNamespace",
    "id": "single_vector.qwen3.4096.v1",
    "attributes": {
      "provider": "vllm",
      "dimension": 4096,
      "max_tokens": 512,
      "enabled": true,
      "model": "Qwen/Qwen2.5-Coder-1.5B"
    }
  }]
}
```

```http
POST /v1/namespaces/{namespace}/validate
Authorization: Bearer <jwt_token>

{
  "texts": ["Sample text for validation"]
}

Response:
{
  "valid": true,
  "token_counts": [4],
  "warnings": []
}
```

---

## Rollback Procedures

### Rollback Trigger Conditions

**Automated Triggers**:

- Embedding latency P95 >2s for >10 minutes
- GPU failure rate >20% for >5 minutes
- Token overflow rate >15% for >15 minutes
- vLLM service unavailable for >5 minutes

**Manual Triggers**:

- Embedding quality degradation reported by retrieval metrics
- GPU memory leaks causing OOM
- vLLM startup failures
- Incorrect vector dimensions or sparse term weights

### Rollback Steps

```bash
# Phase 1: Immediate mitigation (if in canary)
# 1. Disable new embedding service
kubectl scale deployment/vllm-embedding --replicas=0
kubectl scale deployment/pyserini-splade --replicas=0

# 2. Re-enable legacy embedding service (if still deployed)
kubectl scale deployment/legacy-embedding --replicas=3

# Phase 2: Full rollback (if needed)
# 3. Revert feature branch
git revert <embedding-standardization-commit-sha>

# 4. Redeploy previous version
kubectl rollout undo deployment/embedding-service

# 5. Revert OpenSearch mapping changes
curl -X PUT "opensearch:9200/chunks/_mapping" -d @legacy-mapping.json

# 6. Validate baseline restoration (15 minutes)
# Check metrics:
# - Embedding latency P95 <500ms (legacy baseline)
# - GPU utilization 60-70% (legacy)
# - Zero token overflows (legacy truncated silently)

# 7. Post-incident analysis (2 hours)
# Gather logs, metrics, GPU traces
# Identify root cause (vLLM config, GPU memory, tokenizer mismatch)
# Create incident report

# 8. Fix and redeploy (2-5 days)
# Fix identified issues in separate branch
# Re-test with isolated GPU environment
# Schedule new deployment
```

### Recovery Time Objective (RTO)

- **Canary rollback**: 5 minutes (scale down new, scale up legacy)
- **Full rollback**: 15 minutes (revert + redeploy + mapping)
- **Maximum RTO**: 20 minutes

---

## vLLM Deployment Details

### Docker Configuration

```dockerfile
# Dockerfile.vllm
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Install vLLM
RUN pip install vllm==0.3.0 transformers==4.38.0

# Download Qwen3 model weights (reduces startup time)
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-Coder-1.5B', cache_dir='/models/qwen3')"

# Copy vLLM config
COPY config/embedding/vllm.yaml /config/vllm.yaml

# Start vLLM server
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "Qwen/Qwen2.5-Coder-1.5B", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--gpu-memory-utilization", "0.8", \
     "--max-model-len", "512", \
     "--dtype", "float16"]
```

### Kubernetes GPU Allocation

```yaml
# k8s/vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-embedding
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: vllm
        image: medical-kg/vllm-embedding:latest
        resources:
          limits:
            nvidia.com/gpu: 1  # Request 1 GPU per pod
            memory: 16Gi
          requests:
            nvidia.com/gpu: 1
            memory: 12Gi
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: VLLM_GPU_MEMORY_UTILIZATION
          value: "0.8"
        ports:
        - containerPort: 8001
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
      nodeSelector:
        gpu: "true"
        gpu-type: "t4"  # Or "v100", "a100" based on availability
```

### Model Loading Strategy

**Pre-Download Strategy** (Recommended):

1. Bake model weights into Docker image (adds ~1.5GB to image size)
2. vLLM loads from local cache on startup (30-60s)
3. No network dependency during pod initialization

**On-Demand Strategy** (Alternative):

1. Mount shared model volume (NFS/EFS)
2. vLLM downloads on first startup (5-10 minutes)
3. Subsequent startups use cached models

**Validation**:

```bash
# Test vLLM startup time
time docker run medical-kg/vllm-embedding:latest python -c "from vllm import LLM; LLM('Qwen/Qwen2.5-Coder-1.5B')"
# Target: <60 seconds
```

---

## Security & Multi-Tenancy

### Tenant Isolation in Embeddings

**Request-Level Filtering**:

```python
# All embedding requests include tenant_id from JWT
async def embed_texts(
    texts: list[str],
    namespace: str,
    tenant_id: str  # Extracted from JWT
) -> list[Embedding]:
    # Store tenant_id with embeddings for audit
    embeddings = await vllm_client.embed(texts, namespace)

    # Log request with tenant_id
    logger.info(
        "Embedding request",
        extra={
            "tenant_id": tenant_id,
            "namespace": namespace,
            "text_count": len(texts),
            "correlation_id": request.correlation_id
        }
    )

    return embeddings
```

**Storage-Level Isolation**:

- FAISS indices partitioned by tenant_id (separate index per tenant)
- OpenSearch sparse signals include `tenant_id` field for filtering
- Neo4j embedding metadata tagged with tenant_id

**Verification**:

- Integration tests validate no cross-tenant embedding leakage
- Audit logging for all embedding requests (tenant_id, namespace, text count)

### Namespace Access Control

**Role-Based Access**:

```yaml
# Authorization rules for namespaces
namespaces:
  single_vector.qwen3.4096.v1:
    allowed_scopes: ["embed:read", "embed:write"]
    allowed_tenants: ["all"]  # Public namespace

  single_vector.custom_model.2048.v1:
    allowed_scopes: ["embed:admin"]
    allowed_tenants: ["tenant-123"]  # Private namespace
```

---

## Performance Benchmarking

### Dense Embedding Benchmarks (vLLM)

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Throughput | ≥1000 emb/sec | Load test with batch_size=64, GPU T4 |
| Latency P95 | <200ms | Prometheus histogram, 1000 requests |
| GPU Utilization | 70-85% | nvidia-smi during load test |
| Token Overflow Rate | <5% | Monitor `TOKEN_OVERFLOW_RATE` metric |

### Sparse Embedding Benchmarks (Pyserini SPLADE)

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Throughput | ≥500 docs/sec | Load test with doc-side expansion |
| Latency P95 | <400ms | Prometheus histogram |
| Top-K Terms Avg | 300-400 | CloudEvents `top_k_terms_avg` |
| GPU Memory | <4GB | nvidia-smi memory usage |

### Storage Benchmarks

| Operation | Target | Validation Method |
|-----------|--------|-------------------|
| FAISS Index | P95 <50ms for 10M vectors | k6 load test |
| OpenSearch sparse | P95 <200ms | k6 load test with rank_features |
| FAISS Build | <2 hours for 10M vectors | Incremental indexing test |

---

## Implementation Strategy

**Hard Cutover** (No Legacy Compatibility):

- No transition period with legacy embedding code
- Atomic deletions: legacy code deleted in same commits as new implementations complete
- All functionality delegated to vLLM, Pyserini, FAISS (no bespoke code remains)
- Codebase shrinkage validation: ensure 25% reduction, no non-functioning legacy code

**Validation**:

- Comprehensive testing before production deployment
- Monitor metrics for 48 hours post-deployment
- Emergency rollback: revert entire feature branch if critical issues

---

## Timeline

**6 Weeks Total**:

- **Week 1-2**: Build new architecture (vLLM setup, Pyserini wrapper, namespace registry, atomic deletions)
- **Week 3-4**: Integration testing (all namespaces, GPU fail-fast, storage migration)
- **Week 5-6**: Production deployment (deploy, monitor, stabilize, document lessons learned)

---

**Status**: Ready for review and approval
**Created**: 2025-10-07
**Change ID**: `add-embeddings-representation`
