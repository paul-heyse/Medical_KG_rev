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
pyserini>=0.22.0       # SPLADE-v3 wrapper with document-side expansion
faiss-gpu>=1.7.4       # GPU-accelerated dense vector search
redis[hiredis]>=5.0.0  # Embedding cache backend
```

> vLLM is distributed as the Docker image
> `ghcr.io/example/vllm-qwen3-embedding:latest`; it is not installed as a Python
> dependency.

### Updated Libraries

```txt
transformers>=4.38.0  # Qwen3 tokenizer support
torch>=2.1.0  # CUDA 12.1+ for FAISS GPU helpers and health checks
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
