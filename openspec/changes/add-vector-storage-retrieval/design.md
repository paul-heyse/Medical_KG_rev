# Design: Add Vector Storage & Retrieval System

## Context

The modular document retrieval pipeline requires a high-performance, flexible vector storage and retrieval system to support diverse embedding strategies (dense, sparse, learned-sparse, multi-vector) at billion-vector scale for biomedical knowledge integration.

### Stakeholders

- **ML engineers** deploying and tuning vector indexes for production retrieval
- **Research teams** evaluating compression methods and hybrid retrieval strategies
- **Infrastructure teams** managing vector store deployments and backups
- **Data scientists** analyzing retrieval performance and accuracy

### Constraints

- **English-first optimization**: All analyzers, tokenizers, and tuning optimized for English biomedical text
- **Text-only**: No multimodal content (images, formulas) support
- **Local-first**: All backends must be self-hostable (no cloud-only services)
- **GPU optional**: CPU fallbacks required; GPU only for acceleration
- **Namespace isolation**: Strict dimension governance per namespace to prevent errors

### Goals

1. **Universal interface**: Single `VectorStorePort` protocol supporting 10+ backends with swap-via-configuration
2. **Advanced compression**: 4-40× memory reduction via int8, PQ, OPQ, binary quantization with tunable accuracy trade-offs
3. **Billion-vector scale**: DiskANN and GPU support for massive collections (>1B vectors)
4. **Hybrid retrieval**: Multi-strategy orchestration (dense + BM25 + SPLADE) with fusion ranking
5. **Production-ready**: <100ms P95 latency, snapshot/backup, health checks, metrics

### Non-Goals

- Multimodal embeddings (images, audio)
- Multilingual optimization (English-only for phase 1)
- Cloud-managed vector stores (Pinecone, Zilliz Cloud)
- Graph-structured retrieval (focus on flat vector search)

## Decisions

### Decision 1: Universal Adapter Interface (`VectorStorePort`)

**Rationale**: Decouple storage backends from retrieval logic via a unified protocol. This enables:

- Adding new backends without modifying ingestion/retrieval services
- Swapping backends via YAML configuration for A/B testing
- Per-namespace backend selection (e.g., Qdrant for dense, OpenSearch for sparse)

**Interface**:

```python
# med/ports/vector_store.py
class VectorStorePort(Protocol):
    def create_or_update_collection(self, namespace: str, params: IndexParams) -> None:
        """Create or update index configuration for namespace."""

    def upsert(self, namespace: str, ids: Sequence[str], vectors: Sequence[list[float]], payloads: Sequence[dict]) -> None:
        """Insert or update vectors with metadata."""

    def knn(self, namespace: str, query_vec: list[float], top_k: int, filters: dict | None = None) -> list[dict]:
        """K-nearest neighbor search with optional metadata filters."""

    def delete(self, namespace: str, ids: Sequence[str]) -> None:
        """Delete vectors by ID."""
```

**Alternatives considered**:

- **Backend-specific APIs**: Rejected due to tight coupling and difficult migration
- **Single backend (Qdrant-only)**: Rejected as limiting for research and performance optimization

**Trade-offs**:

- **Pro**: Maximum flexibility, backend independence, testability
- **Con**: Adapter layer overhead (negligible vs actual vector operations)

---

### Decision 2: Unified Compression Policy

**Rationale**: Different backends support different compression methods (Qdrant: int8/BQ, FAISS: PQ/OPQ, OpenSearch: fp16/PQ). A unified `CompressionPolicy` model enables consistent configuration across backends.

**Model**:

```python
class CompressionPolicy(BaseModel):
    kind: str | None = None  # None | "scalar_int8" | "fp16" | "pq" | "opq_pq" | "binary"
    pq_m: int | None = None        # PQ subvector count
    pq_nbits: int | None = None    # PQ codebook bits (4 or 8)
    opq_m: int | None = None       # OPQ rotation blocks
    notes: dict[str, Any] = {}
```

**Backend translation**:

| Compression | Qdrant | FAISS | Milvus | OpenSearch |
|-------------|--------|-------|--------|------------|
| `scalar_int8` | Scalar quantization | SQ8 | int8 quantizer | encoder: sq |
| `fp16` | N/A (not supported) | fp16 index | N/A | encoder: sq (fp16) |
| `pq` | N/A | IVF_PQ | IVF_PQ | encoder: pq |
| `opq_pq` | N/A | OPQ+IVF_PQ | OPQ upstream | N/A |
| `binary` | Binary quantization | N/A | N/A | N/A |

**Two-stage search** (for PQ/BQ):

```yaml
compression:
  kind: "pq"
  pq_m: 64
  pq_nbits: 8
search:
  reorder_final: true  # Retrieve R by compressed distance, re-score by float32
```

**Alternatives considered**:

- **Backend-specific compression config**: Rejected as requiring duplicate configuration
- **Automatic compression selection**: Rejected as insufficient control for tuning

**Trade-offs**:

- **Pro**: Consistent config, easy A/B testing, clear trade-offs
- **Con**: Not all compression types available in all backends (documented in adapter)

---

### Decision 3: Per-Namespace Index Configuration

**Rationale**: Different embedding models require different index parameters:

- Dense 384D BGE-small: HNSW with int8 compression (fast, accurate)
- Dense 1536D Qwen-3: IVF_PQ with OPQ (memory-constrained)
- Multi-vector ColBERT: Store-only (no index), used for reranking

**Configuration**:

```yaml
vector_store:
  driver: qdrant
  collections:
    dense.bge.384.v1:
      kind: hnsw
      dim: 384
      metric: cosine
      m: 64
      ef_construct: 400
      ef_search: 128
      compression: { kind: "scalar_int8" }
    dense.qwen.1536.v1:
      kind: ivf_pq
      dim: 1536
      metric: ip
      nlist: 32768
      nprobe: 64
      compression: { kind: "opq_pq", opq_m: 64, pq_m: 64, pq_nbits: 8 }
    multi.colbertv2.128.v1:
      kind: none  # store-only for late-interaction
      dim: 128
```

**Dimension validation**:

```python
def _validate_dimensions(namespace: str, vectors: Sequence[list[float]]) -> None:
    """Validate vector dimensions match namespace config."""
    expected_dim = get_namespace_dim(namespace)
    for vec in vectors:
        if len(vec) != expected_dim:
            raise DimensionMismatchError(
                f"Namespace {namespace} expects {expected_dim}D vectors, got {len(vec)}D"
            )
```

**Alternatives considered**:

- **Global index config**: Rejected as insufficient for diverse embedding models
- **Auto-detect dimensions**: Rejected as unsafe (silent errors)

**Trade-offs**:

- **Pro**: Type safety, prevents dimension drift, optimal parameters per model
- **Con**: Requires explicit namespace configuration (mitigated by clear examples)

---

### Decision 4: Multi-Strategy Retrieval Orchestration

**Rationale**: Single retrieval strategy (e.g., dense-only) is insufficient for biomedical text:

- **Dense vectors**: Capture semantic similarity but miss exact terminology
- **BM25**: Captures exact terms but misses paraphrases
- **SPLADE**: Learned sparse balances both but requires tuning

Hybrid retrieval combining strategies improves nDCG@10 by 10-15% vs single-strategy.

**Architecture**:

```python
class RetrievalRouter:
    def __init__(
        self,
        dense_stores: dict[str, VectorStorePort],
        sparse_stores: dict[str, SparseRetriever],
        fusion_method: str = "rrf",
    ):
        self.dense = dense_stores
        self.sparse = sparse_stores
        self.fusion = fusion_method

    async def retrieve(
        self,
        query: str,
        strategies: list[str],  # ["dense", "bm25", "splade"]
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[ScoredDocument]:
        # Parallel fan-out
        tasks = []
        if "dense" in strategies:
            tasks.append(self._dense_search(query, filters))
        if "bm25" in strategies:
            tasks.append(self._bm25_search(query, filters))
        if "splade" in strategies:
            tasks.append(self._splade_search(query, filters))

        results = await asyncio.gather(*tasks)

        # Fusion
        if self.fusion == "rrf":
            fused = self._reciprocal_rank_fusion(results)
        elif self.fusion == "weighted":
            fused = self._weighted_fusion(results, weights=self.weights)

        return fused[:top_k]
```

**Fusion methods**:

1. **Weighted linear**: `score = w1*dense_score + w2*bm25_score + w3*splade_score`
   - Requires tuning weights
   - Good when scores are calibrated

2. **RRF (Reciprocal Rank Fusion)**: `score = Σ(1 / (rank_i + k))`
   - Parameter-free (k=60 typical)
   - Robust to score distribution differences
   - **Recommended as default**

**Alternatives considered**:

- **Single-strategy**: Rejected due to lower accuracy
- **Sequential retrieval**: Rejected due to latency (parallel 3× faster)
- **LLM-based fusion**: Rejected as too slow and GPU-intensive

**Trade-offs**:

- **Pro**: Higher accuracy, robust to query variation
- **Con**: Increased latency (mitigated by parallel execution)

---

### Decision 5: Backend Selection Strategy

**Rationale**: Different backends excel at different scenarios. Provide clear guidance:

| Backend | Best For | Pros | Cons |
|---------|----------|------|------|
| **Qdrant** | Default dense storage | Easy filtering, snapshots, int8/BQ, GPU indexing | Limited PQ support |
| **FAISS** | Memory-constrained | Full PQ/OPQ control, GPU variants | Manual payload management |
| **Milvus** | GPU acceleration | GPU IVF/PQ/CAGRA, DiskANN | Operational complexity |
| **OpenSearch** | Single-engine hybrid | BM25 + SPLADE + kNN in one index | Less mature vector support |
| **Weaviate** | Built-in hybrid | Integrated BM25f fusion | Less compression options |
| **DiskANN** | Billion-vector scale | SSD-resident, low RAM | Query latency variability |
| **pgvector** | PostgreSQL integration | SQL filters, familiar tooling | Limited index types |

**Default selection**:

- **Dense (default)**: Qdrant (HNSW + int8)
- **Sparse**: OpenSearch (BM25 + SPLADE rank_features)
- **GPU-accelerated**: Milvus or FAISS GPU
- **Memory-constrained**: FAISS OPQ+IVF_PQ
- **Massive scale**: DiskANN or Milvus DiskANN

**Configuration**:

```yaml
vector_store:
  driver: qdrant  # Default
  # driver: faiss  # Switch for PQ/OPQ
  # driver: milvus  # Switch for GPU IVF/CAGRA
  # driver: diskann  # Switch for billion-vector scale
```

**Alternatives considered**:

- **Single backend only**: Rejected as limiting for research and optimization
- **Automatic backend selection**: Rejected as requiring complex heuristics

**Trade-offs**:

- **Pro**: Flexibility for diverse scenarios, clear guidance
- **Con**: More backends to maintain (mitigated by optional imports)

---

### Decision 6: GPU Integration Strategy

**Rationale**: GPUs dramatically accelerate indexing (3-10×) and search for large collections, but not all deployments have GPUs. Strategy:

- **GPU-gated operations**: Fail-fast if GPU required but unavailable
- **CPU fallbacks**: Provide CPU alternatives for non-critical paths
- **Batch processing**: Amortize GPU overhead across multiple operations

**Implementation**:

```python
class GPUManager:
    def __init__(self, require_gpu: bool = False):
        self.gpu_available = torch.cuda.is_available()
        if require_gpu and not self.gpu_available:
            raise GPUUnavailableError("GPU required but not available")

    def get_device(self) -> str:
        return "cuda:0" if self.gpu_available else "cpu"

# Qdrant: GPU-accelerated indexing
if gpu_manager.gpu_available and config.gpu_indexing:
    client.create_collection(
        ...,
        optimizers_config=OptimizersConfig(indexing_threshold=0),  # Immediate GPU indexing
    )

# FAISS: GPU index
if gpu_manager.gpu_available:
    index = faiss.GpuIndexFlatL2(resource, dim)
else:
    index = faiss.IndexFlatL2(dim)
```

**GPU fail-fast scenarios**:

- Embedding generation (already handled by `EmbeddingService`)
- SPLADE encoding (dense model on GPU)
- Optional: GPU-accelerated indexing (Qdrant, Milvus, FAISS)

**Alternatives considered**:

- **GPU-only**: Rejected as limiting deployment flexibility
- **Silent CPU fallback**: Rejected as hiding performance degradation

**Trade-offs**:

- **Pro**: Clear expectations, faster when GPU available
- **Con**: Deployment complexity (mitigated by container images)

---

### Decision 7: Compression Evaluation Harness

**Rationale**: Compression methods trade accuracy for memory/latency. Evaluation harness enables data-driven selection.

**Metrics**:

1. **Recall@K**: % of true nearest neighbors in top-K results
2. **nDCG@K**: Normalized discounted cumulative gain (rank-aware)
3. **Memory reduction**: Compressed size / original size
4. **Query latency**: P50, P95, P99 in milliseconds
5. **Index build time**: For compression methods requiring training

**Evaluation runner**:

```python
class CompressionEvaluator:
    def evaluate_compression(
        self,
        baseline: VectorStorePort,  # float32 exact
        compressed: VectorStorePort,  # with compression
        queries: list[Query],
        ground_truth: list[list[str]],
    ) -> CompressionMetrics:
        # Retrieve with both
        baseline_results = [baseline.knn(...) for q in queries]
        compressed_results = [compressed.knn(...) for q in queries]

        # Compute metrics
        recall_10 = compute_recall_at_k(compressed_results, ground_truth, k=10)
        ndcg_10 = compute_ndcg_at_k(compressed_results, ground_truth, k=10)
        memory_ratio = compressed.memory_usage() / baseline.memory_usage()
        latency_p95 = np.percentile(compressed_latencies, 95)

        return CompressionMetrics(
            recall_10=recall_10,
            ndcg_10=ndcg_10,
            memory_reduction=1.0 / memory_ratio,
            latency_p95=latency_p95,
        )
```

**Leaderboard**:

```
| Backend | Compression | Recall@10 | nDCG@10 | Memory Reduction | P95 Latency |
|---------|-------------|-----------|---------|------------------|-------------|
| Qdrant  | None        | 100.0%    | 1.000   | 1.0×             | 25ms        |
| Qdrant  | int8        | 99.5%     | 0.998   | 4.0×             | 22ms        |
| Qdrant  | BQ          | 95.2%     | 0.975   | 32.0×            | 8ms         |
| FAISS   | PQ64x8      | 97.8%     | 0.985   | 16.0×            | 18ms        |
| FAISS   | OPQ+PQ64x8  | 98.5%     | 0.992   | 16.0×            | 20ms        |
```

**Alternatives considered**:

- **Manual tuning**: Rejected as subjective and time-consuming
- **End-to-end eval only**: Rejected as insufficient for isolation

**Trade-offs**:

- **Pro**: Data-driven decisions, reproducible benchmarks
- **Con**: Requires test queries and ground truth (one-time cost)

---

## Risks / Trade-offs

### Risk 1: Backend Proliferation Increases Maintenance

**Mitigation**:

- Unified `VectorStorePort` interface limits coupling
- Optional adapters via lazy imports (only load what's configured)
- Core focus on 3 backends: Qdrant (default), FAISS (PQ), OpenSearch (hybrid)
- Others marked as "community-supported" with lower maintenance priority

### Risk 2: Compression Tuning Requires Expertise

**Mitigation**:

- Provide tuned defaults per backend (documented in adapter)
- Evaluation harness automates A/B testing
- Clear documentation of trade-offs (recall vs memory vs latency)
- Start with int8 (simple, safe 4× reduction)

### Risk 3: GPU Unavailability Limits Performance

**Mitigation**:

- CPU fallbacks for all critical paths
- Fail-fast for GPU-only operations with clear errors
- Document GPU requirements and benefits
- Container images with CUDA pre-installed

### Risk 4: Multi-Strategy Retrieval Increases Latency

**Mitigation**:

- Parallel fan-out (3 strategies in ~same time as 1)
- Caching for repeated queries
- Adjustable top-K per strategy (retrieve 50 each, fuse to 10)
- Single-strategy mode for latency-critical scenarios

### Risk 5: Dimension Mismatch Causes Silent Errors

**Mitigation**:

- Strict dimension validation at upsert and query time
- Per-namespace dimension registry
- Clear error messages with namespace and expected/actual dims
- Integration tests for all embedding models

### Risk 6: IVF Indexes Require Retraining

**Mitigation**:

- Monitor data distribution drift
- Schedule periodic retraining (weekly/monthly)
- Hot-swap indexes (train new, switch atomically)
- Document retraining triggers and procedures

---

## Migration Plan

### Phase 1: Core Infrastructure (Weeks 1-2)

1. Implement `VectorStorePort` interface and models
2. Create Qdrant adapter (default dense)
3. Create OpenSearch adapter (BM25 + SPLADE)
4. Implement `RetrievalRouter` with RRF fusion
5. Add dimension validation and namespace registry

### Phase 2: Compression & Advanced Backends (Weeks 3-4)

1. Add compression support to Qdrant (int8, BQ)
2. Implement FAISS adapter (IVF_PQ, OPQ)
3. Implement Milvus adapter (GPU IVF/PQ)
4. Add two-stage search (compressed + reorder)
5. Create compression evaluation harness

### Phase 3: Additional Backends & GPU (Week 5)

1. Implement remaining backends (pgvector, Weaviate, DiskANN, embedded libs)
2. Add GPU-accelerated indexing (Qdrant, FAISS, Milvus)
3. Implement GPU memory monitoring
4. Add batch processing optimizations

### Phase 4: Integration & Testing (Weeks 6-7)

1. Integrate with `IngestionService` for automatic vector writing
2. Extend `RetrievalService` to use `RetrievalRouter`
3. Create end-to-end tests for all backends
4. Run compression benchmarks and create leaderboard
5. Document tuning guidelines

### Rollback Plan

- New system; no migration required
- Can run in parallel with existing retrieval (if any)
- Per-namespace rollout enables gradual adoption
- Disable via `vector_store.enabled: false` in config

---

## Open Questions

1. **Should we support cross-namespace queries (search multiple namespaces)?**
   - Defer to Phase 2; add if use case emerges

2. **What is optimal balance between parallelism and resource usage?**
   - Start with 3 concurrent strategies; tune based on profiling

3. **Should we cache query results for repeated queries?**
   - Yes, add TTL-based cache for hot queries (Redis)

4. **How to handle index versioning (reindex on embedding model update)?**
   - Use namespace versioning: `dense.bge.1024.v1` → `dense.bge.1024.v2`

5. **Should we support approximate filters (filter after KNN vs during)?**
   - Prefer during-KNN filtering (exact); add approximate if performance requires

---

## References

### Academic Papers

- Malkov, Y., & Yashunin, D. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. *IEEE TPAMI*.
- Jégou, H., Douze, M., & Schmid, C. (2011). Product quantization for nearest neighbor search. *IEEE TPAMI*.
- Ge, T., et al. (2014). Optimized product quantization for approximate nearest neighbor search. *CVPR*.
- Subramanya, S. J., et al. (2019). DiskANN: Fast accurate billion-point nearest neighbor search on a single node. *NeurIPS*.

### Technical Documentation

- Qdrant: <https://qdrant.tech/documentation/>
- FAISS: <https://github.com/facebookresearch/faiss/wiki>
- Milvus: <https://milvus.io/docs>
- OpenSearch k-NN: <https://opensearch.org/docs/latest/search-plugins/knn/>
- Weaviate: <https://weaviate.io/developers/weaviate>
- Vespa: <https://docs.vespa.ai/>
- pgvector: <https://github.com/pgvector/pgvector>

### Internal Documents

- `1) docs/Vector_Storage_and_Retrieval.md` - Comprehensive vector storage methodology catalogue
- `1) docs/Modular Document Retrieval Pipeline – Design & Scaffold.pdf` - System architecture
- Research report: *Local Vector Retrieval and Reranking Systems for Biomedical Document Pipelines*
