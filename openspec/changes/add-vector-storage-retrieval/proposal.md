# Proposal: Add Vector Storage & Retrieval System

## Why

The modular document retrieval pipeline requires a sophisticated, high-performance vector storage and retrieval system to support multiple embedding strategies (dense, sparse, learned-sparse, multi-vector) with advanced compression techniques for production-scale biomedical knowledge integration.

**Current limitations**:

- No unified interface for diverse vector storage backends (Qdrant, FAISS, Milvus, OpenSearch k-NN)
- Missing support for advanced compression (PQ/OPQ, scalar quantization, binary quantization)
- No orchestration for multi-strategy hybrid retrieval (BM25 + SPLADE + dense vectors)
- Lack of flexible backend switching without code changes

**Opportunity**: Implement a comprehensive vector storage system with:

- Universal adapter interface (`VectorStorePort`) supporting 10+ backends
- Advanced compression policies (int8, fp16, PQ, OPQ, binary quantization)
- GPU-accelerated indexing and search where available
- Multi-strategy retrieval orchestration with fusion ranking
- Per-namespace index configuration and dimension governance

This system enables production-ready, high-performance retrieval at billion-vector scale with configurable accuracy/latency/memory trade-offs.

## What Changes

### Core Capabilities

1. **Universal Vector Store Interface**
   - `VectorStorePort` protocol with create/upsert/knn/delete operations
   - `IndexParams` model with compression policy support
   - Per-namespace configuration with dimension validation
   - Support for dense, sparse, learned-sparse, and multi-vector embeddings

2. **Dense Vector Store Adapters** (10+ backends)
   - **Qdrant** (Rust HNSW): int8/binary quantization, GPU indexing, named vectors (default)
   - **FAISS** (C++/GPU): Flat, IVF_FLAT, IVF_PQ, HNSW, OPQ+IVF_PQ with reorder
   - **Milvus/Milvus-Lite** (C++/Go): GPU IVF/PQ, CAGRA, DiskANN option
   - **OpenSearch k-NN**: Lucene HNSW and FAISS engine with IVF/PQ
   - **Weaviate** (Go): HNSW with built-in hybrid BM25f fusion
   - **Vespa** (Java/C++): HNSW with rank profiles and ONNX
   - **pgvector** (PostgreSQL): ivfflat and HNSW
   - **DiskANN** (C++): SSD-resident ANN for massive scales
   - **Embedded libraries**: hnswlib, NMSLIB, Annoy, ScaNN
   - **Other local stores**: LanceDB, DuckDB-VSS, sqlite-vss, ChromaDB

3. **Sparse & Learned-Sparse Adapters**
   - **BM25/BM25F** (OpenSearch): Field boosts, biomedical analyzers
   - **SPLADE/uniCOIL/DeepImpact**: OpenSearch rank_features integration
   - **Neural-sparse** (OpenSearch ML): Sparse encoding processors

4. **Advanced Vector Compression**
   - **Scalar quantization**: int8 (4× memory reduction), fp16 (2×)
   - **Product Quantization (PQ)**: m subvectors, 4-bit or 8-bit codes
   - **Optimized PQ (OPQ)**: Rotation for better accuracy
   - **Binary quantization (BQ)**: 40× speedups with float reorder
   - **Two-stage search**: Compressed first-pass + float reorder for accuracy

5. **Retrieval Orchestration**
   - Parallel fan-out to multiple strategies (dense, BM25, SPLADE, ColBERT)
   - Fusion ranking: weighted linear and RRF (Reciprocal Rank Fusion)
   - Single-engine hybrids (Weaviate, OpenSearch all-in-one)
   - Per-namespace routing and result aggregation

6. **GPU Integration**
   - GPU-accelerated indexing (Qdrant, Milvus, FAISS)
   - GPU search for IVF/PQ and CAGRA
   - Fail-fast policy for GPU unavailability
   - Batch processing for efficiency

### Configuration

```yaml
vector_store:
  driver: qdrant  # or faiss, milvus, opensearch_knn, etc.
  server:
    url: "http://localhost:6333"
  collections:
    dense.bge.1024.v1:
      kind: hnsw
      dim: 1024
      metric: cosine
      m: 64
      ef_construct: 400
      ef_search: 128
      compression:
        kind: "scalar_int8"
      search:
        reorder_final: true
    dense.qwen.1536.v1:
      kind: ivf_pq
      dim: 1536
      metric: ip
      nlist: 32768
      nprobe: 64
      compression:
        kind: "opq_pq"
        opq_m: 64
        pq_m: 64
        pq_nbits: 8
      search:
        reorder_final: true
    multi.colbertv2.128.v1:
      kind: none  # store-only for late-interaction
      dim: 128

retrieval:
  strategies:
    - dense
    - bm25
    - splade
  fusion_method: rrf  # or weighted
  weights:
    dense: 0.35
    splade: 0.50
    bm25: 0.15
  gpu_enabled: true
```

### Implementation Structure

```
med/
  vstore/
    qdrant_store.py         # Default: HNSW + int8/BQ + GPU
    faiss_store.py          # Maximum control: Flat/IVF/PQ/HNSW
    milvus_store.py         # GPU IVF/PQ + CAGRA + DiskANN
    opensearch_knn_store.py # Lucene HNSW or FAISS engine
    weaviate_store.py       # Hybrid HNSW + BM25f
    pgvector_store.py       # PostgreSQL integration
    diskann_store.py        # SSD-resident ANN
    vespa_store.py          # Rank profiles + ONNX
    [embedded libs...]      # hnswlib, NMSLIB, Annoy, ScaNN
  sparse/
    bm25_os.py             # BM25/BM25F lexical
    splade_doc.py          # SPLADE → rank_features
    neural_sparse_os.py    # OpenSearch ML neural-sparse
  orchestration/
    retrieval_router.py    # Multi-strategy orchestration
    fusion.py              # RRF and weighted fusion
    namespace_router.py    # Per-namespace routing
```

## Impact

### Affected Specs

- **New**: `vector-storage` (dense vector store adapters with compression)
- **New**: `sparse-retrieval` (BM25, SPLADE, neural-sparse)
- **New**: `retrieval-orchestration` (multi-strategy fusion)

### Affected Code

- New package: `med/vstore/` for vector store adapters
- New package: `med/sparse/` for sparse retrieval
- New package: `med/orchestration/` for retrieval routing
- Integration: `IngestionService` → write embeddings to vector stores
- Integration: `RetrievalService` → orchestrate multi-strategy search
- Configuration: Extend YAML schema for vector store and retrieval settings

### Benefits

1. **Performance**: Billion-vector scale with <100ms P95 latency via compression and GPU acceleration
2. **Flexibility**: Swap backends via configuration without code changes
3. **Accuracy**: Hybrid retrieval outperforms single-strategy by 10-15% nDCG
4. **Memory efficiency**: 4-40× reduction via compression with minimal recall loss
5. **Scalability**: DiskANN and GPU options for massive collections

### Migration

- New system; no migration required
- Existing retrieval (if any) can run in parallel during transition
- Per-namespace rollout enables gradual adoption

### Risks

1. **Complexity**: 10+ backends increase maintenance surface
   - **Mitigation**: Unified `VectorStorePort` interface; optional adapters via lazy imports
2. **Compression tuning**: PQ/OPQ require parameter selection
   - **Mitigation**: Provide tuned defaults per backend; evaluation harness for A/B testing
3. **GPU dependency**: Some optimizations require GPU
   - **Mitigation**: CPU fallbacks; fail-fast policy with clear errors
4. **Dimension drift**: Mismatched embedding dimensions cause errors
   - **Mitigation**: Strict dimension validation per namespace; version governance

### Success Metrics

- P95 retrieval latency <100ms for 10M-vector collections
- nDCG@10 improvement of 10-15% with hybrid retrieval vs single-strategy
- Memory reduction of 4× with int8, 8-16× with PQ
- Support for 1B+ vectors via DiskANN or GPU IVF/PQ
- Zero dimension mismatch errors via namespace validation
