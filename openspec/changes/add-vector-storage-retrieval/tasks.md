# Implementation Tasks: Add Vector Storage & Retrieval System

## 1. Core Interfaces & Models

- [x] 1.1 Define `VectorStorePort` protocol (create_or_update_collection, upsert, knn, delete)
- [x] 1.2 Create `IndexParams` model (kind, metric, dim, HNSW/IVF/DiskANN parameters)
- [x] 1.3 Create `CompressionPolicy` model (kind, pq_m, pq_nbits, opq_m)
- [x] 1.4 Implement namespace registry with dimension validation
- [x] 1.5 Create vector store factory with backend selection

## 2. Dense Vector Store Adapters (Primary)

### 2.1 Qdrant (Default)

- [x] 2.1.1 Implement `QdrantStore` with HNSW support
- [x] 2.1.2 Add scalar int8 quantization support
- [x] 2.1.3 Add binary quantization (BQ) with float reorder
- [x] 2.1.4 Implement GPU-accelerated indexing
- [x] 2.1.5 Add named vectors for multi-vector (ColBERT)
- [x] 2.1.6 Implement payload filters and metadata indexing

### 2.2 FAISS

- [x] 2.2.1 Implement `FAISSStore` with Flat/IVF_FLAT/HNSW support
- [x] 2.2.2 Add IVF_PQ with training pipeline
- [x] 2.2.3 Implement OPQ+IVF_PQ via index_factory
- [x] 2.2.4 Add GPU variants (GpuIndexFlatL2, GpuIndexIVFPQ)
- [x] 2.2.5 Implement scalar quantization (SQ8, fp16)
- [x] 2.2.6 Add reorder functionality for PQ
- [x] 2.2.7 Create sidecar KV store (SQLite/LMDB) for payloads
- [x] 2.2.8 Implement index persistence (write_index/read_index)

### 2.3 Milvus/Milvus-Lite

- [x] 2.3.1 Implement `MilvusStore` with IVF_FLAT/IVF_PQ/HNSW
- [x] 2.3.2 Add GPU IVF/PQ support
- [x] 2.3.3 Implement GPU_CAGRA for high-QPS graphs
- [x] 2.3.4 Add DiskANN option
- [x] 2.3.5 Implement hybrid vector+scalar filters
- [x] 2.3.6 Add Milvus-Lite embedded mode support

### 2.4 OpenSearch k-NN

- [ ] 2.4.1 Implement `OpenSearchKNNStore` with Lucene HNSW
- [ ] 2.4.2 Add FAISS engine support (IVF, PQ encoder)
- [ ] 2.4.3 Implement _train API for centroids/codebooks
- [ ] 2.4.4 Add fp16/PQ compression via encoder settings
- [ ] 2.4.5 Integrate with BM25 and rank_features in same index
- [ ] 2.4.6 Implement hybrid query DSL (kNN + lexical)

## 3. Additional Dense Vector Stores

- [ ] 3.1 Implement `WeaviateStore` (HNSW + hybrid BM25f fusion)
- [ ] 3.2 Implement `VespaStore` (HNSW + rank profiles + ONNX)
- [ ] 3.3 Implement `PgvectorStore` (ivfflat and HNSW)
- [ ] 3.4 Implement `DiskANNStore` (SSD-resident ANN via diskannpy)
- [ ] 3.5 Implement `HNSWLibIndex` (embedded HNSW)
- [ ] 3.6 Implement `NMSLibIndex` (embedded HNSW variants)
- [ ] 3.7 Implement `AnnoyIndex` (random projection trees)
- [ ] 3.8 Implement `ScaNNIndex` (partition + asymmetric hashing)
- [ ] 3.9 Implement `LanceDBStore` (columnar on-disk)
- [ ] 3.10 Implement `DuckDBVSSStore` (embedded SQL + vectors)
- [ ] 3.11 Implement `ChromaStore` (simple local RAG store)

## 4. Sparse & Learned-Sparse Adapters

- [ ] 4.1 Implement `BM25Retriever` for OpenSearch (field boosts, analyzers)
- [ ] 4.2 Implement `BM25FRetriever` for OpenSearch (multi-field BM25)
- [ ] 4.3 Implement `SPLADEDocWriter` (write rank_features)
- [ ] 4.4 Implement `SPLADEQueryEncoder` (query-side sparse encoding)
- [ ] 4.5 Implement `NeuralSparseRetriever` (OpenSearch ML neural-sparse)
- [ ] 4.6 Create sparse vector query builder for OpenSearch

## 5. Compression Infrastructure

- [x] 5.1 Implement `CompressionManager` (policy validation and routing)
- [x] 5.2 Add int8 scalar quantization utilities
- [x] 5.3 Add fp16 scalar quantization utilities
- [x] 5.4 Implement PQ training pipeline (k-means on subvectors)
- [x] 5.5 Implement OPQ rotation matrix learning
- [x] 5.6 Add binary quantization utilities (bit packing/unpacking)
- [x] 5.7 Implement two-stage search (compressed + float reorder)
- [x] 5.8 Create compression evaluation harness (recall vs latency)

## 6. Retrieval Orchestration

- [x] 6.1 Implement `RetrievalRouter` (multi-strategy orchestration)
- [x] 6.2 Add parallel fan-out to multiple namespaces
- [x] 6.3 Implement weighted linear fusion
- [x] 6.4 Implement RRF (Reciprocal Rank Fusion)
- [x] 6.5 Add per-namespace result routing
- [x] 6.6 Implement single-engine hybrid mode (Weaviate, OpenSearch)
- [x] 6.7 Add result aggregation and deduplication
- [x] 6.8 Implement filter propagation to backends

## 7. GPU Integration

- [ ] 7.1 Add GPU availability checks (fail-fast policy)
- [ ] 7.2 Implement GPU-accelerated indexing (Qdrant, FAISS, Milvus)
- [ ] 7.3 Add GPU search support (FAISS GpuIndex, Milvus GPU_IVF)
- [ ] 7.4 Implement batch processing for GPU efficiency
- [ ] 7.5 Add GPU memory monitoring and utilization metrics
- [ ] 7.6 Create GPU fallback strategies (CPU when unavailable)

## 8. Configuration & Validation

- [x] 8.1 Extend YAML schema for vector_store configuration
- [x] 8.2 Add per-namespace index parameter validation
- [x] 8.3 Implement dimension governance (strict validation)
- [x] 8.4 Add compression policy validation
- [ ] 8.5 Create backend capability detection (GPU, compression types)
- [ ] 8.6 Implement configuration migration utilities

## 9. Integration with Existing Services

- [x] 9.1 Integrate `EmbeddingService` output → vector store upsert
- [ ] 9.2 Extend `IngestionService` to write to configured namespaces
- [ ] 9.3 Update `RetrievalService` to use `RetrievalRouter`
- [ ] 9.4 Add namespace selection based on embedding kind
- [ ] 9.5 Implement batch writing for ingestion efficiency

## 10. Evaluation Harness

- [ ] 10.1 Create ANN parameter sweep (m, ef_search, nlist, nprobe)
- [ ] 10.2 Implement compression A/B testing (float vs int8 vs PQ vs BQ)
- [ ] 10.3 Add hybrid retrieval evaluation (single vs multi-strategy)
- [ ] 10.4 Create recall@K and nDCG@K metrics
- [ ] 10.5 Implement latency profiling (P50, P95, P99)
- [ ] 10.6 Add memory usage tracking per backend
- [ ] 10.7 Create leaderboard for backend comparison

## 11. Testing

- [x] 11.1 Unit tests for `VectorStorePort` implementations
- [x] 11.2 Unit tests for compression utilities
- [x] 11.3 Unit tests for fusion algorithms (weighted, RRF)
- [ ] 11.4 Integration tests for each backend (Qdrant, FAISS, Milvus, OpenSearch)
- [ ] 11.5 Performance tests for compression methods
- [ ] 11.6 End-to-end tests for multi-strategy retrieval
- [ ] 11.7 GPU integration tests (with GPU availability checks)
- [ ] 11.8 Dimension validation tests (mismatch detection)

## 12. Documentation

- [ ] 12.1 Document `VectorStorePort` interface
- [ ] 12.2 Create backend selection guide (when to use which)
- [ ] 12.3 Document compression policies and trade-offs
- [ ] 12.4 Add tuning guides per backend (HNSW, IVF, PQ parameters)
- [ ] 12.5 Create YAML configuration examples
- [ ] 12.6 Document GPU integration and requirements
- [ ] 12.7 Add troubleshooting guide (dimension mismatches, OOM, etc.)

## 13. Operations & Monitoring

- [ ] 13.1 Add Prometheus metrics (latency, throughput, memory)
- [ ] 13.2 Implement index snapshot/backup utilities
- [ ] 13.3 Add index rebuild triggers (retraining for IVF)
- [ ] 13.4 Create health checks per backend
- [ ] 13.5 Implement namespace migration utilities
- [ ] 13.6 Add compression ratio monitoring

## Dependencies

- **Upstream**: `add-universal-embedding-system` (embeddings must be generated first)
- **Downstream**: `add-reranking-fusion-system` (reranking operates on retrieval results)

## Estimated Effort

- Core interfaces & Qdrant/FAISS/OpenSearch: 2 weeks
- Additional backends (Milvus, pgvector, embedded libs): 1.5 weeks
- Compression infrastructure: 1 week
- Retrieval orchestration & fusion: 1 week
- GPU integration & testing: 1 week
- Evaluation harness & tuning: 1 week
- **Total**: 7.5 weeks

## Success Criteria

- [ ] All backends implement `VectorStorePort` interface
- [ ] Compression policies (int8, PQ, BQ) functional with 4-40× memory reduction
- [ ] Multi-strategy retrieval achieves 10-15% nDCG@10 improvement vs single-strategy
- [ ] P95 latency <100ms for 10M-vector collections
- [ ] GPU-accelerated indexing 3-10× faster than CPU (when available)
- [ ] Zero dimension mismatch errors via namespace validation
- [ ] Evaluation harness produces recall/latency curves per backend
