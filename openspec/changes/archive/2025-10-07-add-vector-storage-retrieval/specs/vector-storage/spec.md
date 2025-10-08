# Spec: Vector Storage System

**Change ID**: add-vector-storage-retrieval

**Version**: 1.0

**Status**: Proposed

---

## ADDED Requirements

### Requirement: Security & Multi-Tenant Integration

The vector storage system SHALL integrate with OAuth 2.0 authentication and enforce strict multi-tenant isolation.

#### Scenario: Tenant isolation in vector storage

- **WHEN** vectors are upserted or queried
- **THEN** all operations SHALL be scoped by tenant_id from authenticated context
- **AND** tenant_id SHALL be included in all payloads and filters
- **AND** cross-tenant access SHALL be prevented at storage layer

#### Scenario: Scope-based access control

- **WHEN** vector storage operations are invoked
- **THEN** the system SHALL verify `index:write` scope for upsert/delete operations
- **AND** verify `index:read` scope for KNN queries
- **AND** return 403 Forbidden if scope is missing

#### Scenario: Audit logging for storage operations

- **WHEN** storage operations execute
- **THEN** the system SHALL log: user_id, tenant_id, namespace, operation, vector_count, duration
- **AND** include correlation_id for distributed tracing
- **AND** log security events (unauthorized access attempts)

---

### Requirement: Error Handling & Status Codes

The vector storage system SHALL provide comprehensive error handling with RFC 7807 Problem Details.

#### Scenario: Dimension mismatch error

- **WHEN** vector dimensions don't match namespace configuration
- **THEN** the system SHALL return 422 Unprocessable Entity
- **AND** include Problem Details with expected vs actual dimensions

#### Scenario: Index not found error

- **WHEN** operations reference non-existent namespace
- **THEN** the system SHALL return 404 Not Found
- **AND** list available namespaces in error response

#### Scenario: Resource exhaustion error

- **WHEN** storage backend reaches capacity limits
- **THEN** the system SHALL return 507 Insufficient Storage
- **AND** include capacity metrics in response

#### Scenario: Backend unavailable error

- **WHEN** vector store backend is unreachable
- **THEN** the system SHALL return 503 Service Unavailable
- **AND** trigger circuit breaker after repeated failures

---

### Requirement: Versioning & Backward Compatibility

The vector storage system SHALL support index versioning and migration strategies.

#### Scenario: Index version tracking

- **WHEN** collections are created
- **THEN** each collection SHALL include index_version metadata (e.g., "hnsw:v1.0")
- **AND** version SHALL be immutable after creation

#### Scenario: Schema evolution

- **WHEN** index schema changes (new metadata fields)
- **THEN** new fields SHALL be optional
- **AND** existing vectors SHALL remain queryable
- **AND** reindexing strategy SHALL be documented

#### Scenario: Deprecated index migration

- **WHEN** index type is deprecated (e.g., IVF → HNSW)
- **THEN** migration tool SHALL be provided
- **AND** old index supported for 2 major versions
- **AND** migration path documented with downtime estimates

---

### Requirement: Performance SLOs & Circuit Breakers

The vector storage system SHALL enforce strict performance SLOs for KNN queries.

#### Scenario: KNN latency SLO

- **WHEN** KNN query executes
- **THEN** P95 latency SHALL be <50ms for top_k=10
- **AND** P95 latency SHALL be <200ms for top_k=1000
- **AND** operations exceeding 5× SLO SHALL trigger alerts

#### Scenario: Upsert throughput SLO

- **WHEN** vectors are bulk upserted
- **THEN** throughput SHALL be >1000 vectors/second
- **AND** batch operations SHALL be preferred over individual upserts

#### Scenario: Circuit breaker on backend failures

- **WHEN** storage backend fails 5 consecutive times
- **THEN** circuit breaker SHALL open
- **AND** subsequent requests SHALL fail-fast with 503
- **AND** health checks SHALL attempt recovery

#### Scenario: Resource monitoring

- **WHEN** storage operations execute
- **THEN** the system SHALL monitor memory, disk, and network usage
- **AND** emit metrics for query latency, index size, vector count

---

### Requirement: Comprehensive Testing Requirements

The vector storage system SHALL include comprehensive test coverage.

#### Scenario: Contract tests for VectorStorePort

- **WHEN** new storage backend is implemented
- **THEN** contract tests SHALL verify VectorStorePort protocol compliance
- **AND** test CRUD operations, filtering, pagination

#### Scenario: Performance regression tests

- **WHEN** storage implementation changes
- **THEN** performance tests SHALL verify KNN latency within SLO
- **AND** measure indexing throughput
- **AND** test large-scale scenarios (1M+ vectors)

#### Scenario: Integration tests with embedding service

- **WHEN** embeddings are generated
- **THEN** integration tests SHALL verify embeddings are storable
- **AND** verify KNN returns correct neighbors
- **AND** test end-to-end (embed → index → retrieve)

---

### Requirement: Universal Vector Store Interface

The system SHALL provide a `VectorStorePort` protocol that defines a uniform interface for all vector storage backends.

#### Scenario: Create collection with index parameters

- **WHEN** a namespace is configured with index parameters (HNSW, IVF_PQ, etc.)
- **THEN** the vector store SHALL create or update the collection with specified parameters
- **AND** validate dimension compatibility

#### Scenario: Upsert vectors with metadata

- **WHEN** vectors are upserted with IDs, vectors, and payload metadata
- **THEN** the vector store SHALL insert new vectors or update existing ones
- **AND** validate vector dimensions match namespace configuration
- **AND** index metadata for filtering

#### Scenario: K-nearest neighbor search

- **WHEN** a query vector is provided with top-K and optional filters
- **THEN** the vector store SHALL return K nearest neighbors
- **AND** apply metadata filters if specified
- **AND** include similarity scores and payloads

#### Scenario: Delete vectors by ID

- **WHEN** vector IDs are provided for deletion
- **THEN** the vector store SHALL remove vectors from index
- **AND** confirm deletion success

---

### Requirement: Qdrant Dense Vector Store (Default)

The system SHALL provide a `QdrantStore` adapter implementing `VectorStorePort` with HNSW indexing, compression, and GPU support.

#### Scenario: Create HNSW collection

- **WHEN** configured with HNSW parameters (m, ef_construct, ef_search)
- **THEN** Qdrant SHALL create collection with specified HNSW graph
- **AND** support cosine, IP, and L2 metrics

#### Scenario: Enable int8 scalar quantization

- **WHEN** configured with `compression.kind: "scalar_int8"`
- **THEN** Qdrant SHALL store vectors as int8 (4× memory reduction)
- **AND** maintain recall >99% vs float32

#### Scenario: Enable binary quantization with reorder

- **WHEN** configured with `compression.kind: "binary"` and `search.reorder_final: true`
- **THEN** Qdrant SHALL use binary vectors for first-stage search (40× speedup)
- **AND** re-score top candidates with float32 vectors for accuracy

#### Scenario: GPU-accelerated indexing

- **WHEN** configured with GPU indexing enabled and GPU available
- **THEN** Qdrant SHALL use GPU for index building (3-10× faster)
- **AND** fall back to CPU if GPU unavailable

#### Scenario: Named vectors for multi-vector storage

- **WHEN** configured with multiple vector fields (e.g., dense + token vectors)
- **THEN** Qdrant SHALL store all vectors with independent indexing
- **AND** support retrieval by any named vector

---

### Requirement: FAISS Dense Vector Store

The system SHALL provide a `FAISSStore` adapter with support for Flat, IVF, HNSW, and PQ compression.

#### Scenario: Create IVF_PQ index with training

- **WHEN** configured with IVF_PQ parameters (nlist, nprobe, pq_m, pq_nbits)
- **THEN** FAISS SHALL train centroids and PQ codebooks on sample data
- **AND** create compressed index (8-16× memory reduction)

#### Scenario: Optimized PQ (OPQ) rotation

- **WHEN** configured with `compression.kind: "opq_pq"`
- **THEN** FAISS SHALL learn rotation matrix for better subvector independence
- **AND** improve recall by 2-5% vs standard PQ

#### Scenario: Two-stage search with reorder

- **WHEN** configured with `search.reorder_final: true`
- **THEN** FAISS SHALL retrieve R candidates by PQ distance
- **AND** re-score candidates with float32 vectors
- **AND** return top-K most accurate results

#### Scenario: GPU index for brute-force baseline

- **WHEN** configured with GPU and Flat index
- **THEN** FAISS SHALL use GpuIndexFlatL2 for exact search
- **AND** achieve <50ms latency for <1M vectors on GPU

#### Scenario: Persist index to disk

- **WHEN** index build completes
- **THEN** FAISS SHALL write index to file via `faiss.write_index`
- **AND** store ID-to-payload mapping in SQLite sidecar

---

### Requirement: Milvus Dense Vector Store

The system SHALL provide a `MilvusStore` adapter with GPU IVF/PQ, CAGRA, and DiskANN support.

#### Scenario: Create GPU IVF_PQ index

- **WHEN** configured with `gpu.enabled: true` and `gpu.kind: "GPU_IVF_PQ"`
- **THEN** Milvus SHALL build IVF_PQ index on GPU (5-10× faster)
- **AND** serve queries on GPU when available

#### Scenario: Create GPU_CAGRA graph index

- **WHEN** configured with `gpu.kind: "GPU_CAGRA"`
- **THEN** Milvus SHALL build CAGRA proximity graph on GPU
- **AND** support high-QPS queries with low latency

#### Scenario: DiskANN for massive scale

- **WHEN** configured with `kind: "diskann"`
- **THEN** Milvus SHALL create SSD-resident Vamana graph
- **AND** page data from disk during queries
- **AND** support billion-vector collections

---

### Requirement: OpenSearch k-NN Vector Store

The system SHALL provide an `OpenSearchKNNStore` adapter with Lucene HNSW and FAISS engine support.

#### Scenario: Create Lucene HNSW index

- **WHEN** configured with `engine: "lucene"` and HNSW parameters
- **THEN** OpenSearch SHALL create native Lucene k-NN index
- **AND** integrate with BM25 and rank_features in same index

#### Scenario: Create FAISS IVF_PQ index via engine

- **WHEN** configured with `engine: "faiss"` and PQ encoder
- **THEN** OpenSearch SHALL use FAISS backend for IVF_PQ
- **AND** call _train API to compute centroids and codebooks
- **AND** support fp16 and int8 scalar quantization

#### Scenario: Hybrid query with dense and lexical

- **WHEN** query includes both kNN and BM25 clauses
- **THEN** OpenSearch SHALL execute both in single request
- **AND** combine scores via weighted sum or script

---

### Requirement: Additional Dense Vector Stores

The system SHALL provide adapters for pgvector, Weaviate, Vespa, DiskANN, and embedded libraries.

#### Scenario: Weaviate hybrid search

- **WHEN** configured with Weaviate driver
- **THEN** Weaviate SHALL execute HNSW vector search
- **AND** fuse with BM25f lexical scores automatically
- **AND** return unified ranked results

#### Scenario: Vespa rank profiles

- **WHEN** configured with Vespa driver and rank profile
- **THEN** Vespa SHALL execute HNSW search
- **AND** apply custom rank profile (ONNX model, BM25, recency)
- **AND** return scored results

#### Scenario: pgvector SQL integration

- **WHEN** configured with pgvector driver
- **THEN** pgvector SHALL create ivfflat or HNSW index in PostgreSQL
- **AND** support SQL filters and joins
- **AND** enable transactional vector operations

#### Scenario: DiskANN SSD-resident search

- **WHEN** configured with DiskANN driver
- **THEN** DiskANN SHALL build Vamana graph on NVMe
- **AND** page small chunks during queries
- **AND** support collections exceeding RAM by 10-100×

#### Scenario: Embedded hnswlib index

- **WHEN** configured with hnswlib driver
- **THEN** hnswlib SHALL create in-process HNSW index
- **AND** provide fast CPU search
- **AND** use SQLite sidecar for payloads

---

### Requirement: Compression Policy Management

The system SHALL support configurable compression policies per namespace with backend-specific translation.

#### Scenario: Configure scalar int8 quantization

- **WHEN** configured with `compression.kind: "scalar_int8"`
- **THEN** the system SHALL apply int8 quantization (4× memory reduction)
- **AND** maintain recall >99% for most datasets

#### Scenario: Configure fp16 quantization

- **WHEN** configured with `compression.kind: "fp16"`
- **THEN** the system SHALL use half-precision floats (2× memory reduction)
- **AND** achieve near-identical recall to float32

#### Scenario: Configure PQ compression

- **WHEN** configured with `compression: {kind: "pq", pq_m: 64, pq_nbits: 8}`
- **THEN** the system SHALL train PQ codebooks with m subvectors
- **AND** use nbits per subvector (typically 8-bit = 256 centroids)
- **AND** achieve 8-16× memory reduction

#### Scenario: Configure OPQ+PQ compression

- **WHEN** configured with `compression: {kind: "opq_pq", opq_m: 64, pq_m: 64, pq_nbits: 8}`
- **THEN** the system SHALL learn OPQ rotation matrix
- **AND** apply rotation before PQ encoding
- **AND** improve recall by 2-5% vs standard PQ

#### Scenario: Configure binary quantization

- **WHEN** configured with `compression.kind: "binary"` (Qdrant-specific)
- **THEN** the system SHALL convert vectors to 1-bit per dimension
- **AND** use Hamming distance for first-stage search
- **AND** achieve 32-40× memory reduction

#### Scenario: Enable two-stage reorder

- **WHEN** configured with `search.reorder_final: true`
- **THEN** the system SHALL retrieve R candidates (e.g., 1000) by compressed distance
- **AND** re-score candidates using original float32 vectors
- **AND** return top-K (e.g., 10) most accurate results

---

### Requirement: Namespace Dimension Validation

The system SHALL validate vector dimensions match namespace configuration at upsert and query time.

#### Scenario: Detect dimension mismatch at upsert

- **WHEN** vectors with incorrect dimensions are upserted to a namespace
- **THEN** the system SHALL raise `DimensionMismatchError`
- **AND** report expected vs actual dimensions
- **AND** reject the upsert operation

#### Scenario: Detect dimension mismatch at query

- **WHEN** a query vector with incorrect dimensions is provided
- **THEN** the system SHALL raise `DimensionMismatchError`
- **AND** prevent query execution

#### Scenario: Validate namespace configuration

- **WHEN** a namespace is created with index parameters
- **THEN** the system SHALL store expected dimension in registry
- **AND** enforce dimension checks on all operations

---

### Requirement: Per-Namespace Index Configuration

The system SHALL support independent index configuration per namespace via YAML.

#### Scenario: Configure HNSW for small dense model

- **WHEN** namespace is `dense.bge.384.v1`
- **THEN** the system SHALL use HNSW with m=64, ef_construct=400
- **AND** enable int8 compression
- **AND** achieve <25ms P95 latency

#### Scenario: Configure IVF_PQ for large dense model

- **WHEN** namespace is `dense.qwen.1536.v1`
- **THEN** the system SHALL use IVF_PQ with nlist=32768, nprobe=64
- **AND** enable OPQ rotation
- **AND** reduce memory by 16× with <2% recall loss

#### Scenario: Configure store-only for multi-vector

- **WHEN** namespace is `multi.colbertv2.128.v1`
- **THEN** the system SHALL store token vectors without indexing
- **AND** enable retrieval by ID for reranking
- **AND** skip search operations

---

### Requirement: GPU-Accelerated Operations

The system SHALL support GPU-accelerated indexing and search with fail-fast policy.

#### Scenario: GPU-accelerated indexing (Qdrant)

- **WHEN** configured with `gpu_indexing: true` and GPU available
- **THEN** Qdrant SHALL use GPU for HNSW index building
- **AND** achieve 3-10× faster index creation

#### Scenario: GPU IVF search (FAISS, Milvus)

- **WHEN** configured with GPU and IVF index
- **THEN** FAISS/Milvus SHALL execute queries on GPU
- **AND** achieve higher throughput for large K

#### Scenario: GPU fail-fast on unavailability

- **WHEN** GPU operations are configured but GPU unavailable
- **THEN** the system SHALL raise `GPUUnavailableError` at startup
- **AND** refuse to start service
- **AND** provide clear error message

#### Scenario: CPU fallback for optional GPU

- **WHEN** GPU acceleration is optional (`gpu_enabled: false`)
- **THEN** the system SHALL use CPU implementations
- **AND** log warning about performance degradation

---

### Requirement: Sparse Retrieval (BM25, SPLADE, Neural-Sparse)

The system SHALL provide sparse and learned-sparse retrieval adapters for OpenSearch.

#### Scenario: BM25 lexical search

- **WHEN** query text is provided for BM25 search
- **THEN** OpenSearch SHALL execute BM25 ranking
- **AND** apply field boosts if configured
- **AND** return scored documents

#### Scenario: BM25F multi-field search

- **WHEN** query spans multiple fields (title, body, keywords)
- **THEN** OpenSearch SHALL apply BM25F with per-field weights
- **AND** combine field scores

#### Scenario: SPLADE document indexing

- **WHEN** documents are indexed with SPLADE embeddings
- **THEN** OpenSearch SHALL store term weights in `rank_features` field
- **AND** enable efficient retrieval via term expansion

#### Scenario: SPLADE query expansion

- **WHEN** query is encoded with SPLADE
- **THEN** OpenSearch SHALL expand query with weighted terms
- **AND** execute rank_features query
- **AND** return documents with learned-sparse scores

#### Scenario: Neural-sparse retrieval (OpenSearch ML)

- **WHEN** configured with neural-sparse processor
- **THEN** OpenSearch SHALL encode documents with sparse encoder
- **AND** enable neural query operator
- **AND** return documents with neural-sparse scores

---

### Requirement: Index Persistence and Snapshots

The system SHALL support index persistence, snapshots, and backups per backend.

#### Scenario: Qdrant snapshot creation

- **WHEN** snapshot is requested for Qdrant collection
- **THEN** Qdrant SHALL create point-in-time snapshot
- **AND** store snapshot to disk or S3
- **AND** enable restore to any instance

#### Scenario: FAISS index persistence

- **WHEN** FAISS index is built or updated
- **THEN** FAISS SHALL write index to versioned file
- **AND** persist ID-to-payload mapping in SQLite
- **AND** enable loading at startup

#### Scenario: OpenSearch snapshot repository

- **WHEN** OpenSearch index snapshot is requested
- **THEN** OpenSearch SHALL create snapshot via snapshot API
- **AND** store to configured repository (S3, NFS, etc.)

#### Scenario: Index rebuild trigger

- **WHEN** data distribution shifts significantly (IVF)
- **THEN** the system SHALL trigger index retraining
- **AND** build new index in background
- **AND** hot-swap indexes atomically

---

### Requirement: Health Checks and Monitoring

The system SHALL provide health checks and metrics per vector store backend.

#### Scenario: Backend health check

- **WHEN** health check endpoint is called
- **THEN** each vector store adapter SHALL report health status
- **AND** include collection count, vector count, index status

#### Scenario: Query latency metrics

- **WHEN** queries are executed
- **THEN** the system SHALL emit Prometheus metrics
- **AND** label by namespace, backend, and filter presence
- **AND** track P50, P95, P99 latencies

#### Scenario: Memory usage tracking

- **WHEN** collections are active
- **THEN** the system SHALL report memory usage per namespace
- **AND** compare to uncompressed baseline
- **AND** calculate compression ratio

#### Scenario: GPU utilization metrics

- **WHEN** GPU operations are active
- **THEN** the system SHALL report GPU memory usage and utilization
- **AND** emit alerts on GPU OOM risk

---

## Implementation Notes

### Backend Priority

1. **Tier 1 (production-critical)**: Qdrant, FAISS, OpenSearch k-NN
2. **Tier 2 (GPU/scale)**: Milvus, DiskANN
3. **Tier 3 (specialized)**: Weaviate, Vespa, pgvector
4. **Tier 4 (experimental)**: Embedded libs (hnswlib, NMSLIB, etc.)

### Compression Defaults

- **General purpose**: int8 (4× reduction, <1% recall loss)
- **Memory-constrained**: PQ64x8 (16× reduction, 2-3% recall loss)
- **Extreme memory**: OPQ+PQ64x8 (16× reduction, 1-2% recall loss)
- **Fast first-stage**: Binary quantization + reorder (40× speedup, reorder recovers accuracy)

### HNSW Tuning

- **High recall**: m=64, ef_construct=400-512, ef_search=128-256
- **Balanced**: m=32, ef_construct=200, ef_search=64-128
- **Fast**: m=16, ef_construct=100, ef_search=32-64

### IVF Tuning

- **nlist**: ~√N (e.g., 32768 for 1B vectors)
- **nprobe**: 10-64 (higher = better recall, slower)
- **Combine with PQ**: IVF coarse quantization + PQ fine quantization

### Reorder Heuristic

- Retrieve R = 3-10× top-K by compressed distance
- Re-score R with float32 vectors
- Return top-K accurate results
- Typical: Retrieve 100, reorder, return 10

---

## Dependencies

- **Upstream**: `add-universal-embedding-system` (embeddings must be generated)
- **Downstream**: `add-reranking-fusion-system` (reranking consumes retrieval results)
- **Python packages**: `qdrant-client`, `faiss-gpu`, `pymilvus`, `opensearch-py`, `weaviate-client`, `pgvector`, `hnswlib`, `nmslib`, `annoy`, `scann`

---

## Implementation Notes

### Monitoring & Alerting Thresholds

**Prometheus Metrics** (all labeled by namespace, backend, operation, tenant_id):

- `vector_store_operations_total` (counter) - Total operations
- `vector_store_knn_duration_seconds` (histogram) - KNN query latency with buckets: [0.01, 0.05, 0.1, 0.2, 0.5, 1]
- `vector_store_upsert_duration_seconds` (histogram) - Upsert latency
- `vector_store_errors_total` (counter) - Errors by error_type
- `vector_store_index_size_bytes` (gauge) - Index size per namespace
- `vector_store_vector_count` (gauge) - Total vectors per namespace
- `vector_store_circuit_breaker_state` (gauge) - Circuit breaker states

**Alert Rules**:

- `VectorStoreHighLatency`: KNN P95 > 100ms for 5 minutes → Page on-call
- `VectorStoreHighErrorRate`: Error rate > 5% for 5 minutes → Page on-call
- `VectorStoreCircuitBreakerOpen`: Circuit breaker open > 1 minute → Notify team
- `VectorStoreDiskFull`: Disk usage > 80% → Warning, > 90% → Page
- `VectorStoreIndexCorruption`: Health check failures > 3 → Page on-call

### Data Validation Rules

**Vector Validation**:

- `vector_id` format: `^[a-z0-9_-]+:[a-z0-9_-]+:[a-z_]+:\d+$` (matches chunk_id)
- `namespace` format: `^[a-z0-9_.]+\.[a-z0-9_-]+\.\d+\.v\d+$`
- `tenant_id` format: `^[a-z0-9-]{8,64}$`
- `vector` dimensions: match namespace `dim` exactly
- `payload` size: ≤ 10KB per vector

**Index Parameters Validation**:

- `dim`: 128 ≤ value ≤ 4096
- `m` (HNSW links): 4 ≤ value ≤ 128
- `ef_construct`: 100 ≤ value ≤ 1000
- `ef_search`: 10 ≤ value ≤ 500
- `nlist` (IVF): 100 ≤ value ≤ 100,000
- `nprobe`: 1 ≤ value ≤ nlist

### API Versioning

**Vector Storage API Endpoints**:

- `/v1/vectors/{namespace}` - Current stable API
- `/v2/vectors/{namespace}` - Future breaking changes (reserved)

**Version Headers**:

- Request: `Accept: application/vnd.medkg.vector.v1+json`
- Response: `Content-Type: application/vnd.medkg.vector.v1+json`
- Response: `X-API-Version: 1.0`

**Breaking Change Policy**:

- Index format changes require new major version
- Reindexing required for major version upgrades
- Old indexes supported read-only for 12 months
- Migration tools provided with downtime estimates

### Security Considerations

**Input Validation**:

- Reject batch upserts > 50MB
- Validate all vector IDs against format regex
- Sanitize metadata payloads (no code execution)
- Verify namespace exists before operations

**Rate Limiting**:

- Per-tenant: 1000 KNN queries/minute, 500 upserts/minute
- Per-user: 500 KNN queries/minute, 200 upserts/minute
- Burst: 100 operations
- Return 429 with Retry-After header when exceeded

**Secrets Management**:

- Backend credentials: Environment variables or Vault
- TLS certificates: Mounted from secure storage
- API keys: Rotated every 90 days

### Performance Tuning

**HNSW Parameters** (Qdrant, recommended):

```yaml
hnsw:
  m: 64              # More links = better recall, higher memory
  ef_construct: 400  # Higher = better index quality, slower build
  ef_search: 128     # Higher = better recall, slower query
```

**IVF+PQ Parameters** (FAISS, for scale):

```yaml
ivf_pq:
  nlist: 32768       # Voronoi cells (sqrt to 4x of data size)
  nprobe: 64         # Cells to search (2-10% of nlist)
  pq_m: 64           # PQ subquantizers (divisor of dim)
  pq_nbits: 8        # Bits per subquantizer
```

**Compression Trade-offs**:

- `scalar_int8`: 4× smaller, <2% recall loss, 2× faster
- `pq`: 8-32× smaller, 5-10% recall loss, 1.5× faster
- `binary`: 32× smaller, 10-15% recall loss, 5× faster
