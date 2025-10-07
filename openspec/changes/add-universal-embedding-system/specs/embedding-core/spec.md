# Specification: Embedding Core System

## ADDED Requirements

### Requirement: Security & Multi-Tenant Integration

The embedding system SHALL integrate with the existing OAuth 2.0 authentication and multi-tenant isolation infrastructure.

#### Scenario: Tenant isolation in embeddings

- **WHEN** embeddings are generated
- **THEN** all EmbeddingRecord objects SHALL include tenant_id extracted from authenticated context
- **AND** tenant_id SHALL be validated against source document tenant_id
- **AND** tenant_id SHALL be immutable after creation

#### Scenario: Scope-based access control

- **WHEN** embedding service is invoked
- **THEN** the system SHALL verify `embed:write` scope in JWT token
- **AND** return 403 Forbidden if scope is missing
- **AND** log unauthorized access attempts

#### Scenario: Audit logging for embedding operations

- **WHEN** embedding operations complete
- **THEN** the system SHALL log: user_id, tenant_id, namespace, batch_size, duration, model_id
- **AND** include correlation_id for distributed tracing
- **AND** scrub PII from log messages

---

### Requirement: Error Handling & Status Codes

The embedding system SHALL provide comprehensive error handling with standardized HTTP status codes and RFC 7807 Problem Details.

#### Scenario: Invalid input error

- **WHEN** embedder receives malformed text input
- **THEN** the system SHALL return 400 Bad Request
- **AND** include RFC 7807 Problem Details with type, title, detail, instance
- **AND** log error with correlation_id

#### Scenario: Model configuration error

- **WHEN** invalid model_id or namespace specified
- **THEN** the system SHALL return 422 Unprocessable Entity
- **AND** include validation errors in Problem Details
- **AND** list available models and namespaces

#### Scenario: Resource exhaustion error

- **WHEN** embedding operation exceeds memory or GPU limits
- **THEN** the system SHALL return 503 Service Unavailable
- **AND** include Retry-After header
- **AND** trigger circuit breaker after repeated failures

#### Scenario: GPU unavailable for dense embeddings

- **WHEN** dense embedder requires GPU but none available
- **THEN** the system SHALL return 503 Service Unavailable
- **AND** include clear error message about GPU requirement
- **AND** fail-fast without CPU fallback (as per design)

---

### Requirement: Versioning & Backward Compatibility

The embedding system SHALL support versioning for model implementations and output schemas.

#### Scenario: Model version tracking

- **WHEN** embeddings are generated
- **THEN** each EmbeddingRecord SHALL include model_version field (e.g., "bge-large-en-v1.5:v1.0")
- **AND** version SHALL be immutable after creation
- **AND** enable querying embeddings by model version

#### Scenario: Schema evolution

- **WHEN** EmbeddingRecord schema changes (new fields added)
- **THEN** new fields SHALL be optional with defaults
- **AND** existing embeddings SHALL remain queryable
- **AND** migration scripts SHALL handle schema updates

#### Scenario: Deprecated model migration

- **WHEN** embedding model is deprecated
- **THEN** deprecation warning SHALL be logged
- **AND** migration path to new model SHALL be documented
- **AND** deprecated model SHALL remain functional for 2 major versions

---

### Requirement: Performance SLOs & Circuit Breakers

The embedding system SHALL enforce performance SLOs and implement circuit breakers for failing operations.

#### Scenario: Dense embedding latency SLO

- **WHEN** dense embedding operation executes
- **THEN** P95 latency SHALL be <150ms for batches <100 texts
- **AND** P95 latency SHALL be <500ms for batches <1000 texts
- **AND** operations exceeding 5× SLO SHALL trigger alerts

#### Scenario: Sparse embedding latency SLO

- **WHEN** sparse (SPLADE) embedding operation executes
- **THEN** P95 latency SHALL be <100ms for batches <100 texts
- **AND** P95 latency SHALL be <300ms for batches <1000 texts

#### Scenario: Circuit breaker on repeated failures

- **WHEN** embedder fails 5 consecutive times
- **THEN** circuit breaker SHALL open
- **AND** subsequent requests SHALL fail-fast with 503
- **AND** circuit SHALL attempt recovery after exponential backoff

#### Scenario: Resource monitoring

- **WHEN** embedding operations execute
- **THEN** the system SHALL monitor GPU memory usage per batch
- **AND** reject batches exceeding 4GB GPU memory limit
- **AND** emit metrics for memory usage, throughput, GPU utilization

---

### Requirement: Comprehensive Testing Requirements

The embedding system SHALL include comprehensive test coverage with contract, performance, and integration tests.

#### Scenario: Contract tests for BaseEmbedder interface

- **WHEN** new embedder is implemented
- **THEN** contract tests SHALL verify BaseEmbedder protocol compliance
- **AND** validate EmbeddingRecord output schema
- **AND** test dimension validation and namespace enforcement

#### Scenario: Performance regression tests

- **WHEN** embedder implementation changes
- **THEN** performance tests SHALL verify latency within SLO
- **AND** measure throughput (embeddings/second)
- **AND** compare against baseline to detect regressions

#### Scenario: Integration tests with downstream services

- **WHEN** embeddings are generated
- **THEN** integration tests SHALL verify embeddings are storable in vector stores
- **AND** verify embeddings are queryable via KNN
- **AND** test end-to-end pipeline (chunk → embed → index → retrieve)

---

### Requirement: BaseEmbedder Interface

The system SHALL provide a universal `BaseEmbedder` protocol interface supporting all embedding paradigms (dense, sparse, multi-vector, neural-sparse).

#### Scenario: Embedder implements required interface

- **GIVEN** a new embedding adapter is created
- **WHEN** the adapter is registered
- **THEN** the adapter SHALL implement `embed_documents()` accepting text list
- **AND** the adapter SHALL implement `embed_queries()` for query-specific encoding
- **AND** both methods SHALL return list of `EmbeddingRecord` objects

#### Scenario: Paradigm-specific output

- **GIVEN** different embedding paradigms
- **WHEN** embeddings are generated
- **THEN** dense embedders SHALL populate `vectors` field with single vectors
- **AND** multi-vector embedders SHALL populate `vectors` with token vectors
- **AND** sparse embedders SHALL populate `terms` field with term→weight maps
- **AND** `kind` field SHALL correctly identify paradigm

### Requirement: Namespace Management

The system SHALL enforce namespace-based embedding management with automatic dimension validation.

#### Scenario: Namespace format

- **GIVEN** an embedder configuration
- **WHEN** namespace is created
- **THEN** namespace SHALL follow format `{kind}.{model}.{dim}.{version}`
- **AND** namespace SHALL uniquely identify embedding configuration
- **AND** conflicts SHALL be detected and rejected

#### Scenario: Dimension introspection and validation

- **GIVEN** embeddings are generated
- **WHEN** first embedding is created for namespace
- **THEN** system SHALL introspect actual dimension from output
- **AND** system SHALL validate dimension matches configuration
- **AND** mismatches SHALL raise DimensionMismatchError

### Requirement: Dense Bi-Encoder Adapters

The system SHALL provide production-ready dense embedding adapters for BGE, E5, GTE, SPECTER, SapBERT via Sentence-Transformers.

#### Scenario: BGE embedding generation

- **GIVEN** BGE-large-en model configuration
- **WHEN** documents are embedded
- **THEN** embedder SHALL produce 1024-D L2-normalized vectors
- **AND** embedder SHALL use mean pooling
- **AND** batch processing SHALL be GPU-accelerated when available

#### Scenario: E5 prefix enforcement

- **GIVEN** E5 model with query_prefix and passage_prefix configured
- **WHEN** queries are embedded
- **THEN** embedder SHALL prepend "query: " prefix
- **WHEN** documents are embedded
- **THEN** embedder SHALL prepend "passage: " prefix
- **AND** prefixes SHALL be enforced automatically

### Requirement: SPLADE Sparse Embedder

The system SHALL provide SPLADE document and query encoders for learned-sparse retrieval.

#### Scenario: Document-side term expansion

- **GIVEN** SPLADE-v3 model
- **WHEN** documents are encoded
- **THEN** embedder SHALL generate term→weight map with ~30k vocabulary
- **AND** top-K terms SHALL be selected (default: 400)
- **AND** weights SHALL be positive floats
- **AND** output SHALL map to OpenSearch rank_features

### Requirement: ColBERT Multi-Vector Embedder

The system SHALL provide ColBERT-v2 late-interaction embeddings via RAGatouille.

#### Scenario: Token-level vector generation

- **GIVEN** ColBERT-v2 model
- **WHEN** documents are encoded
- **THEN** embedder SHALL generate N token vectors (N ≤ max_doc_tokens)
- **AND** vectors SHALL be 128-D
- **AND** output SHALL include token positions for MaxSim scoring

### Requirement: Storage Routing

The system SHALL automatically route embeddings to appropriate storage backends based on namespace kind.

#### Scenario: Dense embedding routing

- **GIVEN** embeddings with kind="single_vector"
- **WHEN** storage is determined
- **THEN** router SHALL select Qdrant, FAISS, or Milvus backend
- **AND** collection SHALL use namespace dimensions

#### Scenario: Sparse embedding routing

- **GIVEN** embeddings with kind="sparse"
- **WHEN** storage is determined
- **THEN** router SHALL select OpenSearch backend
- **AND** mapping SHALL use rank_features field type

### Requirement: Batch Processing

The system SHALL support efficient batch processing with configurable sizes and GPU utilization.

#### Scenario: Automatic batching

- **GIVEN** large document set
- **WHEN** embeddings are generated
- **THEN** documents SHALL be batched (default: 32)
- **AND** batches SHALL be processed in parallel when GPU available
- **AND** progress SHALL be tracked and reported

### Requirement: GPU Fail-Fast

The system SHALL enforce GPU availability when required and fail immediately if unavailable.

#### Scenario: GPU check on initialization

- **GIVEN** embedder requires GPU
- **WHEN** embedder is initialized without CUDA
- **THEN** system SHALL raise RuntimeError
- **AND** error message SHALL indicate GPU requirement
- **AND** no CPU fallback SHALL occur

### Requirement: Evaluation Harness

The system SHALL provide embedding quality evaluation via retrieval metrics.

#### Scenario: Retrieval impact measurement

- **GIVEN** multiple embedders
- **WHEN** evaluation runs
- **THEN** harness SHALL measure Recall@10, Recall@20, nDCG@10
- **AND** metrics SHALL isolate embedding impact
- **AND** leaderboard SHALL rank by retrieval quality

**Total: 10 core requirements with 16 scenarios**

(Additional requirements for framework adapters and experimental embedders follow similar pattern)

---

## Implementation Notes

### Monitoring & Alerting Thresholds

**Prometheus Metrics** (all labeled by model_id, kind, namespace, batch_size, tenant_id):

- `embedding_operations_total` (counter) - Total embedding operations
- `embedding_operations_duration_seconds` (histogram) - Operation latency with buckets: [0.01, 0.05, 0.1, 0.5, 1, 2, 5]
- `embedding_errors_total` (counter) - Errors by error_type
- `embedding_vectors_produced_total` (counter) - Total embeddings created
- `embedding_gpu_memory_bytes` (gauge) - GPU memory usage per batch
- `embedding_gpu_utilization_percent` (gauge) - GPU utilization
- `embedding_circuit_breaker_state` (gauge) - Circuit breaker states (0=closed, 1=open, 2=half-open)

**Alert Rules**:

- `EmbeddingHighLatency`: P95 > 300ms for 5 minutes → Page on-call
- `EmbeddingHighErrorRate`: Error rate > 5% for 5 minutes → Page on-call
- `EmbeddingCircuitBreakerOpen`: Circuit breaker open > 1 minute → Notify team
- `EmbeddingGPUMemoryHigh`: GPU memory > 3GB → Warning
- `EmbeddingGPUUnavailable`: GPU required but unavailable > 2 minutes → Page on-call

### Data Validation Rules

**EmbeddingRecord Validation**:

- `model_id` format: `^[a-zA-Z0-9/_-]{3,128}$`
- `namespace` format: `^[a-z0-9_.]+\.[a-z0-9_-]+\.\d+\.v\d+$` (e.g., "dense.bge.1024.v1")
- `tenant_id` format: `^[a-z0-9-]{8,64}$`
- `dim`: 128 ≤ value ≤ 4096 (dense), None (sparse)
- `vectors` length: 1 ≤ len ≤ 512 (multi-vector), exactly 1 (single-vector)
- `terms` size: 100 ≤ len ≤ 50,000 (sparse)

**Configuration Validation**:

- `batch_size`: 1 ≤ value ≤ 128
- `max_doc_tokens`: 128 ≤ value ≤ 8192
- `topk_terms` (SPLADE): 100 ≤ value ≤ 1000
- `normalize`: boolean
- `pooling`: ∈ {"mean", "cls", "max"}

### API Versioning

**Embedding API Endpoints**:

- `/v1/embed` - Current stable API
- `/v2/embed` - Future breaking changes (reserved)

**Version Headers**:

- Request: `Accept: application/vnd.medkg.embed.v1+json`
- Response: `Content-Type: application/vnd.medkg.embed.v1+json`
- Response: `X-API-Version: 1.0`

**Breaking Change Policy**:

- Breaking changes require new major version
- Old version supported for 12 months after new version release
- Deprecation warnings logged 6 months before sunset
- Migration guide published with new version (includes re-embedding strategy)

### Security Considerations

**Input Validation**:

- Reject batches > 10MB total text
- Sanitize all text content (remove control characters, validate UTF-8)
- Validate all namespaces against registered models before processing

**Rate Limiting**:

- Per-tenant: 500 embedding operations/minute
- Per-user: 200 embedding operations/minute
- Burst: 50 operations
- Return 429 with Retry-After header when exceeded

**Secrets Management**:

- Model paths: Configuration files (not hardcoded)
- HF API tokens: Environment variables or Vault
- TEI endpoints: Configuration with TLS
- vLLM endpoints: Authenticated with API keys

### Dependencies

- **Upstream**: `add-modular-chunking-system` (chunks are inputs)
- **Downstream**: `add-vector-storage-retrieval` (embeddings → vector stores), `add-retrieval-pipeline-orchestration` (embedding stage)
- **Python packages**: `torch>=2.4.0`, `transformers>=4.44.0`, `sentence-transformers>=3.1.0`, `FlagEmbedding`, `ragatouille`, `pyserini`, `httpx`
- **Models**:
  - Dense: `BAAI/bge-large-en-v1.5`, `intfloat/e5-large-v2`, `Alibaba-NLP/gte-large-en-v1.5`
  - Sparse: `naver/splade-v3-lexical`, `naver/efficient-splade-VI-BT-large-doc`
  - Multi-vector: `colbert-ir/colbertv2.0`
  - Biomedical: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, `allenai/specter2`
