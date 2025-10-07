# Spec: Reranking & Fusion System

**Change ID**: add-reranking-fusion-system

**Version**: 1.0

**Status**: Proposed

---

## ADDED Requirements

### Requirement: Security & Multi-Tenant Integration

The reranking system SHALL integrate with OAuth 2.0 authentication and enforce multi-tenant isolation.

#### Scenario: Tenant isolation in reranking

- **WHEN** reranking operations execute
- **THEN** all query-document pairs SHALL be scoped by tenant_id
- **AND** cross-tenant document access SHALL be prevented

#### Scenario: Scope-based access control

- **WHEN** reranking service is invoked
- **THEN** the system SHALL verify `retrieve:read` scope in JWT token
- **AND** return 403 Forbidden if scope is missing

#### Scenario: Audit logging for reranking

- **WHEN** reranking operations complete
- **THEN** the system SHALL log: user_id, tenant_id, reranker_id, pair_count, duration
- **AND** include correlation_id for distributed tracing

---

### Requirement: Error Handling & Status Codes

The reranking system SHALL provide comprehensive error handling with RFC 7807 Problem Details.

#### Scenario: Invalid pair format error

- **WHEN** query-document pairs are malformed
- **THEN** the system SHALL return 400 Bad Request
- **AND** include Problem Details with validation errors

#### Scenario: Model not found error

- **WHEN** invalid reranker_id specified
- **THEN** the system SHALL return 422 Unprocessable Entity
- **AND** list available rerankers

#### Scenario: GPU unavailable error

- **WHEN** reranker requires GPU but none available
- **THEN** the system SHALL return 503 Service Unavailable
- **AND** fail-fast without CPU fallback (for GPU-only models)

---

### Requirement: Versioning & Backward Compatibility

The reranking system SHALL support model versioning and result reproducibility.

#### Scenario: Reranker version tracking

- **WHEN** reranking completes
- **THEN** results SHALL include reranker_version (e.g., "bge-reranker-v2-m3:v1.0")
- **AND** version SHALL be immutable

#### Scenario: Schema evolution

- **WHEN** reranking output schema changes
- **THEN** new fields SHALL be optional with defaults
- **AND** existing clients SHALL remain compatible

---

### Requirement: Performance SLOs & Circuit Breakers

The reranking system SHALL enforce strict performance SLOs.

#### Scenario: Reranking latency SLO

- **WHEN** reranking 100 pairs
- **THEN** P95 latency SHALL be <50ms on GPU
- **AND** operations exceeding 5× SLO SHALL trigger alerts

#### Scenario: Throughput SLO

- **WHEN** reranking batches
- **THEN** throughput SHALL be >2000 pairs/second on GPU

#### Scenario: Circuit breaker on failures

- **WHEN** reranker fails 5 consecutive times
- **THEN** circuit breaker SHALL open
- **AND** subsequent requests SHALL fail-fast with 503

---

### Requirement: Comprehensive Testing Requirements

The reranking system SHALL include comprehensive test coverage.

#### Scenario: Contract tests for RerankerPort

- **WHEN** new reranker is implemented
- **THEN** contract tests SHALL verify RerankerPort protocol compliance
- **AND** validate score ranges and ordering

#### Scenario: Performance regression tests

- **WHEN** reranker implementation changes
- **THEN** performance tests SHALL verify latency within SLO
- **AND** measure nDCG@10 quality vs baseline

#### Scenario: Integration tests with retrieval

- **WHEN** reranking completes
- **THEN** integration tests SHALL verify reranked results improve nDCG
- **AND** test end-to-end (retrieve → rerank → return)

---

### Requirement: Universal Reranker Interface

The system SHALL provide a `RerankerPort` protocol that defines a uniform interface for all reranking methods.

#### Scenario: Batch score query-document pairs

- **WHEN** a batch of (query, document) pairs is provided to a reranker
- **THEN** the reranker SHALL return normalized scores (0-1 range) for each pair
- **AND** process in batches for GPU efficiency (16-64 pairs typical)

#### Scenario: GPU-accelerated reranking

- **WHEN** GPU is available and reranker supports GPU
- **THEN** the reranker SHALL use GPU for inference (FP16 precision)
- **AND** achieve 3-5× speedup vs CPU

#### Scenario: CPU fallback for reranking

- **WHEN** GPU is unavailable and reranker supports CPU
- **THEN** the reranker SHALL use CPU inference
- **AND** apply ONNX quantization (int8) for acceleration if available

---

### Requirement: Cross-Encoder Rerankers

The system SHALL provide cross-encoder rerankers for pointwise query-document relevance scoring.

#### Scenario: BGE reranker (default)

- **WHEN** configured with `method: cross_encoder` and `model: BAAI/bge-reranker-v2-m3`
- **THEN** the system SHALL load BGE reranker model
- **AND** score query-document pairs with high precision
- **AND** achieve nDCG@10 improvement of 8-12% vs fusion-only

#### Scenario: MiniLM cross-encoder (fast)

- **WHEN** configured with `model: cross-encoder/ms-marco-MiniLM-L6-v2`
- **THEN** the system SHALL use distilled MiniLM model
- **AND** achieve <10ms per pair on CPU
- **AND** trade slight accuracy for speed

#### Scenario: MonoT5 pointwise reranker

- **WHEN** configured with `model: castorini/monot5-base-msmarco`
- **THEN** the system SHALL use T5-based relevance prediction
- **AND** format inputs as "Query: X Document: Y Relevant:"
- **AND** parse true/false token probabilities as scores

#### Scenario: Qwen reranker via vLLM

- **WHEN** configured with `model: Qwen/Qwen-Reranker` and vLLM endpoint
- **THEN** the system SHALL call vLLM OpenAI-compatible API
- **AND** extract relevance scores from response
- **AND** support large reranker models (>1B params)

---

### Requirement: Late-Interaction Reranking (ColBERT)

The system SHALL provide ColBERTv2 late-interaction reranking via MaxSim.

#### Scenario: ColBERT index reranking

- **WHEN** configured with `method: late_interaction` and `source: colbert_index`
- **THEN** the system SHALL use the external ColBERT indexer
- **AND** compute MaxSim (max cosine per query token)
- **AND** achieve high accuracy for abbreviation-heavy queries

#### Scenario: Qdrant multivector ColBERT reranking

- **WHEN** configured with `source: qdrant_multivector`
- **THEN** the system SHALL fetch token vectors from Qdrant named vectors
- **AND** compute MaxSim locally
- **AND** support custom ColBERT models

#### Scenario: Batch MaxSim computation

- **WHEN** multiple query-document pairs are reranked
- **THEN** the system SHALL batch MaxSim computation
- **AND** use GPU matrix operations for efficiency
- **AND** process 100+ pairs in <50ms on GPU

---

### Requirement: Lexical Reranking

The system SHALL provide BM25 and BM25F lexical reranking on candidate sets.

#### Scenario: BM25 candidate reranking

- **WHEN** configured with `method: lexical` and `method: bm25`
- **THEN** the system SHALL execute OpenSearch terms query on candidate IDs
- **AND** re-score with BM25 weights
- **AND** improve precision for exact-term matches

#### Scenario: BM25F multi-field reranking

- **WHEN** configured with `method: bm25f` and field weights
- **THEN** the system SHALL apply per-field BM25 scores
- **AND** combine with configured weights (e.g., title: 2.0, body: 1.0)
- **AND** boost title matches

#### Scenario: Fast lexical rerank (<5ms)

- **WHEN** lexical reranking is applied to 100 candidates
- **THEN** the system SHALL complete reranking in <5ms P95
- **AND** serve as fallback when GPU unavailable

---

### Requirement: Learned-to-Rank (LTR)

The system SHALL provide learned-to-rank integration with OpenSearch and Vespa.

#### Scenario: OpenSearch LTR with feature extraction

- **WHEN** configured with `method: ltr` and `engine: opensearch`
- **THEN** the system SHALL extract features (BM25, SPLADE, dense, recency)
- **AND** call OpenSearch LTR plugin with model name
- **AND** return re-scored results

#### Scenario: Train LambdaMART LTR model

- **WHEN** training data (queries, docs, relevance labels) is provided
- **THEN** the system SHALL extract features for training set
- **AND** train LambdaMART or XGBoost model
- **AND** upload model to OpenSearch

#### Scenario: Vespa rank profile reranking

- **WHEN** configured with `engine: vespa` and rank profile name
- **THEN** the system SHALL execute Vespa query with specified rank profile
- **AND** apply first-phase and second-phase ranking
- **AND** support ONNX model integration

---

### Requirement: Fusion Algorithms

The system SHALL provide fusion algorithms to combine multi-strategy retrieval results.

#### Scenario: Reciprocal Rank Fusion (RRF)

- **WHEN** configured with `fusion.method: rrf` and k parameter
- **THEN** the system SHALL compute RRF score: Σ(1 / (rank_i + k))
- **AND** use k=60 as default (robust baseline)
- **AND** handle score distribution differences

#### Scenario: Weighted linear fusion

- **WHEN** configured with `fusion.method: weighted` and strategy weights
- **THEN** the system SHALL normalize scores per strategy
- **AND** compute weighted sum: w1*score1 + w2*score2 + w3*score3
- **AND** validate weights sum to 1.0

#### Scenario: Learned fusion (LTR)

- **WHEN** configured with `fusion.method: learned` and LTR model
- **THEN** the system SHALL extract scores from all strategies as features
- **AND** apply LTR model to predict final relevance
- **AND** return ranked results

---

### Requirement: Score Normalization

The system SHALL provide score normalization methods for fusion.

#### Scenario: Min-max normalization

- **WHEN** scores are normalized with min-max method
- **THEN** the system SHALL compute: (score - min) / (max - min)
- **AND** map scores to [0, 1] range

#### Scenario: Z-score normalization

- **WHEN** scores are normalized with z-score method
- **THEN** the system SHALL compute: (score - mean) / std_dev
- **AND** handle outliers gracefully

#### Scenario: Softmax normalization

- **WHEN** scores are normalized with softmax method
- **THEN** the system SHALL compute: exp(score) / Σ exp(scores)
- **AND** ensure scores sum to 1.0

#### Scenario: Per-strategy normalization

- **WHEN** fusion combines multiple strategies
- **THEN** the system SHALL normalize scores independently per strategy
- **AND** apply before weighted combination

---

### Requirement: Result Deduplication

The system SHALL deduplicate results by document ID before reranking.

#### Scenario: Detect duplicates across strategies

- **WHEN** same document appears in multiple result lists (BM25, SPLADE, dense)
- **THEN** the system SHALL identify duplicates by doc_id
- **AND** preserve highest-ranking instance

#### Scenario: Aggregate scores for duplicates

- **WHEN** configured with score aggregation method (max, mean, sum)
- **THEN** the system SHALL combine scores across strategies
- **AND** use max score as default (most optimistic)

#### Scenario: Merge metadata for duplicates

- **WHEN** duplicates have different metadata (highlights, explanations)
- **THEN** the system SHALL merge metadata from all sources
- **AND** preserve unique information

---

### Requirement: Two-Stage Retrieval Pipeline

The system SHALL provide a two-stage pipeline for efficient retrieval and reranking.

#### Scenario: First-stage retrieval (1000 candidates)

- **WHEN** query is executed with two-stage pipeline
- **THEN** the system SHALL retrieve large candidate set (1000 typical)
- **AND** use fast retrieval methods (HNSW, BM25)
- **AND** maximize recall

#### Scenario: Candidate selection for reranking

- **WHEN** first-stage results are available
- **THEN** the system SHALL select top candidates for reranking (100 typical)
- **AND** apply fusion if multiple strategies used
- **AND** deduplicate before reranking

#### Scenario: Second-stage reranking

- **WHEN** candidates are selected
- **THEN** the system SHALL apply reranker to candidate set
- **AND** process in batches (16-64 pairs)
- **AND** return top-K final results (10 typical)

#### Scenario: Explain mode (show scores at each stage)

- **WHEN** configured with explain mode enabled
- **THEN** the system SHALL include scores from all stages
- **AND** show first-stage scores, fusion scores, reranking scores
- **AND** enable debugging and tuning

---

### Requirement: Batch Processing for GPU Efficiency

The system SHALL batch reranking operations for GPU efficiency.

#### Scenario: Dynamic batch size selection

- **WHEN** GPU is available for reranking
- **THEN** the system SHALL determine optimal batch size based on GPU memory
- **AND** use 16-64 pairs per batch (typical)
- **AND** avoid GPU OOM

#### Scenario: Async batch processing

- **WHEN** multiple queries are processed concurrently
- **THEN** the system SHALL aggregate query-document pairs into batches
- **AND** process batches asynchronously
- **AND** distribute results to original queries

#### Scenario: Batch timeout and fallback

- **WHEN** batch processing exceeds timeout threshold
- **THEN** the system SHALL process available pairs immediately
- **AND** fallback to CPU for remaining pairs if needed

---

### Requirement: Reranking Score Caching

The system SHALL cache reranking scores for repeated queries.

#### Scenario: Cache reranking scores with TTL

- **WHEN** query-document pairs are reranked
- **THEN** the system SHALL cache scores in Redis with TTL (3600s default)
- **AND** generate cache key: hash(query + doc_id + model)

#### Scenario: Cache hit retrieval

- **WHEN** same query-document pair is reranked again
- **THEN** the system SHALL retrieve cached score
- **AND** skip model inference
- **AND** achieve <1ms cache retrieval latency

#### Scenario: Cache invalidation on index update

- **WHEN** vector index is updated or documents re-indexed
- **THEN** the system SHALL invalidate affected cached scores
- **AND** ensure cache coherency

#### Scenario: Cache hit rate monitoring

- **WHEN** reranking is active
- **THEN** the system SHALL track cache hit rate
- **AND** emit Prometheus metric
- **AND** target >30% hit rate for production queries

---

### Requirement: Reranker Health Checks and Monitoring

The system SHALL provide health checks and metrics for rerankers.

#### Scenario: Reranker health check

- **WHEN** health check endpoint is called
- **THEN** each reranker SHALL report status (loaded, GPU available, model version)
- **AND** include sample inference latency

#### Scenario: Reranking latency metrics

- **WHEN** reranking operations are executed
- **THEN** the system SHALL emit Prometheus metrics
- **AND** label by reranker method and batch size
- **AND** track P50, P95, P99 latencies

#### Scenario: GPU utilization tracking

- **WHEN** GPU reranking is active
- **THEN** the system SHALL monitor GPU memory usage and utilization
- **AND** emit alerts on GPU OOM risk
- **AND** target >70% GPU utilization for efficiency

---

## Implementation Notes

### Reranker Selection Guide

| Reranker | Best For | Latency (100 pairs) | nDCG@10 Gain |
|----------|----------|---------------------|--------------|
| **BGE-reranker** | Default, high accuracy | ~40ms (GPU FP16) | +8-12% |
| **MiniLM** | Fast CPU/GPU | ~25ms (GPU) | +5-8% |
| **MonoT5** | Maximum quality | ~80ms (GPU) | +10-15% |
| **ColBERT** | Abbreviations, terms | ~30ms (GPU MaxSim) | +7-10% |
| **BM25** | Fast fallback | <5ms | +2-5% |
| **LTR** | Feature-rich | ~20ms | +6-10% |

### Default Configuration

```yaml
reranking:
  enabled: true
  method: cross_encoder
  model: BAAI/bge-reranker-v2-m3
  batch_size: 32
  gpu_enabled: true

fusion:
  method: rrf
  k: 60
  normalize: true
  deduplicate: true

pipeline:
  retrieve_candidates: 1000
  rerank_candidates: 100
  return_top_k: 10
```

### Fusion Method Selection

- **RRF**: Parameter-free, robust, good default
- **Weighted**: When strategies have calibrated scores and known quality
- **Learned (LTR)**: When training data available and features rich

---

## Dependencies

- **Upstream**: `add-vector-storage-retrieval` (retrieval provides candidates for reranking)
- **Downstream**: `add-retrieval-pipeline-orchestration` (orchestration integrates reranking)
- **Python packages**: `sentence-transformers`, `transformers`, `colbert-ai`, `onnxruntime`, OpenSearch LTR plugin, Vespa client

---

## Implementation Notes

### Monitoring & Alerting Thresholds

**Prometheus Metrics** (all labeled by reranker_id, batch_size, tenant_id):

- `reranking_operations_total` (counter) - Total reranking operations
- `reranking_duration_seconds` (histogram) - Latency with buckets: [0.01, 0.05, 0.1, 0.2, 0.5, 1]
- `reranking_errors_total` (counter) - Errors by error_type
- `reranking_pairs_processed_total` (counter) - Total query-doc pairs scored
- `reranking_gpu_utilization_percent` (gauge) - GPU utilization
- `reranking_circuit_breaker_state` (gauge) - Circuit breaker states

**Alert Rules**:

- `RerankingHighLatency`: P95 > 100ms for 5 minutes → Page on-call
- `RerankingHighErrorRate`: Error rate > 5% for 5 minutes → Page on-call
- `RerankingCircuitBreakerOpen`: Circuit breaker open > 1 minute → Notify team
- `RerankingGPUUnavailable`: GPU required but unavailable > 2 minutes → Page on-call

### Data Validation Rules

**Input Validation**:

- `query` length: 1 ≤ len ≤ 1000 characters
- `document` length: 10 ≤ len ≤ 10,000 characters
- `batch_size`: 1 ≤ value ≤ 128
- `top_k`: 1 ≤ value ≤ 1000

**Output Validation**:

- `score`: 0.0 ≤ value ≤ 1.0 (normalized)
- Score ordering: descending by default
- Pair count: output matches input count

### API Versioning

**Reranking API Endpoints**:

- `/v1/rerank` - Current stable API
- `/v2/rerank` - Future breaking changes (reserved)

**Version Headers**:

- Request: `Accept: application/vnd.medkg.rerank.v1+json`
- Response: `Content-Type: application/vnd.medkg.rerank.v1+json`
- Response: `X-API-Version: 1.0`

### Security Considerations

**Input Validation**:

- Reject batches > 5MB total text
- Sanitize query and document texts
- Validate tenant_id scope

**Rate Limiting**:

- Per-tenant: 500 reranking operations/minute
- Per-user: 200 reranking operations/minute
- Burst: 50 operations
- Return 429 with Retry-After header when exceeded

### Performance Tuning

**Batch Sizes** (GPU):

- Cross-encoder: 32-64 pairs optimal
- ColBERT: 16-32 pairs optimal
- LLM reranker: 4-8 pairs optimal

**Model Selection**:

- Fast: `ms-marco-MiniLM-L6-v2` (40ms P95, good quality)
- Balanced: `bge-reranker-v2-m3` (50ms P95, best quality)
- Specialized: `medcpt-reranker` (biomedical queries)
