# Spec: Retrieval Pipeline Orchestration

**Change ID**: add-retrieval-pipeline-orchestration

**Version**: 1.0

**Status**: Proposed

---

## ADDED Requirements

### Requirement: Ingestion Pipeline Orchestration

The system SHALL orchestrate end-to-end document ingestion through chunking, embedding, and indexing stages.

#### Scenario: Document ingestion workflow

- **WHEN** a document is submitted for ingestion
- **THEN** the system SHALL execute pipeline stages: chunking → embedding → indexing
- **AND** track job state in ledger (queued → processing → completed)
- **AND** publish progress events via SSE

#### Scenario: Async processing via Kafka

- **WHEN** ingestion stages execute
- **THEN** the system SHALL use Kafka topics for inter-stage communication
- **AND** enable horizontal scaling of workers
- **AND** guarantee at-least-once processing

#### Scenario: Multi-namespace embedding and indexing

- **WHEN** document is embedded
- **THEN** the system SHALL generate embeddings for all configured namespaces
- **AND** route embeddings to appropriate vector stores
- **AND** index in parallel where possible

#### Scenario: Job ledger state tracking

- **WHEN** ingestion progresses through stages
- **THEN** the system SHALL update job state in Redis/Postgres ledger
- **AND** record stage timestamps and durations
- **AND** enable status queries by job_id

#### Scenario: Error handling and retry

- **WHEN** transient error occurs during ingestion
- **THEN** the system SHALL retry with exponential backoff
- **AND** respect max retry limit (3 attempts default)
- **AND** move to DLQ after max retries

---

### Requirement: Query Pipeline Orchestration

The system SHALL orchestrate real-time query processing through retrieval, fusion, and reranking stages.

#### Scenario: Multi-strategy retrieval

- **WHEN** query is executed
- **THEN** the system SHALL fan out to enabled strategies (dense, BM25, SPLADE) in parallel
- **AND** collect results within per-strategy timeout (50ms default)
- **AND** continue with available results if strategy times out

#### Scenario: Fusion stage

- **WHEN** multi-strategy results are collected
- **THEN** the system SHALL deduplicate by doc_id
- **AND** apply configured fusion algorithm (RRF, weighted)
- **AND** produce unified ranked candidate list

#### Scenario: Reranking stage

- **WHEN** fusion completes
- **THEN** the system SHALL select top candidates for reranking (100 default)
- **AND** call configured reranker with batch processing
- **AND** return top-K final results (10 default)

#### Scenario: End-to-end latency target

- **WHEN** query pipeline executes
- **THEN** the system SHALL complete within 100ms P95 (target)
- **AND** enforce total timeout
- **AND** return partial results if timeout exceeded

#### Scenario: Explain mode

- **WHEN** query includes explain flag
- **THEN** the system SHALL include scores from all stages
- **AND** show retrieval scores, fusion scores, reranking scores
- **AND** enable debugging and tuning

---

### Requirement: Per-Source Profile Configuration

The system SHALL support per-source profiles with customized pipeline configurations.

#### Scenario: PMC (PubMed Central) profile

- **WHEN** document source is PMC or OpenAlex
- **THEN** the system SHALL apply PMC profile
- **AND** use semantic_splitter chunking (650 tokens)
- **AND** enable dense + SPLADE retrieval
- **AND** use cross-encoder reranking

#### Scenario: DailyMed (drug labels) profile

- **WHEN** document source is DailyMed
- **THEN** the system SHALL apply DailyMed profile
- **AND** use section_aware chunking (450 tokens)
- **AND** enable dense + BM25 retrieval
- **AND** use lexical reranking

#### Scenario: ClinicalTrials.gov profile

- **WHEN** document source is ClinicalTrials.gov
- **THEN** the system SHALL apply ClinicalTrials.gov profile
- **AND** use clinical_role chunking (350 tokens)
- **AND** enable dense + SPLADE + BM25 retrieval
- **AND** use late-interaction (ColBERT) reranking

#### Scenario: Explicit profile override

- **WHEN** query includes profile parameter
- **THEN** the system SHALL use specified profile
- **AND** override auto-detected profile

#### Scenario: Fallback to default profile

- **WHEN** source-specific profile not found
- **THEN** the system SHALL use default profile
- **AND** log profile selection decision

---

### Requirement: State Management and Resilience

The system SHALL provide robust state management and failure handling.

#### Scenario: Correlation ID propagation

- **WHEN** request enters system
- **THEN** the system SHALL generate UUID correlation ID
- **AND** propagate through all pipeline stages
- **AND** include in all logs, metrics, and traces

#### Scenario: Circuit breaker for failing services

- **WHEN** service experiences repeated failures (5 consecutive default)
- **THEN** the system SHALL open circuit breaker
- **AND** fail-fast subsequent requests
- **AND** attempt recovery with exponential backoff

#### Scenario: Per-stage timeout enforcement

- **WHEN** pipeline stage executes
- **THEN** the system SHALL enforce configured timeout
- **AND** terminate stage if timeout exceeded
- **AND** log timeout event and emit metric

#### Scenario: Graceful degradation

- **WHEN** non-critical component fails (e.g., one retrieval strategy)
- **THEN** the system SHALL continue with available components
- **AND** return partial results
- **AND** include degraded mode indicator in response

---

### Requirement: Monitoring and Observability

The system SHALL provide comprehensive monitoring and tracing.

#### Scenario: Pipeline stage metrics

- **WHEN** pipeline stages execute
- **THEN** the system SHALL emit Prometheus metrics
- **AND** label by stage, profile, and status
- **AND** track P50, P95, P99 latencies

#### Scenario: Distributed tracing

- **WHEN** request is processed
- **THEN** the system SHALL create OpenTelemetry trace
- **AND** include span for each stage
- **AND** propagate trace context via Kafka headers
- **AND** integrate with Jaeger

#### Scenario: Structured logging

- **WHEN** pipeline events occur
- **THEN** the system SHALL log with correlation ID
- **AND** include stage name, doc_id/query, duration
- **AND** scrub sensitive data (PII)

#### Scenario: Alerting on SLO violations

- **WHEN** P95 latency exceeds 200ms
- **THEN** the system SHALL trigger alert
- **AND** notify operations team
- **AND** include context (stage, profile, error rate)

---

### Requirement: Evaluation Framework

The system SHALL provide automated evaluation of retrieval quality.

#### Scenario: Ground truth dataset evaluation

- **WHEN** evaluation harness runs
- **THEN** the system SHALL load test queries and relevant documents
- **AND** execute retrieval pipeline for each query
- **AND** compute nDCG@K, Recall@K, MRR metrics

#### Scenario: Per-stage evaluation

- **WHEN** evaluation includes stage metrics
- **THEN** the system SHALL measure chunking quality (boundary F1)
- **AND** measure retrieval recall before fusion
- **AND** measure fusion improvement
- **AND** measure reranking improvement

#### Scenario: Nightly automated evaluation

- **WHEN** scheduled evaluation runs (nightly default)
- **THEN** the system SHALL execute evaluation on all profiles
- **AND** generate report with metrics and comparisons
- **AND** detect regressions vs previous runs

#### Scenario: A/B test experiment

- **WHEN** A/B test is configured (variant A vs B, 50/50 split)
- **THEN** the system SHALL route traffic based on split ratio
- **AND** track per-variant metrics (latency, nDCG, error rate)
- **AND** compute statistical significance
- **AND** generate A/B test report with recommendation

---

### Requirement: Integration with REST and GraphQL APIs

The system SHALL expose pipeline operations via REST and GraphQL endpoints.

#### Scenario: REST ingestion endpoint

- **WHEN** POST request to `/ingest` with document and profile
- **THEN** the system SHALL initiate ingestion pipeline
- **AND** return job_id immediately (async processing)
- **AND** emit SSE progress events to `/jobs/{job_id}/events`

#### Scenario: REST query endpoint

- **WHEN** POST request to `/query` with query text and profile
- **THEN** the system SHALL execute query pipeline
- **AND** return results within 100ms P95
- **AND** include explain details if requested

#### Scenario: GraphQL query mutation

- **WHEN** GraphQL query mutation is executed
- **THEN** the system SHALL execute query pipeline
- **AND** return typed SearchResult objects
- **AND** support nested field selection (scores, highlights, etc.)

#### Scenario: SSE job progress streaming

- **WHEN** client subscribes to `/jobs/{job_id}/events`
- **THEN** the system SHALL stream stage completion events
- **AND** include stage name, status, timestamp
- **AND** close stream on job completion or error

---

### Requirement: Configuration-Driven Pipeline Assembly

The system SHALL support configuration-driven pipeline definition without code changes.

#### Scenario: YAML pipeline definition

- **WHEN** pipeline configuration is loaded from YAML
- **THEN** the system SHALL validate all components exist
- **AND** assemble pipeline with specified stages and parameters
- **AND** apply profile-specific overrides

#### Scenario: Enable/disable pipeline stages

- **WHEN** stage is disabled in configuration
- **THEN** the system SHALL skip stage in pipeline
- **AND** log stage skip decision
- **AND** continue with remaining stages

#### Scenario: Hot-reload configuration (safe changes)

- **WHEN** configuration file is updated (non-breaking changes)
- **THEN** the system SHALL detect changes
- **AND** reload configuration without restart
- **AND** apply to new requests immediately

---

## Implementation Notes

### Ingestion Pipeline Flow

```
Document → Kafka(ingest.chunking.v1) → ChunkingWorker
    → Kafka(ingest.chunks.v1) → EmbeddingWorker
    → Kafka(ingest.embeddings.v1) → IndexingWorker
    → Kafka(ingest.indexed.v1) → Complete
```

### Query Pipeline Flow

```
Query → RetrievalOrchestrator (parallel: dense, BM25, SPLADE)
    → FusionOrchestrator (RRF/weighted)
    → RerankOrchestrator (cross-encoder/late-interaction)
    → FinalSelectorOrchestrator (top-K selection)
    → Results
```

### Default Configuration

```yaml
ingestion_pipeline:
  kafka:
    topics:
      requests: ingest.chunking.v1
      chunks: ingest.chunks.v1
      embeddings: ingest.embeddings.v1
      indexed: ingest.indexed.v1
  retry:
    max_attempts: 3
    backoff_multiplier: 2
    initial_delay_ms: 1000

query_pipeline:
  timeouts:
    retrieval_ms: 50
    fusion_ms: 5
    reranking_ms: 50
    total_ms: 120
  graceful_degradation: true

profiles:
  default:
    ingestion:
      chunking: { primary: semantic_splitter, target_tokens: 600 }
    query:
      strategies: [dense, bm25]
      fusion: rrf
      reranking: cross_encoder
```

### Latency Budget

- Retrieval (all strategies parallel): 50ms
- Fusion: 5ms
- Reranking: 50ms
- Overhead (routing, logging): 5ms
- **Total target**: 100ms P95

### Profile Selection Priority

1. Explicit profile parameter in request
2. Auto-detected by source metadata
3. Fallback to default profile

---

## Dependencies

- **Upstream**: All retrieval components (`ChunkingService`, `EmbeddingService`, `VectorStoreService`, `RerankerService`)
- **Infrastructure**: Kafka, Redis/Postgres (ledger), Prometheus, Jaeger
- **Python packages**: `aiokafka`, `redis`, `psycopg2`, `opentelemetry-api`, `prometheus-client`
