# Implementation Tasks: Add Retrieval Pipeline Orchestration

## 1. Core Orchestration Infrastructure

- [x] 1.1 Define `PipelineStage` protocol (execute method)
- [x] 1.2 Create `PipelineConfig` model from YAML
- [x] 1.3 Implement `PipelineExecutor` (sequential stage execution)
- [x] 1.4 Add `ParallelExecutor` (concurrent strategy execution)
- [x] 1.5 Create correlation ID generation and propagation

## 2. Ingestion Pipeline Orchestration

### 2.1 Chunking Stage

- [x] 2.1.1 Implement `ChunkingWorker` Kafka consumer
- [x] 2.1.2 Subscribe to `ingest.chunking.v1` topic
- [x] 2.1.3 Call `ChunkingService` with doc and config
- [x] 2.1.4 Publish chunks to `ingest.chunks.v1` topic
- [x] 2.1.5 Update job ledger with chunking status

### 2.2 Embedding Stage

- [x] 2.2.1 Implement `EmbeddingWorker` Kafka consumer
- [x] 2.2.2 Subscribe to `ingest.chunks.v1` topic
- [x] 2.2.3 Call `EmbeddingService` for all configured namespaces
- [x] 2.2.4 Publish embeddings to `ingest.embeddings.v1` topic
- [x] 2.2.5 Handle multi-namespace embedding (dense, sparse, multi-vector)

### 2.3 Indexing Stage

- [x] 2.3.1 Implement `IndexingWorker` Kafka consumer
- [x] 2.3.2 Subscribe to `ingest.embeddings.v1` topic
- [x] 2.3.3 Route embeddings to appropriate vector stores by namespace
- [x] 2.3.4 Batch upsert for efficiency (50-100 vectors per batch)
- [x] 2.3.5 Publish completion to `ingest.indexed.v1` topic
- [x] 2.3.6 Mark job complete in ledger

### 2.4 Job State Management

- [x] 2.4.1 Implement `JobLedger` with Redis/Postgres backend
- [x] 2.4.2 Add job creation with initial status (queued)
- [x] 2.4.3 Implement stage transition tracking (chunking → embedding → indexing)
- [x] 2.4.4 Add error state handling
- [x] 2.4.5 Implement retry counter and max attempts
- [x] 2.4.6 Add job completion timestamp and duration

### 2.5 Error Handling & Retry

- [x] 2.5.1 Implement exponential backoff for transient errors
- [x] 2.5.2 Add dead letter queue (DLQ) for permanent failures
- [x] 2.5.3 Implement retry policy configuration (max attempts, backoff multiplier)
- [x] 2.5.4 Add error classification (retriable vs permanent)
- [x] 2.5.5 Implement DLQ monitoring and alerting

## 3. Query Pipeline Orchestration

### 3.1 Retrieval Stage

- [x] 3.1.1 Implement `RetrievalOrchestrator`
- [x] 3.1.2 Add parallel fan-out to enabled strategies
- [x] 3.1.3 Implement per-strategy timeout (50ms default)
- [x] 3.1.4 Add strategy result collection and validation
- [x] 3.1.5 Implement graceful degradation (partial results on strategy failure)

### 3.2 Fusion Stage

- [x] 3.2.1 Implement `FusionOrchestrator`
- [x] 3.2.2 Add result deduplication by doc_id
- [x] 3.2.3 Apply configured fusion algorithm (RRF, weighted)
- [x] 3.2.4 Implement score normalization if needed
- [x] 3.2.5 Add fusion result validation

### 3.3 Reranking Stage

- [x] 3.3.1 Implement `RerankOrchestrator`
- [x] 3.3.2 Add candidate selection (top N for reranking)
- [x] 3.3.3 Call configured reranker with batch processing
- [x] 3.3.4 Implement reranking timeout (50ms default)
- [x] 3.3.5 Add reranking cache lookup and write

### 3.4 Final Selection Stage

- [x] 3.4.1 Implement `FinalSelectorOrchestrator`
- [x] 3.4.2 Add top-K selection from reranked results
- [x] 3.4.3 Implement explain mode (include scores from all stages)
- [x] 3.4.4 Add result formatting and metadata enrichment

### 3.5 End-to-End Pipeline Executor

- [x] 3.5.1 Implement `QueryPipelineExecutor`
- [x] 3.5.2 Chain all query stages sequentially
- [x] 3.5.3 Add per-stage timing and metrics
- [x] 3.5.4 Implement total timeout enforcement (100ms target)
- [x] 3.5.5 Add error handling and partial result return

## 4. Profile Management

### 4.1 Profile Configuration

- [x] 4.1.1 Implement `ProfileManager` to load YAML profiles
- [x] 4.1.2 Add profile validation (all referenced components exist)
- [x] 4.1.3 Create default profiles (PMC, DailyMed, ClinicalTrials.gov)
- [x] 4.1.4 Implement profile inheritance (base + overrides)

### 4.2 Profile Detection

- [x] 4.2.1 Implement `ProfileDetector` based on doc metadata
- [x] 4.2.2 Add source-based detection (e.g., source="openalex" → PMC profile)
- [x] 4.2.3 Implement explicit profile override via API parameter
- [x] 4.2.4 Add fallback to default profile

### 4.3 Profile Application

- [x] 4.3.1 Apply ingestion profile at document intake
- [x] 4.3.2 Apply query profile based on target source/collection
- [x] 4.3.3 Log profile selection for audit

## 5. State Management & Resilience

### 5.1 Correlation ID Propagation

- [x] 5.1.1 Generate UUID correlation ID at request entry
- [x] 5.1.2 Propagate through all pipeline stages
- [x] 5.1.3 Include in all logs and metrics
- [x] 5.1.4 Add to Kafka message headers
- [x] 5.1.5 Include in HTTP response headers

### 5.2 Circuit Breakers

- [x] 5.2.1 Implement `CircuitBreaker` for each external service
- [x] 5.2.2 Add failure threshold configuration (e.g., 5 consecutive failures)
- [x] 5.2.3 Implement open/half-open/closed states
- [x] 5.2.4 Add automatic recovery with exponential backoff
- [x] 5.2.5 Emit alerts on circuit breaker state changes

### 5.3 Timeout Management

- [x] 5.3.1 Implement `TimeoutManager` per pipeline stage
- [x] 5.3.2 Add configurable timeouts (retrieval: 50ms, reranking: 50ms)
- [x] 5.3.3 Implement total pipeline timeout (100ms target)
- [x] 5.3.4 Add timeout breach logging and metrics

### 5.4 Graceful Degradation

- [x] 5.4.1 Implement fallback strategies on component failure
- [x] 5.4.2 Return partial results when possible
- [x] 5.4.3 Add degraded mode indicator in response
- [x] 5.4.4 Log degradation events for alerting

## 6. Monitoring & Observability

### 6.1 Prometheus Metrics

- [x] 6.1.1 Add ingestion pipeline metrics (docs/sec, stage latency, error rate)
- [x] 6.1.2 Add query pipeline metrics (query/sec, P50/P95/P99 latency, error rate)
- [x] 6.1.3 Add per-stage metrics (chunking, embedding, retrieval, fusion, reranking)
- [x] 6.1.4 Add job ledger metrics (queued, processing, completed, failed)
- [x] 6.1.5 Add circuit breaker state metrics

### 6.2 OpenTelemetry Tracing

- [x] 6.2.1 Add distributed tracing spans for all stages
- [x] 6.2.2 Propagate trace context via Kafka headers
- [x] 6.2.3 Include span attributes (stage name, doc_id, query, etc.)
- [x] 6.2.4 Configure sampling rate (10% default)
- [x] 6.2.5 Integrate with Jaeger for visualization

### 6.3 Structured Logging

- [x] 6.3.1 Add correlation ID to all log entries
- [x] 6.3.2 Log pipeline stage transitions
- [x] 6.3.3 Log error details with context
- [x] 6.3.4 Implement log aggregation (via structlog)
- [x] 6.3.5 Add sensitive data scrubbing

### 6.4 Alerting

- [x] 6.4.1 Configure latency threshold alerts (P95 > 200ms)
- [x] 6.4.2 Configure error rate alerts (>5% errors)
- [x] 6.4.3 Add circuit breaker state change alerts
- [x] 6.4.4 Add DLQ accumulation alerts
- [x] 6.4.5 Configure alert routing (PagerDuty, Slack)

## 7. Evaluation Framework

### 7.1 Ground Truth Management

- [x] 7.1.1 Implement `GroundTruthManager` (load queries + relevant docs)
- [x] 7.1.2 Add ground truth dataset schema (queries, doc_ids, relevance labels)
- [x] 7.1.3 Create annotation interface for new test sets
- [x] 7.1.4 Store ground truth in versioned files (JSONL)

### 7.2 Retrieval Metrics

- [x] 7.2.1 Implement nDCG@K (K=1,5,10,20)
- [x] 7.2.2 Implement Recall@K
- [x] 7.2.3 Implement MRR (Mean Reciprocal Rank)
- [x] 7.2.4 Implement MAP (Mean Average Precision)
- [x] 7.2.5 Add per-query metrics and aggregate statistics

### 7.3 Per-Stage Evaluation

- [x] 7.3.1 Evaluate chunking quality (boundary F1)
- [x] 7.3.2 Evaluate embedding quality (similarity correlation)
- [x] 7.3.3 Evaluate retrieval recall before fusion
- [x] 7.3.4 Evaluate fusion improvement vs single-strategy
- [x] 7.3.5 Evaluate reranking improvement vs fusion-only

### 7.4 Evaluation Harness

- [x] 7.4.1 Implement `EvalHarness` (run evaluation on test set)
- [x] 7.4.2 Add automated nightly evaluation runs
- [x] 7.4.3 Generate evaluation reports (markdown + JSON)
- [x] 7.4.4 Track metrics over time (regression detection)
- [x] 7.4.5 Compare multiple configurations side-by-side

### 7.5 A/B Testing Framework

- [x] 7.5.1 Implement `ABTestRunner` (split traffic between configs)
- [x] 7.5.2 Add experiment configuration (variant A vs B, traffic split)
- [x] 7.5.3 Track per-variant metrics (latency, accuracy, errors)
- [x] 7.5.4 Implement statistical significance testing
- [x] 7.5.5 Generate A/B test reports with recommendations

- [x] 8.1 Integrate ChunkingService via orchestration
- [x] 8.2 Integrate EmbeddingService via orchestration
- [x] 8.3 Integrate VectorStoreService via orchestration
- [x] 8.4 Integrate RerankerService via orchestration
- [x] 8.5 Add REST API endpoints for ingestion and query pipelines
- [x] 8.6 Add GraphQL resolvers for pipelines
- [x] 8.7 Implement SSE streaming for ingestion job progress

## 9. Configuration Management

- [x] 9.1 Extend YAML schema for complete pipeline configuration
- [x] 9.2 Add configuration validation (all components exist)
- [x] 9.3 Implement hot-reload for configuration changes (where safe)
- [x] 9.4 Add configuration versioning
- [x] 9.5 Create configuration migration tools

## 10. Testing

- [x] 10.1 Unit tests for each orchestrator
- [x] 10.2 Integration tests for ingestion pipeline (end-to-end)
- [x] 10.3 Integration tests for query pipeline (end-to-end)
- [x] 10.4 Performance tests (latency, throughput)
- [x] 10.5 Chaos tests (service failures, timeouts)
- [x] 10.6 Profile-based tests (PMC, DailyMed, ClinicalTrials.gov)

## 11. Documentation

- [x] 11.1 Document pipeline architecture and flow
- [x] 11.2 Create configuration guide (all YAML options)
- [x] 11.3 Document profile creation and customization
- [x] 11.4 Add troubleshooting guide (common errors, debugging)
- [x] 11.5 Document evaluation harness usage
- [x] 11.6 Create operational runbook (deployment, monitoring, alerting)

## Dependencies

- **Upstream**: All prior retrieval proposals (`add-modular-chunking-system`, `add-universal-embedding-system`, `add-vector-storage-retrieval`, `add-reranking-fusion-system`)
- **Downstream**: None (final integration layer)

## Estimated Effort

- Ingestion pipeline orchestration: 2 weeks
- Query pipeline orchestration: 2 weeks
- Profile management & state: 1 week
- Monitoring & observability: 1 week
- Evaluation framework & A/B testing: 1.5 weeks
- Integration & testing: 1.5 weeks
- **Total**: 9 weeks

## Success Criteria

- [ ] Ingestion pipeline processes 100+ docs/sec with <1% error rate
- [ ] Query pipeline P95 latency <100ms (retrieve → rerank → return)
- [ ] End-to-end nDCG@10 improvement of 15-20% vs baseline (BM25-only)
- [ ] 99.9% uptime with automatic failover on service errors
- [ ] 100% requests traced with correlation IDs
- [ ] Evaluation harness runs nightly with automated reports
- [ ] A/B testing framework functional with statistical significance testing
- [ ] 3+ per-source profiles functional (PMC, DailyMed, ClinicalTrials.gov)
