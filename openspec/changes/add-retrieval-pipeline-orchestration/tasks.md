# Implementation Tasks: Add Retrieval Pipeline Orchestration

## 1. Core Orchestration Infrastructure

- [ ] 1.1 Define `PipelineStage` protocol (execute method)
- [ ] 1.2 Create `PipelineConfig` model from YAML
- [ ] 1.3 Implement `PipelineExecutor` (sequential stage execution)
- [ ] 1.4 Add `ParallelExecutor` (concurrent strategy execution)
- [ ] 1.5 Create correlation ID generation and propagation

## 2. Ingestion Pipeline Orchestration

### 2.1 Chunking Stage

- [ ] 2.1.1 Implement `ChunkingWorker` Kafka consumer
- [ ] 2.1.2 Subscribe to `ingest.chunking.v1` topic
- [ ] 2.1.3 Call `ChunkingService` with doc and config
- [ ] 2.1.4 Publish chunks to `ingest.chunks.v1` topic
- [ ] 2.1.5 Update job ledger with chunking status

### 2.2 Embedding Stage

- [ ] 2.2.1 Implement `EmbeddingWorker` Kafka consumer
- [ ] 2.2.2 Subscribe to `ingest.chunks.v1` topic
- [ ] 2.2.3 Call `EmbeddingService` for all configured namespaces
- [ ] 2.2.4 Publish embeddings to `ingest.embeddings.v1` topic
- [ ] 2.2.5 Handle multi-namespace embedding (dense, sparse, multi-vector)

### 2.3 Indexing Stage

- [ ] 2.3.1 Implement `IndexingWorker` Kafka consumer
- [ ] 2.3.2 Subscribe to `ingest.embeddings.v1` topic
- [ ] 2.3.3 Route embeddings to appropriate vector stores by namespace
- [ ] 2.3.4 Batch upsert for efficiency (50-100 vectors per batch)
- [ ] 2.3.5 Publish completion to `ingest.indexed.v1` topic
- [ ] 2.3.6 Mark job complete in ledger

### 2.4 Job State Management

- [ ] 2.4.1 Implement `JobLedger` with Redis/Postgres backend
- [ ] 2.4.2 Add job creation with initial status (queued)
- [ ] 2.4.3 Implement stage transition tracking (chunking → embedding → indexing)
- [ ] 2.4.4 Add error state handling
- [ ] 2.4.5 Implement retry counter and max attempts
- [ ] 2.4.6 Add job completion timestamp and duration

### 2.5 Error Handling & Retry

- [ ] 2.5.1 Implement exponential backoff for transient errors
- [ ] 2.5.2 Add dead letter queue (DLQ) for permanent failures
- [ ] 2.5.3 Implement retry policy configuration (max attempts, backoff multiplier)
- [ ] 2.5.4 Add error classification (retriable vs permanent)
- [ ] 2.5.5 Implement DLQ monitoring and alerting

## 3. Query Pipeline Orchestration

### 3.1 Retrieval Stage

- [ ] 3.1.1 Implement `RetrievalOrchestrator`
- [ ] 3.1.2 Add parallel fan-out to enabled strategies
- [ ] 3.1.3 Implement per-strategy timeout (50ms default)
- [ ] 3.1.4 Add strategy result collection and validation
- [ ] 3.1.5 Implement graceful degradation (partial results on strategy failure)

### 3.2 Fusion Stage

- [ ] 3.2.1 Implement `FusionOrchestrator`
- [ ] 3.2.2 Add result deduplication by doc_id
- [ ] 3.2.3 Apply configured fusion algorithm (RRF, weighted)
- [ ] 3.2.4 Implement score normalization if needed
- [ ] 3.2.5 Add fusion result validation

### 3.3 Reranking Stage

- [ ] 3.3.1 Implement `RerankOrchestrator`
- [ ] 3.3.2 Add candidate selection (top N for reranking)
- [ ] 3.3.3 Call configured reranker with batch processing
- [ ] 3.3.4 Implement reranking timeout (50ms default)
- [ ] 3.3.5 Add reranking cache lookup and write

### 3.4 Final Selection Stage

- [ ] 3.4.1 Implement `FinalSelectorOrchestrator`
- [ ] 3.4.2 Add top-K selection from reranked results
- [ ] 3.4.3 Implement explain mode (include scores from all stages)
- [ ] 3.4.4 Add result formatting and metadata enrichment

### 3.5 End-to-End Pipeline Executor

- [ ] 3.5.1 Implement `QueryPipelineExecutor`
- [ ] 3.5.2 Chain all query stages sequentially
- [ ] 3.5.3 Add per-stage timing and metrics
- [ ] 3.5.4 Implement total timeout enforcement (100ms target)
- [ ] 3.5.5 Add error handling and partial result return

## 4. Profile Management

### 4.1 Profile Configuration

- [ ] 4.1.1 Implement `ProfileManager` to load YAML profiles
- [ ] 4.1.2 Add profile validation (all referenced components exist)
- [ ] 4.1.3 Create default profiles (PMC, DailyMed, ClinicalTrials.gov)
- [ ] 4.1.4 Implement profile inheritance (base + overrides)

### 4.2 Profile Detection

- [ ] 4.2.1 Implement `ProfileDetector` based on doc metadata
- [ ] 4.2.2 Add source-based detection (e.g., source="openalex" → PMC profile)
- [ ] 4.2.3 Implement explicit profile override via API parameter
- [ ] 4.2.4 Add fallback to default profile

### 4.3 Profile Application

- [ ] 4.3.1 Apply ingestion profile at document intake
- [ ] 4.3.2 Apply query profile based on target source/collection
- [ ] 4.3.3 Log profile selection for audit

## 5. State Management & Resilience

### 5.1 Correlation ID Propagation

- [ ] 5.1.1 Generate UUID correlation ID at request entry
- [ ] 5.1.2 Propagate through all pipeline stages
- [ ] 5.1.3 Include in all logs and metrics
- [ ] 5.1.4 Add to Kafka message headers
- [ ] 5.1.5 Include in HTTP response headers

### 5.2 Circuit Breakers

- [ ] 5.2.1 Implement `CircuitBreaker` for each external service
- [ ] 5.2.2 Add failure threshold configuration (e.g., 5 consecutive failures)
- [ ] 5.2.3 Implement open/half-open/closed states
- [ ] 5.2.4 Add automatic recovery with exponential backoff
- [ ] 5.2.5 Emit alerts on circuit breaker state changes

### 5.3 Timeout Management

- [ ] 5.3.1 Implement `TimeoutManager` per pipeline stage
- [ ] 5.3.2 Add configurable timeouts (retrieval: 50ms, reranking: 50ms)
- [ ] 5.3.3 Implement total pipeline timeout (100ms target)
- [ ] 5.3.4 Add timeout breach logging and metrics

### 5.4 Graceful Degradation

- [ ] 5.4.1 Implement fallback strategies on component failure
- [ ] 5.4.2 Return partial results when possible
- [ ] 5.4.3 Add degraded mode indicator in response
- [ ] 5.4.4 Log degradation events for alerting

## 6. Monitoring & Observability

### 6.1 Prometheus Metrics

- [ ] 6.1.1 Add ingestion pipeline metrics (docs/sec, stage latency, error rate)
- [ ] 6.1.2 Add query pipeline metrics (query/sec, P50/P95/P99 latency, error rate)
- [ ] 6.1.3 Add per-stage metrics (chunking, embedding, retrieval, fusion, reranking)
- [ ] 6.1.4 Add job ledger metrics (queued, processing, completed, failed)
- [ ] 6.1.5 Add circuit breaker state metrics

### 6.2 OpenTelemetry Tracing

- [ ] 6.2.1 Add distributed tracing spans for all stages
- [ ] 6.2.2 Propagate trace context via Kafka headers
- [ ] 6.2.3 Include span attributes (stage name, doc_id, query, etc.)
- [ ] 6.2.4 Configure sampling rate (10% default)
- [ ] 6.2.5 Integrate with Jaeger for visualization

### 6.3 Structured Logging

- [ ] 6.3.1 Add correlation ID to all log entries
- [ ] 6.3.2 Log pipeline stage transitions
- [ ] 6.3.3 Log error details with context
- [ ] 6.3.4 Implement log aggregation (via structlog)
- [ ] 6.3.5 Add sensitive data scrubbing

### 6.4 Alerting

- [ ] 6.4.1 Configure latency threshold alerts (P95 > 200ms)
- [ ] 6.4.2 Configure error rate alerts (>5% errors)
- [ ] 6.4.3 Add circuit breaker state change alerts
- [ ] 6.4.4 Add DLQ accumulation alerts
- [ ] 6.4.5 Configure alert routing (PagerDuty, Slack)

## 7. Evaluation Framework

### 7.1 Ground Truth Management

- [ ] 7.1.1 Implement `GroundTruthManager` (load queries + relevant docs)
- [ ] 7.1.2 Add ground truth dataset schema (queries, doc_ids, relevance labels)
- [ ] 7.1.3 Create annotation interface for new test sets
- [ ] 7.1.4 Store ground truth in versioned files (JSONL)

### 7.2 Retrieval Metrics

- [ ] 7.2.1 Implement nDCG@K (K=1,5,10,20)
- [ ] 7.2.2 Implement Recall@K
- [ ] 7.2.3 Implement MRR (Mean Reciprocal Rank)
- [ ] 7.2.4 Implement MAP (Mean Average Precision)
- [ ] 7.2.5 Add per-query metrics and aggregate statistics

### 7.3 Per-Stage Evaluation

- [ ] 7.3.1 Evaluate chunking quality (boundary F1)
- [ ] 7.3.2 Evaluate embedding quality (similarity correlation)
- [ ] 7.3.3 Evaluate retrieval recall before fusion
- [ ] 7.3.4 Evaluate fusion improvement vs single-strategy
- [ ] 7.3.5 Evaluate reranking improvement vs fusion-only

### 7.4 Evaluation Harness

- [ ] 7.4.1 Implement `EvalHarness` (run evaluation on test set)
- [ ] 7.4.2 Add automated nightly evaluation runs
- [ ] 7.4.3 Generate evaluation reports (markdown + JSON)
- [ ] 7.4.4 Track metrics over time (regression detection)
- [ ] 7.4.5 Compare multiple configurations side-by-side

### 7.5 A/B Testing Framework

- [ ] 7.5.1 Implement `ABTestRunner` (split traffic between configs)
- [ ] 7.5.2 Add experiment configuration (variant A vs B, traffic split)
- [ ] 7.5.3 Track per-variant metrics (latency, accuracy, errors)
- [ ] 7.5.4 Implement statistical significance testing
- [ ] 7.5.5 Generate A/B test reports with recommendations

## 8. Integration with Existing Services

- [ ] 8.1 Integrate ChunkingService via orchestration
- [ ] 8.2 Integrate EmbeddingService via orchestration
- [ ] 8.3 Integrate VectorStoreService via orchestration
- [ ] 8.4 Integrate RerankerService via orchestration
- [ ] 8.5 Add REST API endpoints for ingestion and query pipelines
- [ ] 8.6 Add GraphQL resolvers for pipelines
- [ ] 8.7 Implement SSE streaming for ingestion job progress

## 9. Configuration Management

- [ ] 9.1 Extend YAML schema for complete pipeline configuration
- [ ] 9.2 Add configuration validation (all components exist)
- [ ] 9.3 Implement hot-reload for configuration changes (where safe)
- [ ] 9.4 Add configuration versioning
- [ ] 9.5 Create configuration migration tools

## 10. Testing

- [ ] 10.1 Unit tests for each orchestrator
- [ ] 10.2 Integration tests for ingestion pipeline (end-to-end)
- [ ] 10.3 Integration tests for query pipeline (end-to-end)
- [ ] 10.4 Performance tests (latency, throughput)
- [ ] 10.5 Chaos tests (service failures, timeouts)
- [ ] 10.6 Profile-based tests (PMC, DailyMed, ClinicalTrials.gov)

## 11. Documentation

- [ ] 11.1 Document pipeline architecture and flow
- [ ] 11.2 Create configuration guide (all YAML options)
- [ ] 11.3 Document profile creation and customization
- [ ] 11.4 Add troubleshooting guide (common errors, debugging)
- [ ] 11.5 Document evaluation harness usage
- [ ] 11.6 Create operational runbook (deployment, monitoring, alerting)

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
