# Proposal: Add Retrieval Pipeline Orchestration

## Why

The modular document retrieval pipeline requires sophisticated orchestration to integrate chunking, embedding, vector storage, retrieval, and reranking into a cohesive end-to-end system for production-scale biomedical knowledge integration.

**Current limitations**:

- No unified orchestration layer connecting all retrieval components
- Missing ingestion-time pipeline (chunk → embed → index)
- Lack of query-time pipeline (retrieve → fuse → rerank → return)
- No configuration-driven pipeline assembly without code changes
- Missing evaluation and A/B testing infrastructure

**Opportunity**: Implement a comprehensive orchestration system with:

- Ingestion pipeline: Document → chunk → embed → index to vector stores
- Query pipeline: Query → retrieve (multi-strategy) → fuse → rerank → results
- Configuration-driven pipeline assembly via YAML
- Async job processing with Kafka for batch ingestion
- Real-time query orchestration with <100ms P95 latency
- Evaluation harness for end-to-end retrieval quality
- A/B testing framework for strategy comparison

This system enables production-ready, high-performance retrieval with 15-20% nDCG@10 improvement vs baseline through optimized multi-stage pipelines.

## What Changes

### Core Capabilities

1. **Ingestion Pipeline Orchestration**
   - Document → Chunking Service → chunks
   - Chunks → Embedding Service → embeddings (dense, sparse, multi-vector)
   - Embeddings → Vector Store Service → indexed vectors
   - Async processing via Kafka topics
   - Job state tracking via ledger
   - Retry logic and dead letter queue

2. **Query Pipeline Orchestration**
   - Query → multi-strategy retrieval (dense, BM25, SPLADE, ColBERT)
   - Parallel fan-out to vector stores
   - Fusion (RRF, weighted) → unified candidate list
   - Reranking (cross-encoder, late-interaction) → final results
   - <100ms P95 end-to-end latency
   - Synchronous request/response

3. **Configuration-Driven Assembly**
   - YAML-defined pipelines with stages and parameters
   - Per-source profile selection (PMC, DailyMed, ClinicalTrials.gov)
   - Strategy enable/disable flags
   - Parameter tuning without code changes

4. **State Management & Monitoring**
   - Job ledger for ingestion state tracking
   - Progress events via Server-Sent Events (SSE)
   - Prometheus metrics per pipeline stage
   - OpenTelemetry distributed tracing
   - Error tracking and alerting

5. **Evaluation & A/B Testing**
   - Ground truth dataset management
   - End-to-end retrieval metrics (nDCG@K, Recall@K, MRR)
   - Per-stage metrics (chunking, embedding, retrieval, reranking)
   - A/B test framework for strategy comparison
   - Experiment tracking and result analysis

6. **Production Features**
   - Circuit breakers for failing services
   - Timeout management per stage
   - Graceful degradation (fallback strategies)
   - Request correlation IDs
   - Multi-tenant isolation

### Configuration

```yaml
ingestion_pipeline:
  enabled: true
  kafka:
    topics:
      requests: ingest.chunking.v1
      results: ingest.indexed.v1
  stages:
    - name: chunking
      service: ChunkingService
      config:
        primary: semantic_splitter
        target_tokens: 600
    - name: embedding
      service: EmbeddingService
      config:
        namespaces:
          - dense.bge.1024.v1
          - sparse.splade.v3.v1
    - name: indexing
      service: VectorStoreService
      config:
        driver: qdrant

query_pipeline:
  enabled: true
  stages:
    - name: retrieval
      strategies:
        - dense
        - bm25
        - splade
      top_k_per_strategy: 1000
    - name: fusion
      method: rrf
      k: 60
    - name: reranking
      method: cross_encoder
      model: BAAI/bge-reranker-v2-m3
      rerank_top_k: 100
    - name: final_selection
      return_top_k: 10

profiles:
  pmc:  # PubMed Central full-text papers
    ingestion:
      chunking:
        primary: semantic_splitter
        target_tokens: 650
    query:
      strategies: [dense, splade]
      fusion: rrf
      reranking: cross_encoder

  dailymed:  # Drug labels (structured)
    ingestion:
      chunking:
        primary: section_aware
        target_tokens: 450
    query:
      strategies: [dense, bm25]
      fusion: weighted
      weights: {dense: 0.6, bm25: 0.4}
      reranking: lexical

  ctgov:  # Clinical trials
    ingestion:
      chunking:
        primary: clinical_role
        target_tokens: 350
    query:
      strategies: [dense, splade, bm25]
      fusion: rrf
      reranking: late_interaction

monitoring:
  metrics:
    enabled: true
    pipeline_latency: true
    stage_latency: true
    error_rate: true
  tracing:
    enabled: true
    sample_rate: 0.1
  alerting:
    latency_threshold_ms: 200
    error_rate_threshold: 0.05
```

### Implementation Structure

```
med/
  orchestration/
    ingestion/
      chunking_worker.py       # Kafka consumer: doc → chunks
      embedding_worker.py      # Kafka consumer: chunks → embeddings
      indexing_worker.py       # Kafka consumer: embeddings → vector store
      job_ledger.py            # State tracking (Redis/Postgres)
      retry_handler.py         # Exponential backoff, DLQ
    query/
      retrieval_orchestrator.py  # Multi-strategy retrieval
      fusion_orchestrator.py     # Apply fusion algorithm
      rerank_orchestrator.py     # Apply reranker
      pipeline_executor.py       # End-to-end query pipeline
    profiles/
      profile_manager.py       # Load and apply per-source profiles
      profile_detector.py      # Detect source from doc metadata
    state/
      correlation_id.py        # Generate and propagate trace IDs
      circuit_breaker.py       # Fail-fast for failing services
      timeout_manager.py       # Per-stage timeout enforcement
    evaluation/
      eval_harness.py          # End-to-end retrieval evaluation
      ab_test_runner.py        # A/B test framework
      metrics_aggregator.py    # Collect and analyze metrics
```

## Impact

### Affected Specs

- **New**: `ingestion-orchestration` (chunking → embedding → indexing pipeline)
- **New**: `query-orchestration` (retrieval → fusion → reranking pipeline)
- **New**: `evaluation-framework` (end-to-end metrics and A/B testing)
- **Modified**: All prior retrieval specs (integrated via orchestration)

### Affected Code

- New package: `med/orchestration/ingestion/` for async ingestion pipeline
- New package: `med/orchestration/query/` for real-time query pipeline
- New package: `med/orchestration/profiles/` for per-source configuration
- New package: `med/orchestration/evaluation/` for testing and metrics
- Integration: All retrieval services connected via orchestrators
- Configuration: Master YAML schema for complete pipeline

### Benefits

1. **End-to-end integration**: Seamless flow from documents to query results
2. **Performance**: <100ms P95 query latency via optimized pipeline
3. **Accuracy**: 15-20% nDCG@10 improvement through multi-stage optimization
4. **Flexibility**: Per-source profiles enable optimal strategies per data type
5. **Observability**: Complete visibility into pipeline performance and errors
6. **Maintainability**: Configuration-driven changes without code deployment

### Migration

- New system; existing retrieval (if any) continues unchanged
- Gradual rollout via per-source profiles
- Can run old and new pipelines in parallel during transition
- A/B testing framework enables data-driven migration decisions

### Risks

1. **Complexity**: Multi-stage pipeline increases operational complexity
   - **Mitigation**: Comprehensive monitoring, circuit breakers, graceful degradation
2. **Latency accumulation**: Each stage adds latency
   - **Mitigation**: Parallel execution, aggressive timeouts, caching
3. **State management**: Job ledger and correlation IDs add overhead
   - **Mitigation**: Efficient Redis/Postgres schemas, async processing
4. **Profile proliferation**: Too many per-source profiles
   - **Mitigation**: Start with 3 profiles (PMC, DailyMed, ClinicalTrials.gov), expand as needed

### Success Metrics

- **Ingestion**: Process 100+ documents/second with <1% error rate
- **Query latency**: P95 <100ms end-to-end (retrieve → rerank → return)
- **Accuracy**: 15-20% nDCG@10 improvement vs baseline (BM25-only)
- **Reliability**: 99.9% uptime, automatic failover on service errors
- **Observability**: 100% requests traced with correlation IDs
- **Evaluation**: Automated nightly runs comparing all configurations
