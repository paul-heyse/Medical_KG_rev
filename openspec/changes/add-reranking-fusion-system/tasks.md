# Implementation Tasks: Add Reranking & Fusion System

## 1. Core Interfaces & Models

- [x] 1.1 Define `RerankerPort` protocol (score_pairs method)
- [x] 1.2 Create `RerankResult` model (doc_id, score, rank)
- [x] 1.3 Create `FusionStrategy` enum (RRF, weighted, learned)
- [x] 1.4 Define `ScoredDocument` model with normalized scores
- [x] 1.5 Create reranker factory with method selection

## 2. Cross-Encoder Rerankers

### 2.1 BGE Reranker (Primary)

- [x] 2.1.1 Implement `BGEReranker` with BAAI/bge-reranker-v2-m3
- [x] 2.1.2 Add FP16 precision support for GPU
- [x] 2.1.3 Implement batch processing (16-64 pairs)
- [x] 2.1.4 Add ONNX optimization for CPU deployment
- [x] 2.1.5 Implement score normalization to [0, 1]

### 2.2 MiniLM Cross-Encoder

- [x] 2.2.1 Implement `MiniLMReranker` with ms-marco-MiniLM
- [x] 2.2.2 Add CPU and GPU support
- [x] 2.2.3 Optimize for fast inference (<10ms per pair)
- [x] 2.2.4 Add quantization support (int8)

### 2.3 MonoT5 Reranker

- [x] 2.3.1 Implement `MonoT5Reranker` with castorini/monot5
- [x] 2.3.2 Add pointwise relevance prediction
- [x] 2.3.3 Implement prompt formatting for T5
- [x] 2.3.4 Add batch processing for efficiency

### 2.4 Qwen Reranker

- [x] 2.4.1 Implement `QwenReranker` via vLLM endpoint
- [x] 2.4.2 Add OpenAI-compatible API integration
- [x] 2.4.3 Implement prompt templates for reranking
- [x] 2.4.4 Add response parsing and score extraction

## 3. Late-Interaction Reranking (ColBERT)

- [x] 3.1 Implement `ColBERTReranker` base class
- [ ] 3.2 Add RAGatouille integration (ColBERTv2 index)
- [ ] 3.3 Implement Qdrant multivector integration (fetch token vectors)
- [x] 3.4 Add MaxSim computation (max cosine per query token)
- [ ] 3.5 Implement batch MaxSim for efficiency
- [ ] 3.6 Add late-interaction cache for token vectors

## 4. Lexical Reranking

- [x] 4.1 Implement `BM25Reranker` for OpenSearch
- [x] 4.2 Add BM25F multi-field reranking
- [x] 4.3 Implement terms query with candidate IDs filter
- [x] 4.4 Add field boost configuration
- [ ] 4.5 Implement explain mode for score debugging

## 5. Learned-to-Rank (LTR)

### 5.1 OpenSearch LTR

- [x] 5.1.1 Implement `OpenSearchLTRReranker`
- [ ] 5.1.2 Add feature extraction (BM25, SPLADE, dense, recency)
- [ ] 5.1.3 Implement LambdaMART/XGBoost model integration
- [ ] 5.1.4 Add model training pipeline (feature generation)
- [ ] 5.1.5 Implement sltr (Learning to Rank) plugin integration

### 5.2 Vespa Rank Profiles

- [x] 5.2.1 Implement `VespaRankProfileReranker`
- [ ] 5.2.2 Add rank profile definition and deployment
- [ ] 5.2.3 Implement ONNX model integration
- [ ] 5.2.4 Add first-phase and second-phase ranking

## 6. Fusion Algorithms

### 6.1 Reciprocal Rank Fusion (RRF)

- [x] 6.1.1 Implement `RRFFusion` algorithm
- [x] 6.1.2 Add configurable k parameter (default 60)
- [x] 6.1.3 Handle duplicate documents across result lists
- [ ] 6.1.4 Add tie-breaking by original score

### 6.2 Weighted Linear Fusion

- [x] 6.2.1 Implement `WeightedFusion` algorithm
- [x] 6.2.2 Add configurable weights per strategy
- [x] 6.2.3 Implement score normalization (required for weighted)
- [x] 6.2.4 Add weight validation (sum to 1.0)

### 6.3 Score Normalization

- [x] 6.3.1 Implement min-max normalization
- [x] 6.3.2 Implement z-score normalization
- [x] 6.3.3 Implement softmax normalization
- [x] 6.3.4 Add per-strategy normalization

### 6.4 Result Deduplication

- [x] 6.4.1 Implement duplicate detection by doc_id
- [x] 6.4.2 Add score aggregation (max, mean, sum)
- [x] 6.4.3 Preserve highest-ranking instance
- [ ] 6.4.4 Add metadata merging for duplicates

## 7. Two-Stage Retrieval Pipeline

- [x] 7.1 Implement `TwoStagePipeline` orchestrator
- [x] 7.2 Add first-stage retrieval (1000 candidates typical)
- [x] 7.3 Implement candidate selection (top 100 for reranking)
- [x] 7.4 Add reranking stage with batch processing
- [x] 7.5 Implement final top-K selection
- [ ] 7.6 Add stage timing and metrics

## 8. Batch Processing & GPU Optimization

- [x] 8.1 Implement `BatchProcessor` for rerankers
- [x] 8.2 Add dynamic batch size selection (based on GPU memory)
- [ ] 8.3 Implement GPU memory monitoring
- [ ] 8.4 Add async batch processing for multiple queries
- [ ] 8.5 Implement batch timeout and fallback
- [ ] 8.6 Add FP16 precision support for GPU rerankers

## 9. Caching System

- [x] 9.1 Implement `RerankCacheManager` with TTL
- [ ] 9.2 Add Redis-based cache backend
- [x] 9.3 Implement cache key generation (query + doc_id + model)
- [ ] 9.4 Add cache invalidation on index updates
- [x] 9.5 Implement cache hit rate monitoring
- [ ] 9.6 Add cache warming for popular queries

## 10. Configuration & Validation

- [x] 10.1 Extend YAML schema for reranking configuration
- [x] 10.2 Add fusion algorithm configuration
- [x] 10.3 Implement reranker method validation
- [ ] 10.4 Add model availability checks
- [x] 10.5 Implement GPU availability validation for GPU rerankers
- [ ] 10.6 Add configuration migration utilities

## 11. Integration with Retrieval Service

- [x] 11.1 Extend `RetrievalService` to include reranking stage
- [x] 11.2 Add fusion algorithm application after multi-strategy retrieval
- [x] 11.3 Implement optional reranking (enable/disable via config)
- [x] 11.4 Add result deduplication before reranking
- [ ] 11.5 Implement explain mode (show scores at each stage)

## 12. Evaluation Harness

- [x] 12.1 Create reranker comparison framework
- [x] 12.2 Implement nDCG@K evaluation (K=1,5,10,20)
- [x] 12.3 Add Recall@K evaluation
- [x] 12.4 Implement MRR (Mean Reciprocal Rank) metric
- [x] 12.5 Add latency profiling (P50, P95, P99)
- [ ] 12.6 Create accuracy vs latency trade-off curves
- [ ] 12.7 Implement A/B testing framework for rerankers
- [ ] 12.8 Add leaderboard for reranker comparison

## 13. Testing

- [x] 13.1 Unit tests for each reranker implementation
- [x] 13.2 Unit tests for fusion algorithms
- [ ] 13.3 Unit tests for score normalization
- [x] 13.4 Integration tests for two-stage pipeline
- [ ] 13.5 Performance tests for batch processing
- [ ] 13.6 GPU integration tests (with availability checks)
- [x] 13.7 Cache behavior tests (hit rate, invalidation)
- [x] 13.8 End-to-end tests with multi-strategy retrieval + reranking

## 14. Documentation

- [ ] 14.1 Document `RerankerPort` interface
- [ ] 14.2 Create reranker selection guide (when to use which)
- [ ] 14.3 Document fusion algorithms and trade-offs
- [ ] 14.4 Add YAML configuration examples
- [ ] 14.5 Document batch processing and GPU optimization
- [ ] 14.6 Create troubleshooting guide (OOM, slow reranking, cache misses)
- [ ] 14.7 Add evaluation harness usage guide

## 15. Operations & Monitoring

- [x] 15.1 Add Prometheus metrics (reranking latency, batch size, GPU util)
- [ ] 15.2 Implement reranker health checks
- [ ] 15.3 Add cache hit rate monitoring
- [ ] 15.4 Create alerts for high reranking latency
- [x] 15.5 Implement model version tracking
- [ ] 15.6 Add GPU memory usage alerts

## Dependencies

- **Upstream**: `add-vector-storage-retrieval` (retrieval results are reranked)
- **Downstream**: `add-retrieval-pipeline-orchestration` (orchestration uses reranking)

## Estimated Effort

- Core interfaces & cross-encoder rerankers: 1.5 weeks
- Late-interaction & lexical reranking: 1 week
- LTR integration (OpenSearch, Vespa): 1 week
- Fusion algorithms & score normalization: 0.5 week
- Two-stage pipeline & batch processing: 1 week
- Caching & GPU optimization: 0.5 week
- Evaluation harness & testing: 1 week
- **Total**: 6.5 weeks

## Success Criteria

- [ ] All rerankers implement `RerankerPort` interface
- [ ] nDCG@10 improvement of 5-15% with reranking vs fusion-only
- [ ] P95 reranking latency <50ms for 100 candidates (batch size 32, FP16)
- [ ] Cache hit rate >30% for repeated queries
- [ ] RRF and weighted fusion functional and tested
- [ ] GPU utilization >70% during batch reranking
- [ ] Support for 4+ reranking methods (CE, late-interaction, lexical, LTR)
- [ ] Evaluation harness produces accuracy/latency trade-off curves
