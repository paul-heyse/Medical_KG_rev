# Proposal: Add Reranking & Fusion System

## Why

Multi-strategy retrieval returns candidates from diverse sources (BM25, SPLADE, dense vectors), but these results require intelligent fusion and reranking to maximize accuracy and relevance for biomedical queries.

**Current limitations**:

- No unified interface for reranking methods (cross-encoders, late-interaction, LTR)
- Missing fusion algorithms to combine multi-strategy results (RRF, weighted)
- Lack of result normalization across different scoring schemes
- No evaluation framework for reranker selection

**Opportunity**: Implement a comprehensive reranking and fusion system with:

- Universal `RerankerPort` protocol supporting 5+ reranking methods
- Fusion algorithms (RRF, weighted linear, learned-to-rank)
- Cross-encoder rerankers (BGE, MiniLM, MonoT5, Qwen)
- Late-interaction reranking (ColBERTv2 MaxSim)
- Lexical reranking (BM25/BM25F on candidate set)
- GPU-accelerated batch reranking
- Evaluation harness for reranker comparison

This system enables 5-15% nDCG@10 improvement over fusion-only via intelligent candidate reranking.

## What Changes

### Core Capabilities

1. **Universal Reranker Interface**
   - `RerankerPort` protocol with `score_pairs` method
   - Support for batch scoring (16-64 pairs typical)
   - Normalized scores (0-1 range) for comparison
   - GPU-accelerated where applicable

2. **Cross-Encoder Rerankers**
   - **BGE-reranker-v2-m3**: SOTA biomedical reranker (BAAI)
   - **MiniLM cross-encoder**: Fast CPU/GPU reranker
   - **MonoT5**: T5-based pointwise reranker (slower, high quality)
   - **Qwen reranker**: Served via vLLM for large models
   - ONNX optimization for CPU deployment

3. **Late-Interaction Reranking**
   - **ColBERTv2**: MaxSim over token vectors
   - Integration with ColBERT indexer or Qdrant multivector storage
   - Efficient for abbreviation-heavy queries (drugs, outcomes)

4. **Lexical Reranking**
   - **BM25 rerank**: Re-score candidates via OpenSearch terms query
   - **BM25F rerank**: Multi-field weighted reranking
   - Fast and effective for exact-term precision

5. **Learned-to-Rank (LTR)**
   - **OpenSearch LTR**: LambdaMART/XGBoost feature-based ranking
   - **Vespa rank profiles**: ONNX model integration
   - Features: BM25 score, SPLADE score, dense score, recency, etc.

6. **Fusion Algorithms**
   - **RRF (Reciprocal Rank Fusion)**: Parameter-free, robust baseline
   - **Weighted linear fusion**: Configurable weights per strategy
   - **Score normalization**: Min-max, z-score, softmax
   - **Result deduplication**: Merge duplicates by doc_id

7. **Reranking Pipeline**
   - **Two-stage retrieval**: Fast retrieval (1000 candidates) → rerank (top 100) → return (10)
   - **Batch processing**: Score 16-64 query-doc pairs per batch for GPU efficiency
   - **Caching**: Cache reranking scores for repeated queries

### Configuration

```yaml
reranking:
  enabled: true
  method: cross_encoder  # or late_interaction, lexical, ltr
  model: BAAI/bge-reranker-v2-m3
  batch_size: 32
  gpu_enabled: true
  cache_ttl: 3600  # seconds

  cross_encoder:
    model: BAAI/bge-reranker-v2-m3
    device: cuda:0
    precision: fp16
    onnx_optimize: false  # Set true for CPU deployment

  late_interaction:
    model: colbert-ir/colbertv2.0
    source: colbert_index  # or qdrant_multivector
    max_doc_tokens: 200

  lexical:
    engine: opensearch
    method: bm25f
    field_weights:
      title: 2.0
      body: 1.0

  ltr:
    engine: opensearch  # or vespa
    model_name: biomedical_ranker_v1
    features:
      - bm25_score
      - splade_score
      - dense_score
      - title_length
      - recency_days

fusion:
  method: rrf  # or weighted, learned
  k: 60  # RRF parameter
  weights:  # For weighted fusion
    dense: 0.35
    splade: 0.50
    bm25: 0.15
  normalize: true
  deduplicate: true

pipeline:
  retrieve_candidates: 1000  # First-stage retrieval
  rerank_candidates: 100     # Pass to reranker
  return_top_k: 10           # Final results
```

### Implementation Structure

```
med/
  rerank/
    ports.py                   # RerankerPort protocol
    ce_bge_reranker.py         # BGE cross-encoder
    ce_mini_lm.py              # MiniLM cross-encoder
    ce_monoT5.py               # MonoT5 pointwise
    ce_qwen_reranker.py        # Qwen via vLLM
    colbert_reranker.py        # ColBERTv2 MaxSim
    lexical_reranker.py        # BM25/BM25F rerank
    ltr_ranker.py              # OpenSearch/Vespa LTR
  fusion/
    rrf_fusion.py              # Reciprocal Rank Fusion
    weighted_fusion.py         # Weighted linear fusion
    score_normalization.py     # Min-max, z-score, softmax
    deduplication.py           # Merge duplicate results
  pipeline/
    two_stage_pipeline.py      # Retrieve → rerank → return
    batch_processor.py         # Batch scoring for GPU
    cache_manager.py           # TTL-based score caching
```

## Impact

### Affected Specs

- **New**: `reranking-system` (cross-encoder, late-interaction, lexical, LTR)
- **New**: `fusion-algorithms` (RRF, weighted, score normalization)
- **Modified**: `retrieval-orchestration` (integrate reranking pipeline)

### Affected Code

- New package: `med/rerank/` for reranker implementations
- New package: `med/fusion/` for fusion algorithms
- New package: `med/pipeline/` for two-stage retrieval
- Integration: `RetrievalService` → add reranking stage after fusion
- Configuration: Extend YAML schema for reranking and fusion settings

### Benefits

1. **Accuracy**: 5-15% nDCG@10 improvement with reranking vs fusion-only
2. **Precision**: Cross-encoders excel at biomedical terminology disambiguation
3. **Efficiency**: Two-stage pipeline reduces compute (1000→100→10 vs rerank all)
4. **Flexibility**: Swap rerankers via configuration without code changes
5. **Robustness**: RRF fusion handles score distribution differences across strategies

### Migration

- New system; no migration required
- Reranking is optional (`reranking.enabled: false` to disable)
- Fusion algorithms can be used standalone without reranking

### Risks

1. **Latency overhead**: Reranking adds 20-50ms P95 latency
   - **Mitigation**: Batch processing, GPU acceleration, cache repeated queries
2. **GPU dependency**: Cross-encoders benefit from GPU
   - **Mitigation**: ONNX quantization for CPU, fallback to lexical rerank
3. **Model selection complexity**: Multiple reranking methods available
   - **Mitigation**: Provide tuned defaults, evaluation harness for comparison
4. **Cache coherency**: Cached scores may become stale
   - **Mitigation**: TTL-based expiration, invalidate on index updates

### Success Metrics

- nDCG@10 improvement of 5-15% with reranking vs fusion-only
- P95 reranking latency <50ms for 100 candidates (batch size 32, FP16)
- Cache hit rate >30% for repeated queries
- Support for 4+ reranking methods (cross-encoder, late-interaction, lexical, LTR)
- GPU utilization >70% during batch reranking
