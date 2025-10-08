# Reranking & Fusion Guide

## RerankerPort Interface

All rerankers implement the `RerankerPort` interface defined under `Medical_KG_rev.services.reranking.ports`. The contract requires a `score_pairs` method that accepts a sequence of `QueryDocumentPair` instances and returns a `RerankingResponse` containing ordered `RerankResult` entries. Implementations should:

- Validate tenant isolation before scoring.
- Support configurable batch sizes and honour the `top_k` limiter.
- Normalise scores to the `[0, 1]` interval for downstream fusion compatibility.
- Respect the `requires_gpu` flag to fail fast when GPU acceleration is mandatory.

## Selecting a Reranker

| Scenario | Recommended Reranker | Notes |
| --- | --- | --- |
| Maximum quality, GPU available | `cross_encoder:bge` | Uses FP16 acceleration when deployed on CUDA devices. |
| Low latency CPU workloads | `cross_encoder:minilm` | Balances lexical and dense features with optional INT8 quantisation. |
| Generative scoring | `cross_encoder:monot5` | Applies prompt-style relevance estimation. |
| LLM-backed reranking | `cross_encoder:qwen` | Calls vLLM/OpenAI compatible endpoints. |
| ColBERT late interaction | `late_interaction:colbert_index` | Fetches token vectors from an external ColBERT index. |
| OpenSearch first/second phase ranking | `ltr:opensearch` | Integrates with SLTR feature stores. |

## Model Registry & Selection

- **Configuration**: `config/retrieval/reranking_models.yaml` lists supported rerankers, their HuggingFace identifiers, and rollout metadata. Update this file when onboarding a new model; manifests are cached under `model_cache/rerankers/`.
- **API Overrides**: Clients may select a model via `GET /v1/search?rerank_model=ms-marco-minilm-l12-v2` (REST), GraphQL `RetrieveInput.rerank_model`, or the pipeline query payload. Unknown models gracefully fall back to the default while annotating the response metadata with a `warnings: ["model_fallback"]` entry.
- **Versioning**: Retrieval responses expose `rerank.metrics.model.version` so dashboards and CI checks can track upgrades across deployments.
- **A/B Testing**: Pair the model registry with `Medical_KG_rev.services.evaluation.ABTestRunner` to compare `nDCG@10` deltas before promoting a challenger model. A +5% uplift is required before enabling reranking by default for a tenant.

## Fusion Algorithms & Trade-offs

- **Reciprocal Rank Fusion (RRF)**: Fast heuristic that blends rankings from multiple retrievers. Tie-breaking now honours original retrieval scores to maintain deterministic ordering.
- **Weighted Fusion**: Applies min-max normalisation before combining strategies with configured weights. Validations ensure weights sum to 1.
- **Deduplication**: Duplicate documents merge metadata, highlights, and per-strategy scores before final ranking.

## YAML Configuration Examples

```yaml
reranking:
  enabled: true
  cache_ttl: 1800
  model:
    reranker_id: cross_encoder:bge
    device: cuda:0
    precision: fp16
  fusion:
    strategy: rrf
    rrf_k: 90
  pipeline:
    retrieve_candidates: 1500
    rerank_candidates: 200
    return_top_k: 20
```

Legacy configuration documents can be converted with `Medical_KG_rev.config.migrate_reranking_config` which supports `model_name`, `fusion_strategy`, and `cacheTtl` keys.

## Batch Processing & GPU Optimisation

- `BatchProcessor` adapts batch sizes based on live GPU memory snapshots and can operate asynchronously for multi-query reranking.
- FP16 precision is automatically leveraged for BGE rerankers on CUDA devices; set `precision: fp16` in configuration.
- Long-running batches trigger automatic splits and issue Prometheus alerts for potential GPU saturation.

## Troubleshooting

| Symptom | Likely Cause | Resolution |
| --- | --- | --- |
| `GPUUnavailableError` | Reranker requires CUDA but none detected | Update deployment targets or disable GPU-only rerankers. |
| Low cache hit rate | Index updates invalidated cache | Use cache warming via `RerankingEngine.warm_cache` for popular queries. |
| Slow reranking latency | Oversized batches triggering splits | Check `retrieval_pipeline_stage_duration_seconds` metrics and reduce `rerank_candidates`. |

## Evaluation Harness Usage

```python
from Medical_KG_rev.services.reranking.evaluation.harness import RerankerEvaluator

evaluator = RerankerEvaluator(ground_truth={"q1": {"doc-1", "doc-2"}})
result = evaluator.evaluate("cross_encoder:bge", {"q1": ["doc-1", "doc-3"]}, [12, 18, 22])
curve = evaluator.build_tradeoff_curve([result])
leaderboard = evaluator.leaderboard([result])
```

Trade-off curves return `(latency_p95_ms, ndcg_at_10)` points, while `ab_test` reports metric deltas between baseline and challenger rerankers.
