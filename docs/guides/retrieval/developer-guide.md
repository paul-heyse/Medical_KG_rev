# Retrieval Developer Guide

This guide documents the internals of the hybrid retrieval, fusion, reranking, and evaluation subsystems introduced by the `add-retrieval-ranking-evaluation` OpenSpec change. It is organised by engineering task to provide implementation context for future contributors.

## 1. Architecture Overview

```
┌─────────────────────────┐
│ Gateway (FastAPI/gRPC)  │
│  • Request validation   │
│  • Tenant security      │
└────────────┬────────────┘
             │ RetrieveRequest / EvaluateRequest
┌────────────▼────────────┐
│ RetrievalService        │
│  • Hybrid fan-out       │
│  • Fusion + reranking   │
│  • Routing heuristics   │
└────────────┬────────────┘
             │ HybridSearchCoordinator
┌────────────▼────────────┐
│ Component adapters      │
│  • OpenSearch BM25/SPLADE│
│  • FAISS dense vectors  │
│  • VectorStoreService   │
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│ RerankingEngine         │
│  • Model registry       │
│  • Cache + batching     │
│  • Circuit breaker      │
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│ EvaluationRunner        │
│  • Metrics (Recall/nDCG)│
│  • Bootstrapping        │
│  • Prometheus export    │
└─────────────────────────┘
```

Key design choices:

- **Asynchronous fan-out**: `HybridSearchCoordinator` executes BM25, SPLADE, and dense KNN concurrently using `asyncio.gather`, enforcing per-component timeouts and caching.
- **Explainability**: All stages attach metadata (`component_scores`, `timings_ms`, `rerank` outcome) so clients can diagnose ranking decisions.
- **Extensibility**: Both retrieval components and rerankers are plugin-based (callable protocols / `RerankerFactory`), enabling drop-in replacements.

## 2. Hybrid Coordinator Implementation

- **Location**: `services/retrieval/hybrid.py`
- **Entry point**: `HybridSearchCoordinator.search()` accepts index/query/top-k and resolves the component list via `HybridComponentSettings.resolve_components()`.
- **Caching**: `InMemoryHybridCache` implements a simple asyncio lock + dict cache. Production deployments should swap this for Redis/KeyDB by implementing the `CacheProtocol` interface.
- **Timeouts**: `HybridComponentSettings.timeout_for(component)` converts timeouts to seconds. The coordinator guards each component call with `asyncio.wait_for` and records per-component latency in `HybridSearchResult.timings_ms`.
- **Extending components**: Register a new callable in the `components` mapping passed to the constructor. Each component must accept keyword-only parameters (`index`, `query`, `k`, `filters`, etc.) and return an iterable of candidate dictionaries.

## 3. Fusion Algorithms

- **Default RRF**: Implemented in `services/reranking/fusion/rrf.py`. Scores are accumulated as `score += 1 / (rank + k)` with `k` configurable (default 60). Duplicate document IDs are merged before fusion.
- **Weighted normalisation**: Implemented in `services/reranking/fusion/weighted.py`. Component scores are min–max normalised and combined using the provided weights. Normalisation strategies include min–max and z-score (see `NormalizationStrategy`).
- **Switching algorithms**: `RerankingSettings.fusion.strategy` controls the global default. Per-request overrides are exposed through API parameters (`fusion_method=weighted`).
- **Tie-breaking**: When fused scores tie, the hybrid coordinator falls back to the primary component (BM25) to ensure deterministic ordering.

## 4. Reranking Integration

- **Pipeline**: `TwoStagePipeline` orchestrates fusion + reranking. It calls `FusionService` to build a candidate pool then `RerankingEngine` when reranking is enabled.
- **Model registry**: `RerankerModelRegistry` loads `config/retrieval/reranking_models.yaml`, validates metadata, and caches `ModelHandle` instances. Models are downloaded via `huggingface_hub` and stored under `model_cache/rerankers/`.
- **Caching**: `RerankCacheManager` caches query/document pairs keyed by reranker ID. Cache TTL is controlled by `RerankingSettings.cache_ttl`.
- **Batching**: `BatchProcessor` splits candidate lists into GPU-friendly batches (default 64). Failed batches trip the circuit breaker (`CircuitBreaker` with configurable failure thresholds).
- **GPU enforcement**: `CrossEncoderReranker` checks `model.requires_gpu` and raises `GpuNotAvailableError` if CUDA is unavailable, triggering the fallback path in `RetrievalService.search()`.

## 5. Table Routing Logic

- **Classifier**: `IntentClassifier` (in `services/retrieval/routing/intent_classifier.py`) performs keyword/regex matching and returns `QueryIntent` with confidence.
- **Boost application**: The classifier output feeds into the OpenSearch query builder (see `services/retrieval/query_dsl.py`) which applies multiplicative boosts to table chunks (`is_table=true` or `intent_hint=*`).
- **Manual overrides**: API requests supply `query_intent` or `table_only`. Overrides short-circuit the classifier and set confidence to 1.0.
- **Extending keywords**: Pass custom keyword weights to `IntentClassifier(tabular_keywords={...})` when instantiating per-tenant routers.

## 6. Clinical Boosting Implementation

- **Intent analysis**: Clinical intents (eligibility, adverse events, dosage, results, methods) are derived from chunk metadata produced during ingestion. The router inspects `metadata.intent_hint` and `metadata.section_label`.
- **Boost factors**: Configured in `services/retrieval/query_dsl.py` via weight tables per intent. Boosts are applied through OpenSearch `function_score` queries.
- **Extensibility**: Add new intents by updating the enum/constants in `services/retrieval/router.py` and augmenting the boost table. Ensure ingestion emits matching `intent_hint` tags.

## 7. Evaluation Framework

- **Data model**: `TestSetManager` loads YAML files into `TestSet`/`QueryJudgment` instances, validating schema constraints.
- **Metrics**: `services/evaluation/metrics.py` implements Recall@K, Precision@K, nDCG@K (via `sklearn.metrics.ndcg_score`), MRR, and MAP. `RankingMetrics` aggregates per-query values.
- **Runner**: `EvaluationRunner` executes retrieval functions, captures per-query metrics, summarises means/medians/std dev, and publishes Prometheus gauges (`medicalkg_retrieval_recall_at_k`, etc.).
- **CLI harness**: `tests/performance/run_retrieval_benchmarks.py` wraps the runner, hits the live gateway, and emits JSON summaries suitable for CI pipelines and dashboards.

## 8. API Changes

- **REST**: `/v1/retrieve` accepts new fields (`rerank`, `rerank_model`, `rerank_top_k`, `query_intent`, `table_only`, `explain`). Responses include `meta.rerank` metadata, component timing/error arrays, and `stage_timings` for end-to-end latency.
- **Evaluation endpoint**: `/v1/evaluate` ingests curated queries (`EvaluationRequest`) and returns aggregate metrics (`EvaluationResponse`). Intended for CI/CD and nightly CronJobs.
- **GraphQL**: `RetrieveInput` mirrors REST additions. Schema updates are captured under `docs/schema.graphql`.
- **gRPC**: Proto files expose equivalent fields to ensure parity across protocols.

## 9. Developer Setup Guide

1. Install dependencies: `pip install -r requirements.txt` (requires Python 3.12 and CUDA toolkit when using GPU rerankers).
2. Start local services: `docker-compose up -d` brings up OpenSearch, FAISS, Redis, and supporting services.
3. Seed indices: run ingestion pipelines or load fixtures via `scripts/seed_sample_data.py` (if available) so hybrid retrieval has candidate documents.
4. Launch gateway: `uvicorn Medical_KG_rev.gateway.app:create_app --factory --reload`.
5. Execute performance harness: `k6 run tests/performance/hybrid_suite.js` and `python tests/performance/run_retrieval_benchmarks.py` to validate SLOs.
6. Optional: point `MK_RERANKING__MODEL__MODEL` to alternative reranker IDs for local experimentation. Models cache under `model_cache/rerankers`.

## 10. Configuration Files

| File | Purpose | Key Fields |
| ---- | ------- | ---------- |
| `config/retrieval/components.yaml` | Controls hybrid component enablement, timeouts, and query expansion synonyms. | `defaults.enable_splade`, `defaults.enable_dense`, `components.<name>.timeout_ms`, `synonyms.<term>` |
| `config/retrieval/reranking.yaml` | Per-tenant reranking defaults and A/B test split. | `default_enabled`, `tenants.<tenant>=bool`, `experiment.rerank_ratio` |
| `config/retrieval/reranking_models.yaml` | Model registry for rerankers. | `models.<key>.model_id`, `requires_gpu`, `version`, `metadata.latency_profile` |
| `eval/test_sets/*.yaml` | Gold-standard evaluation datasets. | `version`, `queries[].query_type`, `queries[].relevant_docs[].grade` |
| `ops/k8s/base/configmap-retrieval.yaml` *(added)* | Deployment-specific overrides for the three configs above. Mounted into `/app/config/retrieval` for staging/production environments. | Mirrors the files above; overlays patch tenant toggles and rerank ratios. |

Refer to `Medical_KG_rev.config.settings.RerankingSettings` for the full Pydantic model that consumes these files and exposes environment-variable overrides (`MK_RERANKING__...`).
