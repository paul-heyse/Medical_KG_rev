# Retrieval User Guide

This guide explains how to use the hybrid retrieval stack that combines BM25, SPLADE, dense semantic search, optional cross-encoder reranking, table-aware routing, and clinical intent boosting. Each section maps directly to the user documentation tasks defined in the OpenSpec change proposal.

## 1. Hybrid Retrieval Overview

Hybrid retrieval fan-outs BM25, SPLADE, and dense KNN search in parallel and fuses the results.

- **When to use**: Default for all tenants. Hybrid improves Recall@10 from 65% → 82% on the evaluation set.
- **Configuration**: `config/retrieval/components.yaml` controls enabled components, timeouts, and per-tenant overrides. Override the ConfigMap (`retrieval-config`) when a tenant needs to disable SPLADE or dense search temporarily.
- **Failure handling**: If a component times out the coordinator excludes it from fusion, logs a warning (`retrieval.component_failed`), and returns partial results annotated with `errors` metadata.
- **Latency expectation**: P95 latency is ≤130 ms for hybrid-only queries under 100 QPS.

## 2. Fusion Methods (RRF vs Weighted)

Two fusion strategies are supported:

- **Reciprocal Rank Fusion (default)**: Parameter-free, order independent, and resilient to score scaling. RRF uses `k=60` and is ideal for general-purpose search.
- **Weighted Normalisation Fusion**: Allows explicit component weights (e.g. `weights={"bm25":0.3,"splade":0.4,"dense":0.3}`) with min–max normalisation. Use when domain experts want to bias towards a component (e.g. dense-first paraphrase search).

Switch methods through the reranking settings (`MK_RERANKING__FUSION__STRATEGY=weighted`) or via the REST API (`/v1/search?fusion=weighted`). Weighted fusion requires more score tuning and is recommended for power users only.

## 3. Reranking Guide

Cross-encoder reranking re-scores the fused top-N documents to improve precision on nuanced queries.

- **Models**: Default `BAAI/bge-reranker-base` (balanced quality); alternatives include `ms-marco-MiniLM-L-12-v2` (fast CPU) and `colbert-reranker-v2` (GPU, highest recall).
- **Enabling**: Pass `rerank=true` (REST), set `RetrieveInput.rerank=true` (GraphQL), or enable per-tenant defaults in `config/retrieval/reranking.yaml`.
- **GPU requirements**: GPU-only models (`requires_gpu: true`) fail fast with `GpuNotAvailableError`. Staging and production patches request one NVIDIA GPU per gateway pod.
- **Cost / latency**: Expect +120–150 ms P95 latency for 100 candidate reranking. Use the evaluation harness before enabling by default for a tenant.

## 4. Table Routing Guide

Table-aware routing detects tabular intent and boosts structured chunks.

- **Keywords**: The intent classifier looks for phrases such as “adverse events”, “effect sizes”, “outcome measures”, “results table”.
- **Boosting**: Confidence score maps to `boost = 1 + (2 × confidence)` giving 1–3× boost for table chunks (`is_table=true` or `intent_hint="ae"`).
- **Forcing table-only mode**: Use `table_only=true` to return only table chunks (e.g. “show me all adverse event tables”).
- **Manual override**: `query_intent=tabular` sets boost to the maximum regardless of keyword match.

## 5. Clinical Boosting Guide

Clinical intent boosting prioritises sections matching eligibility, safety, outcomes, dosage, and indications.

- **Detection**: Queries run through the intent classifier. Example: “eligibility criteria for breast cancer trials” → `ELIGIBILITY` intent.
- **Boost factors**: Eligibility (3×), Adverse Events (2×), Results (2×), Methods (1.5×), Dosage (1.5×).
- **Configuration**: Adjust intent keywords/weights in `IntentClassifier` (see `services/retrieval/routing/intent_classifier.py`) or refine section labels during indexing to strengthen signal for specific document collections.
- **UI cues**: Responses include `metadata.intent` and `metadata.section_label` so clients can highlight boosted sections.

## 6. Evaluation Guide

The evaluation framework measures Recall@K, nDCG@K, MRR, and latency using curated gold sets.

- **Test sets**: Packaged under `eval/test_sets/`. Use `TestSetManager.load("test_set_v1")` to load the default 50-query set.
- **Running evaluations**: `python tests/performance/run_retrieval_benchmarks.py --rerank` computes metrics and emits JSON summary.
- **Human-readable reports**: Append `--markdown-output report.md` to generate a Markdown table of metrics for sharing with stakeholders.
- **CI integration**: The script exits with non-zero if HTTP requests fail. Combine with `openspec validate` and add gating rules around the JSON metrics for automated regression checks.
- **Acceptance criteria**: Reranking must deliver ≥+5% nDCG@10 uplift versus hybrid-only before enabling globally.

## 7. API Usage Guide

### REST

```bash
curl -X POST "$BASE_URL/v1/retrieve" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "tenant_id": "oncology",
        "query": "pembrolizumab adverse events",
        "top_k": 10,
        "rerank": true,
        "rerank_model": "bge-reranker-base",
        "query_intent": "tabular"
      }'
```

### GraphQL

```graphql
query TabularAEs {
  retrieve(input: {
    query: "pembrolizumab adverse events",
    topK: 10,
    rerank: true,
    queryIntent: TABULAR
  }) {
    documents { id title metadata }
    rerankMetrics { model { key version } applied }
  }
}
```

### gRPC

Use the `RetrievalService` proto (`proto/medicalkg/retrieval.proto`). The `RetrieveRequest` mirrors REST fields including `rerank`, `query_intent`, and `table_only`.

## 8. Query Optimisation Guide

- **Be explicit about scope**: Supply filters such as `{"dataset": "clinical-trials"}` to focus results.
- **Provide clinical hints**: Add `query_intent` or `table_only` when you know the desired format.
- **Control candidate depth**: Use `rerank_top_k` (REST) or `RetrieveInput.rerankTopK` (GraphQL) when trading latency for quality.
- **Specify reranker model**: `rerank_model=ms-marco-minilm-l12-v2` delivers sub-150 ms reranking for interactive workflows.
- **Batch evaluations**: For analytics workloads run the evaluation harness with different staging overlays (e.g. disable SPLADE via ConfigMap) to compare configurations offline.

## 9. Troubleshooting Guide

| Symptom | Likely Cause | Resolution |
| --- | --- | --- |
| Empty results with `rerank=true` | Hybrid stage returned zero candidates | Retry with `rerank=false` or increase `top_k`/`rerank_top_k`.
| `gpu_unavailable` warning | Reranker requires GPU | Ensure staging/production deployments schedule on GPU nodes or switch to CPU-capable reranker.
| Cache hit rate <40% | Fresh index or long-tail queries | Warm the cache via `/v1/retrieve` batch calls or lower `pipeline.rerank_candidates`.
| Table queries return narrative text | Intent detection confidence below threshold | Add `query_intent=tabular` or expand query with table-specific keywords.
| Clinical boost overweights irrelevant sections | Ambiguous intent | Supply `query_intent` or reduce boost multipliers in tenant configuration.

## 10. FAQ

- **Does hybrid retrieval support tenant isolation?** Yes. All requests are scoped to the caller’s tenant and filtered via metadata before fusion.
- **How many candidates are reranked?** Default is 100 retrieved, 10 returned. Override via `rerank_candidates`/`return_top_k` in the reranking settings.
- **Can I disable SPLADE or dense search?** Yes. Set `components=["bm25"]` per request or toggle `enable_splade`/`enable_dense` in the component config.
- **How do I monitor quality?** Use the Grafana “Retrieval Performance” dashboard and schedule the evaluation CronJob (`retrieval-evaluation`) to post metrics daily.
- **Is the reranking service stateless?** Yes. Models load from the on-disk cache (`model_cache/rerankers`) and can be pre-warmed on deployment.
