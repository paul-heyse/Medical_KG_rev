# Migrating from Legacy Embeddings to the vLLM/Pyserini Stack

This guide outlines the steps required to migrate consumers from the
retired SentenceTransformers/SPLADE implementation to the new
vLLM + Pyserini architecture delivered by the
`add-embeddings-representation` OpenSpec change.

## 1. Namespace Selection

1. Identify the target namespace from `config/embedding/namespaces/`.
2. Update API clients to supply the namespace explicitly:
   - REST: `POST /v1/embed { "namespace": "single_vector.qwen3.4096.v1" }`
   - GraphQL: `embed(input: { namespace: "single_vector.qwen3.4096.v1" })`
   - gRPC: `EmbedRequest.namespace`
3. Remove any legacy model identifiers hard-coded in clients.

## 2. Token Budget Enforcement

The new tokenizer cache enforces per-model token budgets. Before sending
requests, either:

- Call the `TokenizerCache.ensure_within_limit` helper, or
- Reuse the `/v1/embed` error response (`400 token_limit_exceeded`) to
  trim documents client-side.

Exact token counting catches 100% of overflows that the legacy
approximation missed.

## 3. Storage Expectations

Dense vectors are persisted to FAISS (primary) and surfaced through the
vector store service. Sparse vectors are written to OpenSearch using the
`rank_features` field declared in `build_rank_features_mapping`. Existing
callers should stop writing directly to bespoke stores.

## 4. Operational Readiness

1. Build the vLLM container: `docker-compose build vllm-qwen3`
2. Provision GPUs and verify `python -m scripts.embedding.verify_environment`
   reports `"gpu": {"available": true}`.
3. Pre-download models with
   `python -m scripts.embedding.download_models --format text`.
4. Update observability dashboards to track the new `/v1/embeddings`
   endpoint and GPU utilization metrics.

## 5. Deprecation Timeline

- **Day 0**: Deploy namespace-aware clients and new embedding worker.
- **Day 7**: Remove legacy SentenceTransformers and manual batching code.
- **Day 14**: Delete unused model artifacts from object storage.
- **Day 30**: Archive the `add-embeddings-representation` change.

For additional context consult `docs/guides/embedding_catalog.md` and the
OpenSpec design notes.
