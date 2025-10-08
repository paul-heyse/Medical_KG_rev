# Embedding Adapter Catalog

The hard cutover to the standardized embedding stack replaces bespoke
implementations with a small set of library-backed adapters. The table
below lists the production namespaces committed with this change.

| Adapter | Kind | Namespace | Provider | Notes |
| ------- | ---- | --------- | -------- | ----- |
| `OpenAICompatEmbedder` | `single_vector` | `single_vector.qwen3.4096.v1` | vLLM (OpenAI compatible) | Delegates to the GPU-only vLLM server hosting Qwen3-Embedding-8B and returns normalized 4096-d vectors. |
| `PyseriniSparseEmbedder` | `sparse` | `sparse.splade_v3.400.v1` | Pyserini SPLADE | Generates learned sparse term weights for OpenSearch `rank_features` storage with safe empty-text handling. |
| `ColbertIndexerEmbedder` | `multi_vector` | `multi_vector.colbert_v2.128.v1` | ColBERT | Late-interaction embeddings backed by FAISS shards for reranking and multi-vector retrieval. |

Legacy adapters (SentenceTransformers, TEI, LangChain, etc.) were
removed as part of the cutover and can be reinstated only by creating a
new namespace YAML file with explicit ownership approval.

## Configuration Examples

Namespaces are declared via YAML and hydrated by the runtime registry.
The snippet below mirrors the defaults committed to
`config/embedding/namespaces/`.

```yaml
embeddings:
  active_namespaces:
    - single_vector.qwen3.4096.v1
    - sparse.splade_v3.400.v1
    - multi_vector.colbert_v2.128.v1
  providers:
    - name: qwen3
      provider: vllm
      kind: single_vector
      namespace: single_vector.qwen3.4096.v1
      model_id: Qwen/Qwen2.5-Coder-1.5B
      dim: 4096
      parameters:
        endpoint: http://vllm-embedding:8001/v1
        timeout: 60
        normalize: true
    - name: splade-v3
      provider: pyserini
      kind: sparse
      namespace: sparse.splade_v3.400.v1
      model_id: naver/splade-v3
      parameters:
        top_k: 400
    - name: colbert
      provider: colbert
      kind: multi_vector
      namespace: multi_vector.colbert_v2.128.v1
      model_id: colbert-ir/colbertv2.0
      parameters:
        shard_capacity: 100000
```

## Deployment Notes

### vLLM Embedding Endpoint

Build and run the dedicated embedding container with Docker Compose:

```bash
docker-compose up -d vllm-embedding
```

The service exposes an OpenAI-compatible `/v1/embeddings` endpoint on
port `8001` and fails fast when GPU resources are unavailable.

### Model Materialization

Use the helper script introduced with this change to fetch the required
models ahead of time:

```bash
python -m scripts.embedding.download_models --format json
```

The script downloads Qwen3-Embedding-8B into
`models/qwen3-embedding-8b/` and materializes the SPLADE encoder cache in
`models/splade-v3/`. Pair it with
`python -m scripts.embedding.verify_environment` to confirm GPU and
dependency readiness before starting the workers.
