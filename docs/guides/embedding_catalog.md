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

## Namespace Access Control

Every namespace entry now carries explicit access control metadata. The
consolidated `config/embedding/namespaces.yaml` file documents
`allowed_tenants` and `allowed_scopes` for each namespace. Public vectors
such as `single_vector.qwen3.4096.v1` expose `allowed_tenants: ["all"]`,
whereas private namespaces enumerate the specific tenant identifiers.

Complementary configuration files:

- `config/embedding/vllm.yaml` – dense embeddings service parameters validated
  via `Medical_KG_rev.config.load_vllm_config`.
- `config/embedding/pyserini.yaml` – SPLADE/Pyserini settings consumed by
  `load_pyserini_config` to keep OpenSearch options in sync.

Clients MUST call the discovery endpoint before embedding:

```http
GET /v1/namespaces
Authorization: Bearer <token>
```

The response lists `NamespaceInfo` objects containing provider, kind,
dimension, token budget, and ACL metadata. To proactively check whether
texts respect the namespace token budget, call the validation endpoint:

```http
POST /v1/namespaces/{namespace}/validate
Content-Type: application/json
{
  "tenant_id": "tenant-123",
  "texts": ["...", "..."]
}
```

The API returns per-text token counts and `exceeds_budget` flags. Both
endpoints require the `embed:read` scope; embedding operations require
`embed:write`.

## Configuration Examples

Namespaces are declared via YAML and hydrated by the runtime registry.
The snippet below mirrors the defaults committed to
`config/embedding/namespaces/`.

```yaml
namespaces:
  single_vector.qwen3.4096.v1:
    provider: vllm
    kind: single_vector
    model_id: Qwen/Qwen2.5-Embedding-8B-Instruct
    dim: 4096
    max_tokens: 8192
    tokenizer: Qwen/Qwen2.5-Coder-1.5B
    allowed_scopes: ["embed:read", "embed:write"]
    allowed_tenants: ["all"]
  sparse.splade_v3.400.v1:
    provider: pyserini
    kind: sparse
    model_id: naver/splade-v3
    dim: 400
    max_tokens: 512
    tokenizer: naver/splade-v3
    allowed_scopes: ["embed:read", "embed:write"]
    allowed_tenants: ["all"]
  multi_vector.colbert_v2.128.v1:
    provider: colbert
    kind: multi_vector
    model_id: colbert-v2
    dim: 128
    max_tokens: 512
    tokenizer: colbert-ir/colbertv2.0
    allowed_scopes: ["embed:read", "embed:write"]
    allowed_tenants: ["tenant-123", "tenant-456"]
```

## Deployment Notes

### vLLM Embedding Endpoint

Build and run the dedicated embedding container with Docker Compose:

```bash
docker-compose up -d vllm-qwen3
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
