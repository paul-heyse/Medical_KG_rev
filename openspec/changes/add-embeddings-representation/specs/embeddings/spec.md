# Embeddings Capability: Spec Delta

## ADDED Requirements

### Requirement: vLLM Dense Embedding Service (GPU-Only)

The system SHALL provide a vLLM-based dense embedding service serving Qwen3-Embedding-8B via OpenAI-compatible `/v1/embeddings` API with GPU-only enforcement and fail-fast semantics.

**Rationale**: vLLM achieves 5x throughput (1000+ emb/sec) vs sentence-transformers (200 emb/sec), enforces GPU-only policy, and provides OpenAI-compatible API that simplifies client code.

#### Scenario: vLLM embedding request via OpenAI-compatible API

- **GIVEN** a list of 64 chunk texts
- **WHEN** the embedding service calls `POST /v1/embeddings` with `input=[texts]` and `model="Qwen/Qwen2.5-Coder-1.5B"`
- **THEN** vLLM returns 64 embeddings (4096-D vectors) in <2 seconds
- **AND** embeddings are normalized (L2 norm = 1.0)
- **AND** response follows OpenAI format: `{"data": [{"embedding": [...]}]}`

#### Scenario: vLLM GPU fail-fast on startup

- **GIVEN** vLLM service starting on a machine without GPU
- **WHEN** vLLM attempts to initialize
- **THEN** vLLM raises `RuntimeError: No GPU available. vLLM requires CUDA.`
- **AND** the service refuses to start (no CPU fallback)
- **AND** health check returns 503 Service Unavailable

#### Scenario: vLLM health check includes GPU status

- **GIVEN** vLLM service running on GPU
- **WHEN** a client calls `GET /health`
- **THEN** the response includes `{"status": "healthy", "gpu": "available"}`
- **AND** if GPU becomes unavailable, health check returns 503 with `{"status": "unhealthy", "gpu": "unavailable"}`

#### Scenario: vLLM batch processing efficiency

- **GIVEN** a batch of 128 texts to embed
- **WHEN** the embedding service sends the batch to vLLM
- **THEN** vLLM processes the batch in 2 requests (batch_size=64)
- **AND** throughput is ≥1000 embeddings/sec
- **AND** GPU utilization is 60-80% (efficient use without saturation)

---

### Requirement: Pyserini SPLADE Sparse Embedding

The system SHALL provide Pyserini-based SPLADE-v3 sparse embedding with document-side expansion as default, producing term-weight dictionaries for OpenSearch `rank_features` storage.

**Rationale**: Pyserini handles SPLADE complexity (term weighting, top-K pruning) and document-side expansion provides 80% of recall gains with simpler ops than query-side expansion.

#### Scenario: Pyserini document-side expansion with top-K pruning

- **GIVEN** a chunk text: "Significant reduction in HbA1c levels after treatment"
- **WHEN** Pyserini SPLADE wrapper expands the document with `top_k=400`
- **THEN** the output is a dict of term-weight pairs: `{"hba1c": 2.8, "reduction": 2.1, "significant": 1.9, ...}` (up to 400 terms)
- **AND** terms are sorted by weight descending
- **AND** all weights are positive floats

#### Scenario: Pyserini query-side expansion (opt-in)

- **GIVEN** a query: "diabetes treatment" and `expand_query_side=true` flag
- **WHEN** the retrieval service expands the query with Pyserini SPLADE
- **THEN** the query is expanded to `{"diabetes": 3.2, "treatment": 2.5, "glucose": 1.8, ...}` (top_k=100)
- **AND** expanded terms include semantic neighbors (e.g., "glucose" even though not in original query)
- **AND** query-side expansion adds ~50ms latency

#### Scenario: SPLADE empty text handling

- **GIVEN** an empty chunk text ""
- **WHEN** Pyserini SPLADE wrapper attempts to expand
- **THEN** the output is an empty dict `{}`
- **AND** no error is raised (graceful handling)

---

### Requirement: Multi-Namespace Embedding Registry

The system SHALL provide a multi-namespace embedding registry supporting multiple embedding families (single_vector, sparse, multi_vector) with YAML-based configuration and runtime discovery.

**Rationale**: Enables A/B testing new models, gradual migration, and explicit model versioning without refactoring.

#### Scenario: Register namespace from YAML config

- **GIVEN** a YAML config file `config/embedding/namespaces/single_vector.qwen3.4096.v1.yaml`
- **WHEN** the embedding service starts up
- **THEN** the registry loads the namespace with key `"single_vector.qwen3.4096.v1"`
- **AND** the config includes: model_id, provider, dim, endpoint, parameters
- **AND** the namespace is available for embedding requests

#### Scenario: Get namespace config for embedding

- **GIVEN** a namespace `"single_vector.qwen3.4096.v1"` registered
- **WHEN** the embedding service calls `registry.get("single_vector.qwen3.4096.v1")`
- **THEN** the registry returns a `NamespaceConfig` object with provider="vllm", dim=4096, endpoint="<http://vllm-qwen3:8001>"
- **AND** the service routes the embedding request to vLLM

#### Scenario: Unknown namespace error

- **GIVEN** a namespace `"nonexistent-namespace"` not registered
- **WHEN** the embedding service calls `registry.get("nonexistent-namespace")`
- **THEN** the registry raises `ValueError("Namespace 'nonexistent-namespace' not found. Available: single_vector.qwen3.4096.v1, sparse.splade_v3.400.v1")`
- **AND** the error message lists available namespaces

#### Scenario: List namespaces by kind

- **GIVEN** 3 namespaces registered: 2 single_vector, 1 sparse
- **WHEN** the service calls `registry.list_by_kind(EmbeddingKind.SINGLE_VECTOR)`
- **THEN** the registry returns `["single_vector.qwen3.4096.v1", "single_vector.bge.384.v1"]`
- **AND** sparse namespace is excluded

---

### Requirement: Model-Aligned Tokenizers (No Approximation)

The system SHALL use exact tokenizers aligned with embedding models (Qwen3 for dense, SPLADE for sparse) to prevent token overflow failures, replacing approximate token counting.

**Rationale**: Exact tokenizers catch 100% of overflows (vs 85% with approximation), ensuring fail-fast before GPU compute.

#### Scenario: Exact token counting with Qwen3 tokenizer

- **GIVEN** a chunk text with 8500 tokens (exceeds Qwen3 limit of 8192)
- **WHEN** the embedding service counts tokens using `transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")`
- **THEN** the service raises `TokenLimitExceededError("Text has 8500 tokens, max 8192")`
- **AND** the embedding request is rejected before calling vLLM
- **AND** the job ledger is updated with `status="embed_failed", error="token_limit_exceeded"`

#### Scenario: Tokenizer caching for performance

- **GIVEN** the embedding service starts up
- **WHEN** the service initializes
- **THEN** the Qwen3 tokenizer is loaded once and cached in memory
- **AND** subsequent token counting operations reuse the cached tokenizer
- **AND** tokenizer loading overhead (~2s) is amortized across all requests

#### Scenario: Approximate vs exact token count comparison

- **GIVEN** a text with special characters and long biomedical terms
- **WHEN** approximate counting (`len(text) / 4`) estimates 2000 tokens
- **AND** exact tokenizer counts 2450 tokens
- **THEN** the exact count catches the 450-token overflow (if limit is 2048)
- **AND** the approximate method would miss the overflow (allowing a failure at embedding stage)

---

### Requirement: GPU Fail-Fast Enforcement

The system SHALL enforce GPU availability at embedding service startup and during health checks, failing immediately if GPU unavailable with no silent CPU fallbacks.

**Rationale**: Aligns with GPU-only policy, prevents 40-60% quality degradation from CPU fallbacks, provides clear failure semantics.

#### Scenario: GPU availability check on service startup

- **GIVEN** the embedding service starting on a machine with GPU
- **WHEN** the service initializes
- **THEN** the service checks `torch.cuda.is_available()`
- **AND** if GPU is available, service starts successfully
- **AND** if GPU is unavailable, service raises `GpuNotAvailableError("Embedding service requires GPU")`
- **AND** the service refuses to start (no CPU fallback)

#### Scenario: Health endpoint includes GPU check

- **GIVEN** the embedding service running
- **WHEN** a client calls `GET /health`
- **THEN** the service checks GPU availability via `torch.cuda.is_available()`
- **AND** if GPU available, returns 200 OK with `{"status": "healthy", "gpu": "available"}`
- **AND** if GPU unavailable, returns 503 Service Unavailable with `{"status": "unhealthy", "gpu": "unavailable"}`

#### Scenario: Embedding request fails fast when GPU unavailable

- **GIVEN** vLLM service reports GPU unavailable (503)
- **WHEN** the embedding service attempts to embed texts
- **THEN** the service raises `GpuNotAvailableError("vLLM service reports GPU unavailable")`
- **AND** the job ledger is updated with `status="embed_failed", error="gpu_unavailable"`
- **AND** no CPU fallback is attempted
- **AND** the job is marked for manual intervention (no automatic retry)

---

## MODIFIED Requirements

### Requirement: Embedding Service API (Modified)

The embedding service API SHALL accept a `namespace` parameter to specify which embedding model/provider to use, replacing the previous single-model API.

**Previous Behavior**: `embed(texts: list[str]) -> list[np.ndarray]` used a single embedding model with no selection.

**New Behavior**: `embed(texts: list[str], namespace: str) -> list[Embedding]` SHALL route to the appropriate provider (vLLM, Pyserini) based on namespace configuration.

#### Scenario: Embedding with namespace parameter (dense)

- **GIVEN** a list of texts and namespace `"single_vector.qwen3.4096.v1"`
- **WHEN** the embedding service calls `embed(texts, namespace="single_vector.qwen3.4096.v1")`
- **THEN** the service routes to vLLM provider
- **AND** returns embeddings with `namespace="single_vector.qwen3.4096.v1"` metadata
- **AND** embeddings are 4096-D vectors

#### Scenario: Embedding with namespace parameter (sparse)

- **GIVEN** a list of texts and namespace `"sparse.splade_v3.400.v1"`
- **WHEN** the embedding service calls `embed(texts, namespace="sparse.splade_v3.400.v1")`
- **THEN** the service routes to Pyserini provider
- **AND** returns sparse embeddings with term-weight dictionaries
- **AND** embeddings have `namespace="sparse.splade_v3.400.v1"` metadata

#### Scenario: Default namespace fallback

- **GIVEN** a list of texts and no namespace parameter
- **WHEN** the embedding service calls `embed(texts)`
- **THEN** the service uses default namespace from config (`MK_EMBEDDING_DEFAULT_NAMESPACE`)
- **AND** returns embeddings with default namespace metadata

---

## REMOVED Requirements

### Requirement: Direct sentence-transformers Integration (Removed)

**Removed**: The requirement for direct sentence-transformers model loading and inference is **REMOVED** in favor of vLLM serving.

**Reason**: Direct model loading lacks GPU-only enforcement, achieves only 200 emb/sec (vs vLLM 1000+), and requires bespoke batching logic.

**Migration**: All `sentence_transformers.SentenceTransformer` usage replaced with vLLM OpenAI-compatible API calls.

---

### Requirement: Custom SPLADE Implementation (Removed)

**Removed**: The requirement for custom SPLADE term weighting and pruning logic is **REMOVED** in favor of Pyserini wrapper.

**Reason**: Custom implementation duplicated Pyserini functionality, lacked term weighting optimizations, and had maintenance burden.

**Migration**: All custom SPLADE code replaced with `pyserini.encode.SpladeQueryEncoder`.

---

### Requirement: Approximate Token Counting (Removed)

**Removed**: The requirement for approximate token counting (`len(text) / 4`) is **REMOVED** in favor of exact model-aligned tokenizers.

**Reason**: Approximate counting missed 15% of token overflows, causing embedding failures at GPU stage.

**Migration**: All approximate token counting replaced with `transformers.AutoTokenizer` aligned with Qwen3.
