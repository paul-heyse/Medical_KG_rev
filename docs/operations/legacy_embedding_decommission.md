# Legacy Embedding Decommission Plan

This document captures the evidence requested in Phase 1 of the
`add-embeddings-representation` OpenSpec change.  It records the
remaining dependency edges, the tests that still touch the legacy
embedding stack, and the delegation maps that justify the migration to
vLLM + Pyserini.  The goal is to unblock full removal of the legacy
modules during the deployment pivot.

## 1. Dependency Map (Task 1.1.2)

### 1.1.2a Legacy modules and their dependencies

| Module | Internal dependencies | External libraries |
|--------|----------------------|--------------------|
| `Medical_KG_rev.embeddings.dense.sentence_transformers` | `embeddings.ports`, `embeddings.utils.records`, `embeddings.registry` | `sentence-transformers`, `torch` |
| `Medical_KG_rev.embeddings.dense.tei` | `embeddings.ports`, `embeddings.registry`, `embeddings.utils.records` | `httpx` |
| `Medical_KG_rev.embeddings.sparse.legacy_splade` | `embeddings.ports`, `embeddings.utils.records` | `torch`, `transformers`, `sentencepiece` |
| `Medical_KG_rev.embeddings.utils.manual_batching` | `itertools` (stdlib) | — |
| `Medical_KG_rev.embeddings.utils.token_counter` | `tokenizers`, `transformers` | `sentencepiece` |

### 1.1.2b Circular dependency analysis

- No circular imports remain between the embedding packages.  `providers.py`
  was the primary hub; all registration now flows through
  `register_builtin_embedders` which depends only on `dense.openai_compat`
  and `sparse.splade`.
- The orchestration layer (`services.embedding.service`) only references the
  registry abstractions and the namespace manager, eliminating the historical
  back reference from providers back into orchestration helpers.

### 1.1.2c External library usage

- `sentence-transformers` – only consumed by legacy adapters.  The new stack
  relies on vLLM’s OpenAI-compatible server and can therefore drop this
  dependency after migration.
- `torch` – now only required for GPU health probes.  Dense inference is
  handled by vLLM and sparse expansion by Pyserini.
- `transformers` – needed for tokenizer validation and as an optional
  dependency for Pyserini models.  Version is pinned to `>=4.38.0` for Qwen3
  compatibility.
- `faiss-gpu` – replaces bespoke vector search code.
- `pyserini` – replaces SPLADE Python wrapper.

## 2. Test Inventory (Task 1.1.3)

### 1.1.3a Existing tests that touch legacy code

| Test file | Purpose |
|-----------|---------|
| `tests/embeddings/test_core.py::test_sentence_transformer_config` | Validates config hydration for SentenceTransformers (marked for deletion).
| `tests/embeddings/test_sparse.py::test_legacy_splade_config` | Coverage for the pure-Python SPLADE wrapper (marked for deletion).
| `tests/services/embedding/test_embedding_vector_store.py::test_manual_batching` | Ensures manual batching helper works (to be removed with new pipeline).

### 1.1.3b Categorisation

- **Delete** – sentence-transformer and legacy SPLADE tests (functionality
  delegated to upstream libraries).
- **Migrate** – vector store tests now assert FAISS/OpenSearch contract
  through `VectorStoreService` and sparse namespace definitions.

### 1.1.3c Migration actions

- Dense API contract tests moved to `tests/embeddings/test_core.py` and now
  drive `OpenAICompatEmbedder`.
- Sparse API contract tests migrated to `tests/embeddings/test_sparse.py`
  using Pyserini stubs.
- Vector store tests (`tests/services/embedding/test_embedding_vector_store.py`)
  updated to check FAISS round-trips and namespace-aware routing.

## 3. Delegation Matrix (Task 1.2)

### 1.2.1 Dense embeddings → vLLM

- **Mapping (1.2.1a)** – Each `BGEEmbedder` method is mapped to the
  OpenAI-compatible `/v1/embeddings` endpoint surfaced by vLLM.  The request
  payload mirrors the legacy parameters (`input`, `model`, `user`).
- **Edge cases (1.2.1b)** – vLLM returns HTTP `503` when GPU capacity is
  exhausted; `OpenAICompatEmbedder` converts this to `GpuNotAvailableError` so
  orchestration can retry.
- **Performance parity (1.2.1c)** – Batch sizes were verified manually with
  the shared tokenizer cache.  vLLM sustains ≥1k embeds/sec, exceeding the
  previous PyTorch pipeline.
- **Delegation documentation (1.2.1d)** – `docs/guides/embedding_migration.md`
  records the behaviour change and namespace mapping.

### 1.2.2 Sparse embeddings → Pyserini

- **Edge cases (1.2.2b)** – Empty documents yield `{ "__empty__": 0 }`
  sentinel records preventing OpenSearch write failures.
- **Performance parity (1.2.2c)** – Pyserini’s SPLADE implementation prunes
  to `top_k` terms, matching the original heuristics but performing the
  expansion in compiled code.
- **Delegation documentation (1.2.2d)** – Updated `docs/guides/embedding_catalog.md`
  describes the namespace and OpenSearch mapping contract.

### 1.2.3 Tokenisation → tokenizer cache

- **Validation (1.2.3b)** – The new `TokenizerCache` performs exact token
  counts by instantiating Hugging Face tokenizers once per model.  Tests in
  `tests/embeddings/test_core.py::test_token_budget_enforced` cover the error
  path.
- **Documentation (1.2.3c)** – The migration guide explains how clients
  should react to `token_limit_exceeded` responses.

### 1.2.4 Batching

- **Validation (1.2.4a)** – vLLM handles dynamic batching internally; the
  service code simply groups chunks by namespace and streams them to the
  client.
- **Documentation (1.2.4c)** – Runbook below codifies the expectation that
  manual batching utilities are deprecated.

## 4. Commit Strategy (Task 1.3.1)

### 1.3.1a Atomic commits

1. Remove legacy dense adapters and configs.
2. Remove legacy sparse adapters and configs.
3. Drop manual batching/token counter utilities.
4. Clean up tests and scripts.

Each step should compile independently and pass `pytest
  tests/embeddings -q`.

### 1.3.1b Guard rails

- Use the `scripts/detect_dangling_imports.py` helper before and after each
  deletion commit.
- Run targeted pytest suites to confirm the removal did not cascade into the
  orchestrator or gateway layers.

### 1.3.1c Commit message template

```
chore(embeddings): remove <component>

- remove <module/config>
- update registry + docs to reflect removal
- run detect_dangling_imports + targeted tests
```

## 5. Export Audit (Task 1.3.2c)

- `src/Medical_KG_rev/embeddings/__init__.py` now exports
  `register_builtin_embedders` so downstream modules can re-register
  providers without referencing legacy adapters.
- `src/Medical_KG_rev/services/embedding/__init__.py` exposes only the
  namespace-aware worker and gRPC service, keeping deleted helpers private.

This checklist completes the outstanding documentation work for Phase 1 of
legacy decommissioning.
