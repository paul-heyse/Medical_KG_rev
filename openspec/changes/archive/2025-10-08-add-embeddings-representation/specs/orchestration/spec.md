# Orchestration Capability: Spec Delta

## MODIFIED Requirements

### Requirement: Embedding Stage Integration (Modified)

The embedding stage SHALL integrate vLLM client and Pyserini wrapper, supporting multi-namespace embedding with GPU fail-fast enforcement.

**Previous Behavior**: Embedding stage used direct sentence-transformers calls with single model and unclear GPU handling.

**New Behavior**: Embedding stage SHALL route to vLLM or Pyserini based on namespace configuration, enforce GPU availability, and track failures in job ledger.

#### Scenario: Dense embedding stage via vLLM

- **GIVEN** a batch of 150 chunks after chunking stage
- **WHEN** the orchestrator executes the embedding stage with namespace `"single_vector.qwen3.4096.v1"`
- **THEN** the stage calls vLLM client with chunk texts
- **AND** vLLM returns 150 embeddings (4096-D vectors)
- **AND** embeddings are written to FAISS index
- **AND** job ledger is updated: `status="embedding_completed", embeddings_count=150`

#### Scenario: Sparse embedding stage via Pyserini

- **GIVEN** a batch of 150 chunks after chunking stage
- **WHEN** the orchestrator executes the sparse embedding stage with namespace `"sparse.splade_v3.400.v1"`
- **THEN** the stage calls Pyserini wrapper for document-side expansion
- **AND** Pyserini returns 150 sparse embeddings (term-weight dictionaries)
- **AND** sparse embeddings are written to OpenSearch `rank_features` field
- **AND** job ledger is updated: `status="sparse_embedding_completed", embeddings_count=150`

#### Scenario: GPU fail-fast halts embedding stage

- **GIVEN** vLLM service reports GPU unavailable (503)
- **WHEN** the orchestrator executes the embedding stage
- **THEN** the stage raises `GpuNotAvailableError("vLLM service reports GPU unavailable")`
- **AND** job ledger is updated: `status="embed_failed", error="gpu_unavailable", retry_allowed=false`
- **AND** orchestrator does NOT retry (manual intervention required)
- **AND** downstream stages (indexing, extraction) are not executed

#### Scenario: Multi-namespace embedding (dense + sparse)

- **GIVEN** a batch of 150 chunks after chunking stage
- **WHEN** the orchestrator executes embedding with two namespaces: `["single_vector.qwen3.4096.v1", "sparse.splade_v3.400.v1"]`
- **THEN** dense embeddings are generated via vLLM and written to FAISS
- **AND** sparse embeddings are generated via Pyserini and written to OpenSearch
- **AND** both embedding processes run sequentially (dense first, then sparse)
- **AND** job ledger tracks both: `dense_embeddings_count=150, sparse_embeddings_count=150`

---

### Requirement: Job Ledger Embedding Tracking (Modified)

The job ledger SHALL track embedding-specific fields including namespace, GPU availability, embedding counts, and failure types.

**Previous Behavior**: Job ledger tracked high-level embedding status (embedding, embedding_completed, embedding_failed) with minimal detail.

**New Behavior**: Job ledger SHALL track detailed embedding metadata including namespace used, GPU status, embedding counts per namespace, and granular failure types.

#### Scenario: Ledger tracks embedding namespace

- **GIVEN** an embedding job using namespace `"single_vector.qwen3.4096.v1"`
- **WHEN** the orchestrator starts the embedding stage
- **THEN** job ledger is updated: `status="embedding", namespace="single_vector.qwen3.4096.v1"`
- **WHEN** embedding completes successfully
- **THEN** ledger is updated: `status="embedding_completed", embeddings_count=150, embedding_duration_seconds=2.5`

#### Scenario: Ledger tracks GPU failures

- **GIVEN** an embedding job when GPU becomes unavailable
- **WHEN** the embedding stage fails with `GpuNotAvailableError`
- **THEN** job ledger is updated:
  ```json
  {
    "status": "embed_failed",
    "error": "gpu_unavailable",
    "error_message": "vLLM service reports GPU unavailable",
    "gpu_available": false,
    "retry_allowed": false,
    "failed_at": "2025-10-07T14:30:00Z"
  }
  ```

#### Scenario: Ledger tracks multi-namespace embeddings

- **GIVEN** an embedding job with dense and sparse namespaces
- **WHEN** both embedding stages complete
- **THEN** job ledger is updated:
  ```json
  {
    "status": "embedding_completed",
    "embeddings": {
      "single_vector.qwen3.4096.v1": {
        "count": 150,
        "duration_seconds": 2.5,
        "storage": "faiss"
      },
      "sparse.splade_v3.400.v1": {
        "count": 150,
        "duration_seconds": 1.8,
        "storage": "opensearch_rank_features"
      }
    }
  }
  ```

#### Scenario: Ledger migration adds embedding fields

- **GIVEN** 1000 existing job ledger entries without embedding namespace fields
- **WHEN** the ledger migration script runs
- **THEN** all entries have default values:
  - `namespace="single_vector.bge.384.v1"` (legacy default)
  - `gpu_available=null` (unknown for past jobs)
  - `embeddings={}` (empty for jobs before multi-namespace support)

---

## REMOVED Requirements

### Requirement: Orchestration Direct Model Loading (Removed)

**Removed**: The requirement for orchestration to directly load sentence-transformers models is **REMOVED** in favor of vLLM client calls.

**Reason**: Direct model loading in orchestration tight-couples orchestration to embedding implementation, lacks GPU enforcement, and prevents experimentation with new models without orchestration changes.

**Migration**: Orchestration now calls embedding service API (vLLM/Pyserini) via network, decoupling orchestration from embedding implementation.
