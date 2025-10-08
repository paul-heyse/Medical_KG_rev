# Retrieval Spec Delta

## ADDED Requirements

### Requirement: Haystack-Based Chunking Stage

The retrieval system MUST provide a `ChunkStage` implementation using Haystack 2.0+ components. The `HaystackChunker` class SHALL:

- Implement the `ChunkStage` Protocol: `execute(ctx: StageContext, document: Document) -> list[Chunk]`
- Wrap `haystack.components.preprocessors.DocumentSplitter` for semantic chunking
- Convert IR `Document` to Haystack `Document` format before splitting
- Convert Haystack split documents back to IR `Chunk[]` with provenance metadata
- Support configurable split strategies (sentence, word, passage, recursive)
- Preserve layout information (bounding boxes, reading order) from MinerU-processed PDFs
- Apply coherence scoring to chunk boundaries (reuse existing `SemanticSplitter` logic)

#### Scenario: Non-PDF document chunked semantically

- **GIVEN** an IR Document from ClinicalTrials.gov with 3 sections (background, methods, results)
- **WHEN** `HaystackChunker.execute(ctx, document)` is called
- **THEN** the document is converted to 3 Haystack Documents (one per section)
- **AND** the `DocumentSplitter` splits them into 12 chunks based on sentence boundaries
- **AND** each chunk is converted back to IR `Chunk` with fields:
  - `chunk_id`: f"{doc_id}:chunk:{index}"
  - `text`: Chunk content
  - `metadata`: {"tenant_id", "source_doc_id", "section_id", "coherence_score"}
  - `provenance`: {"stage": "chunk", "chunker": "haystack.DocumentSplitter", "strategy": "sentence"}

#### Scenario: PDF document chunked with layout awareness

- **GIVEN** an IR Document from PMC with layout information (bounding boxes, reading order) from MinerU
- **WHEN** `HaystackChunker.execute(ctx, document)` is called
- **THEN** the chunker respects layout boundaries (figures, tables, columns)
- **AND** each chunk retains `layout_bbox` field for spatial grounding
- **AND** chunks spanning figures include both caption and OCR'd text
- **AND** multi-column text is chunked in reading order (left column first, then right)

---

### Requirement: Haystack-Based Embedding Stage

The retrieval system MUST provide an `EmbedStage` implementation using Haystack 2.0+ for GPU-accelerated embeddings. The `HaystackEmbedder` class SHALL:

- Implement the `EmbedStage` Protocol: `execute(ctx: StageContext, chunks: list[Chunk]) -> EmbeddingBatch`
- Wrap `haystack.components.embedders.OpenAIDocumentEmbedder` pointing to local vLLM endpoint
- Use Qwen-3 model for dense embeddings (dimension 512)
- Batch process chunks (batch_size=32) for GPU efficiency
- Fail-fast if GPU is unavailable (no CPU fallback)
- Apply resilience policy "gpu-bound" (1 retry, circuit breaker on GPU OOM)
- Emit GPU utilization metrics to Prometheus

#### Scenario: Chunks embedded with vLLM Qwen endpoint

- **GIVEN** 64 chunks from a PMC full-text document
- **WHEN** `HaystackEmbedder.execute(ctx, chunks)` is called
- **THEN** the embedder sends 2 batches (32 chunks each) to `http://localhost:8000/v1/embeddings`
- **AND** each request uses OpenAI-compatible format: `{"input": ["text1", "text2", ...], "model": "Qwen/Qwen-3-0.5B"}`
- **AND** the response vectors (512 dimensions) are wrapped in `EmbeddingBatch`
- **AND** Prometheus metric `gpu_embedding_duration_seconds` is recorded

#### Scenario: GPU unavailable fails embed stage immediately

- **GIVEN** a vLLM server that returns HTTP 503 with error "CUDA device not available"
- **WHEN** `HaystackEmbedder.execute(ctx, chunks)` is called
- **THEN** the stage fails immediately with `GpuNotAvailableError`
- **AND** no retries are attempted (gpu-bound policy: max_attempts=1)
- **AND** the job is marked `embed_failed_no_gpu` in Job Ledger
- **AND** a CloudEvent `stage.failed` with `data.reason="no_gpu"` is emitted

---

### Requirement: Haystack-Based SPLADE Expansion Stage

The retrieval system MUST provide a custom Haystack component for SPLADE sparse vector generation. The `HaystackSparseExpander` class SHALL:

- Subclass `haystack.core.component.Component` with `@component` decorator
- Implement GPU-backed SPLADE expansion using local SPLADE model endpoint
- Generate sparse term weights (dimension 400, top-k 400 terms per chunk)
- Integrate into embedding pipeline alongside dense vectors
- Fail-fast if GPU is unavailable
- Apply resilience policy "gpu-bound"

#### Scenario: SPLADE expansion produces sparse vectors

- **GIVEN** 32 chunks to be indexed
- **WHEN** `HaystackSparseExpander.run(documents=chunks)` is called
- **THEN** each chunk is expanded into a sparse vector with 400 dimensions
- **AND** each vector contains ~200 non-zero term weights (e.g., {"cancer": 0.87, "treatment": 0.65, ...})
- **AND** the vectors are formatted for OpenSearch sparse indexing
- **AND** GPU memory usage is tracked via Prometheus

#### Scenario: SPLADE GPU OOM triggers circuit breaker

- **GIVEN** a SPLADE service that fails 5 times with "CUDA out of memory"
- **WHEN** the 6th chunk batch attempts SPLADE expansion
- **THEN** the circuit breaker opens (gpu-bound policy: failure_threshold=5)
- **AND** subsequent calls fail immediately without hitting the SPLADE endpoint
- **AND** after 60 seconds (reset_timeout), one test request is allowed through
- **AND** if it succeeds, the circuit closes and processing resumes

---

### Requirement: Haystack-Based Dual Index Writer

The retrieval system MUST provide an `IndexStage` implementation that writes to both OpenSearch and FAISS. The `HaystackIndexWriter` class SHALL:

- Implement the `IndexStage` Protocol: `execute(ctx: StageContext, batch: EmbeddingBatch) -> IndexReceipt`
- Wrap `haystack.components.writers.OpenSearchDocumentWriter` for BM25 + SPLADE indexing
- Wrap custom FAISS writer for dense vector indexing
- Perform dual writes transactionally (both succeed or both fail)
- Use tenant-aware index naming (e.g., `tenant-123-documents` for OpenSearch)
- Store FAISS index in MinIO with versioning (e.g., `tenants/tenant-123/faiss/index-v1.faiss`)
- Track indexing latency and throughput via Prometheus

#### Scenario: Chunks indexed to OpenSearch and FAISS

- **GIVEN** an EmbeddingBatch with 32 chunks, each having dense (512-d) and sparse (400-d) vectors
- **WHEN** `HaystackIndexWriter.execute(ctx, batch)` is called
- **THEN** OpenSearch receives 32 documents with:
  - `text`: Chunk content
  - `dense_vector`: 512-d embedding
  - `sparse_vector`: 400-d SPLADE weights
  - `metadata`: {"tenant_id", "chunk_id", "source_doc_id", "indexed_at"}
- **AND** FAISS index is updated with 32 dense vectors
- **AND** both operations complete successfully
- **AND** an `IndexReceipt` is returned with `chunks_indexed=32`, `opensearch_ok=true`, `faiss_ok=true`

#### Scenario: OpenSearch write fails, FAISS write rolled back

- **GIVEN** an OpenSearch cluster that returns HTTP 503 (service unavailable)
- **WHEN** `HaystackIndexWriter.execute(ctx, batch)` is called
- **THEN** the OpenSearch write fails after retries (default policy: 3 attempts)
- **AND** the FAISS write is NOT performed (transactional semantics)
- **AND** the stage emits `stage.failed` CloudEvent with error "opensearch_unavailable"
- **AND** the job is marked `index_failed` in Job Ledger

---

### Requirement: Haystack-Based Hybrid Retrieval

The retrieval system MUST provide a retrieval pipeline using Haystack components for BM25, dense, and fusion ranking. The `HaystackRetriever` class SHALL:

- Wrap `haystack.components.retrievers.OpenSearchBM25Retriever` for lexical search
- Wrap custom FAISS retriever for dense semantic search
- Integrate SPLADE sparse retrieval via OpenSearch `script_score` query
- Combine results using Reciprocal Rank Fusion (RRF) with k=60
- Apply tenant-aware filtering to all retrievers
- Support OData query parameters for filtering, sorting, pagination
- Maintain P95 latency < 500ms for retrieval queries

#### Scenario: Hybrid retrieval combines BM25, SPLADE, dense

- **GIVEN** a search query "cancer immunotherapy clinical trials"
- **WHEN** `HaystackRetriever.retrieve(query, tenant_id="tenant-123", top_k=10)` is called
- **THEN** three retrievers execute in parallel:
  - BM25 retriever: Searches OpenSearch with term matching, returns 100 results
  - SPLADE retriever: Expands query with learned terms, searches OpenSearch sparse vectors, returns 100 results
  - Dense retriever: Embeds query with Qwen, searches FAISS, returns 100 results
- **AND** RRF fusion merges the three result lists with reciprocal rank scoring
- **AND** the top 10 fused results are returned with provenance (which retrievers ranked each result)

#### Scenario: Tenant isolation enforced across retrievers

- **GIVEN** OpenSearch indices contain documents for multiple tenants
- **WHEN** a retrieval request for tenant-123 is executed
- **THEN** all three retrievers include `{"term": {"tenant_id": "tenant-123"}}` filter
- **AND** no results from other tenants are returned
- **AND** FAISS search uses tenant-specific index partition

---

## MODIFIED Requirements

### Requirement: Chunking Service API

The chunking service MUST support both legacy direct invocation and stage-based invocation. The `ChunkingService` SHALL:

- Maintain existing `chunk(document_id, text, options)` method for backward compatibility
- Be wrapped by `HaystackChunker` for Dagster ops when `MK_USE_DAGSTER=true`
- Support profile-based chunking (e.g., `pmc` profile for full-text articles with figures)
- Preserve coherence scoring and multi-granularity chunking features

#### Scenario: Legacy chunking service invoked directly

- **GIVEN** feature flag `MK_USE_DAGSTER=false`
- **WHEN** `chunking_service.chunk(doc_id, text, options)` is called
- **THEN** chunks are produced using existing logic (SemanticSplitter, SlidingWindow, etc.)
- **AND** no Haystack components are involved
- **AND** no CloudEvents are emitted

#### Scenario: Dagster invokes chunking via HaystackChunker

- **GIVEN** feature flag `MK_USE_DAGSTER=true`
- **WHEN** the chunk_op executes in a Dagster job
- **THEN** `HaystackChunker.execute()` wraps the existing chunking logic
- **AND** Haystack DocumentSplitter is used for splitting
- **AND** CloudEvents are emitted for stage lifecycle
- **AND** resilience policies are applied

---

## REMOVED Requirements

None (all existing retrieval requirements preserved, augmented with Haystack wrappers)
