# Specification: Embedding Core System

## ADDED Requirements

### Requirement: BaseEmbedder Interface

The system SHALL provide a universal `BaseEmbedder` protocol interface supporting all embedding paradigms (dense, sparse, multi-vector, neural-sparse).

#### Scenario: Embedder implements required interface

- **GIVEN** a new embedding adapter is created
- **WHEN** the adapter is registered
- **THEN** the adapter SHALL implement `embed_documents()` accepting text list
- **AND** the adapter SHALL implement `embed_queries()` for query-specific encoding
- **AND** both methods SHALL return list of `EmbeddingRecord` objects

#### Scenario: Paradigm-specific output

- **GIVEN** different embedding paradigms
- **WHEN** embeddings are generated
- **THEN** dense embedders SHALL populate `vectors` field with single vectors
- **AND** multi-vector embedders SHALL populate `vectors` with token vectors
- **AND** sparse embedders SHALL populate `terms` field with term→weight maps
- **AND** `kind` field SHALL correctly identify paradigm

### Requirement: Namespace Management

The system SHALL enforce namespace-based embedding management with automatic dimension validation.

#### Scenario: Namespace format

- **GIVEN** an embedder configuration
- **WHEN** namespace is created
- **THEN** namespace SHALL follow format `{kind}.{model}.{dim}.{version}`
- **AND** namespace SHALL uniquely identify embedding configuration
- **AND** conflicts SHALL be detected and rejected

#### Scenario: Dimension introspection and validation

- **GIVEN** embeddings are generated
- **WHEN** first embedding is created for namespace
- **THEN** system SHALL introspect actual dimension from output
- **AND** system SHALL validate dimension matches configuration
- **AND** mismatches SHALL raise DimensionMismatchError

### Requirement: Dense Bi-Encoder Adapters

The system SHALL provide production-ready dense embedding adapters for BGE, E5, GTE, SPECTER, SapBERT via Sentence-Transformers.

#### Scenario: BGE embedding generation

- **GIVEN** BGE-large-en model configuration
- **WHEN** documents are embedded
- **THEN** embedder SHALL produce 1024-D L2-normalized vectors
- **AND** embedder SHALL use mean pooling
- **AND** batch processing SHALL be GPU-accelerated when available

#### Scenario: E5 prefix enforcement

- **GIVEN** E5 model with query_prefix and passage_prefix configured
- **WHEN** queries are embedded
- **THEN** embedder SHALL prepend "query: " prefix
- **WHEN** documents are embedded
- **THEN** embedder SHALL prepend "passage: " prefix
- **AND** prefixes SHALL be enforced automatically

### Requirement: SPLADE Sparse Embedder

The system SHALL provide SPLADE document and query encoders for learned-sparse retrieval.

#### Scenario: Document-side term expansion

- **GIVEN** SPLADE-v3 model
- **WHEN** documents are encoded
- **THEN** embedder SHALL generate term→weight map with ~30k vocabulary
- **AND** top-K terms SHALL be selected (default: 400)
- **AND** weights SHALL be positive floats
- **AND** output SHALL map to OpenSearch rank_features

### Requirement: ColBERT Multi-Vector Embedder

The system SHALL provide ColBERT-v2 late-interaction embeddings via RAGatouille.

#### Scenario: Token-level vector generation

- **GIVEN** ColBERT-v2 model
- **WHEN** documents are encoded
- **THEN** embedder SHALL generate N token vectors (N ≤ max_doc_tokens)
- **AND** vectors SHALL be 128-D
- **AND** output SHALL include token positions for MaxSim scoring

### Requirement: Storage Routing

The system SHALL automatically route embeddings to appropriate storage backends based on namespace kind.

#### Scenario: Dense embedding routing

- **GIVEN** embeddings with kind="single_vector"
- **WHEN** storage is determined
- **THEN** router SHALL select Qdrant, FAISS, or Milvus backend
- **AND** collection SHALL use namespace dimensions

#### Scenario: Sparse embedding routing

- **GIVEN** embeddings with kind="sparse"
- **WHEN** storage is determined
- **THEN** router SHALL select OpenSearch backend
- **AND** mapping SHALL use rank_features field type

### Requirement: Batch Processing

The system SHALL support efficient batch processing with configurable sizes and GPU utilization.

#### Scenario: Automatic batching

- **GIVEN** large document set
- **WHEN** embeddings are generated
- **THEN** documents SHALL be batched (default: 32)
- **AND** batches SHALL be processed in parallel when GPU available
- **AND** progress SHALL be tracked and reported

### Requirement: GPU Fail-Fast

The system SHALL enforce GPU availability when required and fail immediately if unavailable.

#### Scenario: GPU check on initialization

- **GIVEN** embedder requires GPU
- **WHEN** embedder is initialized without CUDA
- **THEN** system SHALL raise RuntimeError
- **AND** error message SHALL indicate GPU requirement
- **AND** no CPU fallback SHALL occur

### Requirement: Evaluation Harness

The system SHALL provide embedding quality evaluation via retrieval metrics.

#### Scenario: Retrieval impact measurement

- **GIVEN** multiple embedders
- **WHEN** evaluation runs
- **THEN** harness SHALL measure Recall@10, Recall@20, nDCG@10
- **AND** metrics SHALL isolate embedding impact
- **AND** leaderboard SHALL rank by retrieval quality

**Total: 10 core requirements with 16 scenarios**

(Additional requirements for framework adapters and experimental embedders follow similar pattern)
