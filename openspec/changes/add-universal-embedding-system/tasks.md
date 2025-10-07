# Implementation Tasks: Universal Embedding System

## 1. Core Infrastructure (15 tasks)

- [x] 1.1 Define `BaseEmbedder` protocol interface with `embed_documents()` and `embed_queries()` methods
- [x] 1.2 Create `EmbeddingRecord` Pydantic model supporting all paradigms (single-vector, multi-vector, sparse, neural-sparse)
- [x] 1.3 Define `EmbeddingKind` literal type ("single_vector", "multi_vector", "sparse", "neural_sparse")
- [x] 1.4 Create `EmbedderConfig` Pydantic model for configuration validation
- [x] 1.5 Implement embedder registry in `embeddings/registry.py` with factory pattern
- [x] 1.6 Create `EmbedderFactory` with config-driven instantiation
- [x] 1.7 Implement namespace management in `embeddings/namespace.py` with dimension governance
- [x] 1.8 Create namespace validation with automatic dimension introspection
- [x] 1.9 Build storage router mapping embedding kinds to backends (Qdrant/FAISS/OpenSearch/ColBERT)
- [x] 1.10 Implement batch processing utilities with configurable batch sizes
- [x] 1.11 Create GPU availability checker with fail-fast enforcement
- [x] 1.12 Implement embedding cache for expensive operations (LLM embeddings)
- [x] 1.13 Create normalization utilities (L2 norm for cosine similarity)
- [x] 1.14 Build pooling strategy implementations (mean, CLS, last-token, max)
- [x] 1.15 Implement prefix handler for E5-style models (query:/passage: prefixes)

## 2. Dense Bi-Encoder Adapters (12 tasks)

- [x] 2.1 Implement `SentenceTransformersEmbedder` wrapper for sentence-transformers library
- [x] 2.2 Add BGE model support (bge-small-en, bge-base-en, bge-large-en)
- [x] 2.3 Add E5 model support with automatic prefix enforcement
- [x] 2.4 Add GTE model support (gte-small, gte-base, gte-large)
- [x] 2.5 Add SPECTER model support for scientific papers
- [x] 2.6 Add SapBERT model support for biomedical entity matching
- [x] 2.7 Implement `TEIHTTPEmbedder` for HuggingFace Text-Embeddings-Inference server
- [x] 2.8 Add Jina v3 embedding support via TEI
- [x] 2.9 Implement `OpenAICompatEmbedder` for vLLM-served models (Qwen-3)
- [x] 2.10 Add automatic dimension introspection and validation
- [x] 2.11 Implement batch processing with progress tracking
- [x] 2.12 Add ONNX optimization support for CPU deployment (optional)

## 3. Late-Interaction Multi-Vector Adapters (6 tasks)

- [x] 3.1 Implement `ColBERTRagatouilleEmbedder` wrapper for RAGatouille library
- [x] 3.2 Add ColBERT-v2 model support with token-level vectors
- [x] 3.3 Implement max_doc_tokens truncation and padding
- [x] 3.4 Create FAISS shard management for ColBERT indexes
- [x] 3.5 Implement MaxSim scoring utilities
- [x] 3.6 Add integration with Qdrant multivector storage (optional alternative)

## 4. Learned-Sparse Adapters (8 tasks)

- [x] 4.1 Implement `SPLADEDocEmbedder` for document-side SPLADE expansion
- [x] 4.2 Add SPLADE-v3-lexical model support
- [x] 4.3 Implement top-K term selection (default: 400 terms)
- [x] 4.4 Create `SPLADEQueryEmbedder` for optional query-side encoding
- [x] 4.5 Implement `PyseriniSparseEmbedder` wrapper for uniCOIL/DeepImpact/TILDE
- [x] 4.6 Add OpenSearch rank_features field mapping utilities
- [x] 4.7 Implement term weight normalization strategies
- [x] 4.8 Add vocabulary tracking for sparse embeddings

## 5. Neural-Sparse Adapters (5 tasks)

- [x] 5.1 Implement `OpenSearchNeuralSparseEmbedder` for OS ML plugin integration
- [x] 5.2 Add support for encoder hosting via ML plugin
- [x] 5.3 Add support for external TEI endpoint
- [x] 5.4 Create neural query type generation for OpenSearch
- [x] 5.5 Implement neural-sparse field mapping

## 6. Framework Integration Adapters (9 tasks)

- [x] 6.1 Create `LangChainEmbedderAdapter` wrapper for langchain.embeddings.*
- [x] 6.2 Add LangChain HuggingFace embeddings support
- [x] 6.3 Add LangChain OpenAI embeddings support (via vLLM)
- [x] 6.4 Create `LlamaIndexEmbedderAdapter` wrapper for llama_index.embeddings.*
- [x] 6.5 Add LlamaIndex HuggingFace embeddings support
- [x] 6.6 Add LlamaIndex OpenAI embeddings support (via vLLM)
- [x] 6.7 Create `HaystackEmbedderAdapter` wrapper for haystack embedders
- [x] 6.8 Implement offset mapping for framework adapters to preserve metadata
- [x] 6.9 Add configuration validation for framework-specific parameters

## 7. Experimental Embedders (8 tasks)

- [x] 7.1 Implement `SimLMEmbedder` with representation bottleneck
- [x] 7.2 Add SimLM model loading and inference
- [x] 7.3 Implement `RetroMAEEmbedder` with masked autoencoder approach
- [x] 7.4 Add RetroMAE model loading and inference
- [x] 7.5 Implement `GTREmbedder` for T5-based embeddings
- [x] 7.6 Add GTR model support (Base/Large/XXL variants)
- [x] 7.7 Create `DSISearcher` skeleton for differentiable search index (optional)
- [x] 7.8 Mark all experimental embedders with appropriate warnings

## 8. Configuration System (7 tasks)

- [x] 8.1 Create `config/embeddings.yaml` with namespace configuration
- [x] 8.2 Add per-provider configuration blocks (driver, model_id, endpoint, parameters)
- [x] 8.3 Create active_namespaces configuration for query-time fusion
- [x] 8.4 Add embedding-specific parameters (pooling, normalization, prefixes, batch_size)
- [x] 8.5 Implement configuration validation with Pydantic models
- [x] 8.6 Add environment-based configuration overrides
- [x] 8.7 Create embedder registry population from configuration

## 9. Ingestion Service Integration (6 tasks)

- [x] 9.1 Extend `IngestionService` to invoke embedding pipeline after chunking
- [x] 9.2 Implement namespace selection based on chunk configuration
- [x] 9.3 Add batch processing for chunks with progress tracking
- [x] 9.4 Integrate with storage router for namespace-based persistence
- [x] 9.5 Add telemetry for embedding latency and batch efficiency
- [x] 9.6 Implement error handling and retry logic for embedding failures

## 10. Storage Router Integration (5 tasks)

- [x] 10.1 Create storage router mapping namespaces to backends
- [x] 10.2 Implement dense embedding routing to Qdrant/FAISS/Milvus
- [x] 10.3 Implement sparse embedding routing to OpenSearch rank_features
- [x] 10.4 Implement multi-vector routing to ColBERT FAISS or Qdrant multivector
- [x] 10.5 Implement neural-sparse routing to OpenSearch neural fields

## 11. Retrieval Service Integration (6 tasks)

- [x] 11.1 Extend retrieval service to support multi-strategy embedding
- [x] 11.2 Implement query encoding per active namespace
- [x] 11.3 Add parallel query execution across namespaces
- [x] 11.4 Integrate with fusion layer for multi-strategy results
- [x] 11.5 Add namespace-specific scoring and weights
- [x] 11.6 Implement query-side optimizations (caching, batching)

## 12. Evaluation Harness (6 tasks)

- [x] 12.1 Create `eval/embedding_eval.py` evaluation runner
- [x] 12.2 Implement retrieval metrics (Recall@K, nDCG@K, MRR)
- [x] 12.3 Add zero-shot benchmark support (BEIR, MTEB subsets)
- [x] 12.4 Create embedding quality metrics (semantic similarity correlations)
- [x] 12.5 Implement A/B testing framework for embedder comparison
- [x] 12.6 Create leaderboard visualization per namespace

## 13. Testing (10 tasks)

- [x] 13.1 Create unit tests for BaseEmbedder interface and EmbeddingRecord model
- [x] 13.2 Add tests for each dense embedder with mock/real models
- [x] 13.3 Create tests for sparse embedders with term weight validation
- [x] 13.4 Add tests for multi-vector embedders with ColBERT integration
- [x] 13.5 Create tests for framework adapters with library integration
- [x] 13.6 Add integration tests for namespace management and dimension validation
- [x] 13.7 Create integration tests for storage routing
- [x] 13.8 Add performance tests for batch processing efficiency
- [x] 13.9 Create tests for GPU fail-fast behavior
- [x] 13.10 Add tests for configuration validation and registry

## 14. Documentation (5 tasks)

- [x] 14.1 Write developer guide for adding new embedding adapters
- [x] 14.2 Document each embedder's model, parameters, and use cases
- [x] 14.3 Create configuration examples for common scenarios
- [x] 14.4 Write evaluation harness usage guide
- [x] 14.5 Add API documentation for embeddings module

## 15. Dependencies & Setup (4 tasks)

- [x] 15.1 Add all required dependencies to `pyproject.toml`
- [x] 15.2 Configure TEI server setup documentation
- [x] 15.3 Configure vLLM embedding endpoint setup
- [x] 15.4 Add model download scripts and documentation

## 16. Stabilization Fixes (4 tasks)

- [x] 16.1 Restore fallback chunk metadata compatibility with indexing pipeline
- [x] 16.2 Ensure sliding window fallback returns overlapping segments
- [x] 16.3 Update retrieval indexing to support normalized chunk objects
- [x] 16.4 Align GPU registry namespaces with validation requirements

**Total: 116 tasks across 16 categories**
