# Add Universal Embedding System

## Why

The current system requires a flexible, high-performance embedding infrastructure supporting diverse retrieval strategies: dense bi-encoders, learned sparse (SPLADE), neural-sparse, and multi-vector late-interaction (ColBERT). We need a universal embedding system that:

1. Supports 4 embedding paradigms through unified interface: single-vector dense, multi-vector (ColBERT), learned-sparse (SPLADE/uniCOIL), neural-sparse (OpenSearch)
2. Integrates production embedding models (BGE, E5, GTE, SPECTER, SapBERT) and LLM-based embedders (Qwen-3 via vLLM)
3. Provides adapter wrappers for Sentence-Transformers, HuggingFace TEI, LangChain, LlamaIndex, Haystack embedders
4. Enables namespace-based embedding management with dimension/version governance
5. Supports GPU-accelerated batch processing with fail-fast on unavailability
6. Implements experimental embedders (SimLM, RetroMAE, GTR) alongside production models
7. Provides per-namespace storage routing (Qdrant for dense, OpenSearch for sparse, ColBERT FAISS for multi-vector)
8. Includes evaluation harness for embedding quality (retrieval metrics, zero-shot benchmarks)

This is critical for achieving state-of-the-art retrieval through hybrid strategies (dense + sparse + late-interaction) while maintaining modularity for research and optimization.

## What Changes

### Core Infrastructure

- **BaseEmbedder interface** with `embed_documents()` and `embed_queries()` methods
- **EmbeddingRecord model** supporting all paradigms (single-vector, multi-vector, sparse, neural-sparse)
- **Embedder registry** with 20+ adapter implementations (production + frameworks + experimental)
- **Namespace management** with strict dimension/version governance and automatic introspection
- **Configuration-driven model selection** via YAML with provider-specific parameters
- **Storage routing** by embedding kind to appropriate backend (Qdrant/FAISS/OpenSearch/ColBERT)

### Dense Bi-Encoder Adapters (Production)

- `SentenceTransformersEmbedder` - BGE, E5, GTE, SPECTER, SapBERT with configurable pooling
- `TEIHTTPEmbedder` - HuggingFace Text-Embeddings-Inference server integration
- `OpenAICompatEmbedder` - vLLM-served Qwen-3 embedding models (0.6B-8B)
- E5-style prefix enforcement (query:/passage:) via configuration
- L2 normalization for cosine similarity

### Late-Interaction Multi-Vector Adapters

- `ColBERTRagatouilleEmbedder` - ColBERT-v2 via RAGatouille with MaxSim scoring
- Token-level vector generation and storage integration
- FAISS shard management for ColBERT indexes

### Learned-Sparse Adapters

- `SPLADEDocEmbedder` - SPLADE-v3 document-side expansion with top-K terms
- `SPLADEQueryEmbedder` - Optional query-side SPLADE encoding
- `PyseriniSparseEmbedder` - uniCOIL/DeepImpact/TILDE exporters
- OpenSearch rank_features field mapping

### Neural-Sparse Adapters

- `OpenSearchNeuralSparseEmbedder` - OpenSearch ML neural-sparse pipeline integration
- Encoder hosting (ML plugin) or external TEI endpoint
- Neural query type generation

### Framework Integration Adapters

- `LangChainEmbedderAdapter` - Wrap LangChain embedding classes
- `LlamaIndexEmbedderAdapter` - Wrap LlamaIndex embedding classes
- `HaystackEmbedderAdapter` - Wrap Haystack embedders

### Experimental Embedders

- `SimLMEmbedder` - Representation bottleneck pre-training
- `RetroMAEEmbedder` - Masked autoencoder for retrieval
- `GTREmbedder` - T5-based General Text Representations
- `DSISearcher` - Differentiable Search Index (optional research track)

### Configuration & Orchestration

- Per-namespace configuration with model_id, dim, pooling, normalization
- Active namespace selection for query-time fusion
- Batch processing with configurable sizes
- GPU availability checks with fail-fast enforcement
- Embedding cache for expensive operations

### Integration Points

- Ingestion pipeline integration after chunking
- Namespace-based routing to storage backends
- Retrieval service multi-strategy embedding coordination
- Telemetry for embedding latency and batch sizes

## Impact

### Affected Specs

- **New**: `embedding-core` (interfaces, registry, namespace management)
- **New**: `embedding-adapters` (production and framework adapters)
- **New**: `embedding-experimental` (research-grade embedders)
- **Modified**: `ingestion-orchestration` (embedding pipeline integration)
- **Modified**: `retrieval-system` (multi-strategy embedding coordination)
- **Modified**: `vector-storage-retrieval` (namespace-based storage routing)

### Affected Code

- **New**: `src/Medical_KG_rev/embeddings/` (entire module)
  - `ports.py` - BaseEmbedder interface, EmbeddingRecord model
  - `registry.py` - Embedder registry and factory
  - `namespace.py` - Namespace management and validation
  - `dense/` - Dense bi-encoder adapters
  - `sparse/` - Learned-sparse adapters
  - `multi_vector/` - ColBERT late-interaction
  - `neural_sparse/` - OpenSearch neural-sparse
  - `frameworks/` - LangChain/LlamaIndex/Haystack wrappers
  - `experimental/` - Research embedders
  - `utils/` - Batch processing, GPU checks, caching
- **Modified**: `src/Medical_KG_rev/orchestration/ingestion_service.py`
- **Modified**: `src/Medical_KG_rev/services/retrieval_service.py`
- **New**: `config/embeddings.yaml` - Embedding configuration
- **New**: `tests/embeddings/` - Comprehensive test suite
- **New**: `eval/embedding_eval.py` - Evaluation harness

### Dependencies Added

- `sentence-transformers>=2.3.0` (dense embeddings)
- `FlagEmbedding>=1.2.0` (BGE models)
- `text-embeddings-inference[client]>=1.0.0` (TEI integration)
- `ragatouille>=0.0.7` (ColBERT integration)
- `pyserini>=0.22.0` (sparse embeddings - uniCOIL/DeepImpact)
- `transformers>=4.37.0` (model loading)
- `torch>=2.1.0` (GPU support)
- `onnxruntime-gpu>=1.16.0` (ONNX optimization, optional)

### Breaking Changes

None - this is a new capability that integrates with existing pipeline without modifying current behavior unless explicitly configured.

### Benefits

- **Multi-paradigm**: Support dense, sparse, multi-vector, neural-sparse in unified system
- **Production-ready**: Battle-tested models (BGE, E5, GTE) with proven retrieval performance
- **Hybrid retrieval**: Enable SOTA combinations (dense + SPLADE + ColBERT)
- **Namespace isolation**: Prevent dimension/version conflicts through strict governance
- **Framework compatibility**: Wrap existing ecosystem (LangChain, LlamaIndex, Haystack)
- **Research-friendly**: Experimental embedders (SimLM, RetroMAE, GTR) pre-wired
- **English-optimized**: Default models tuned for English biomedical text
- **GPU-efficient**: Batch processing with fail-fast on unavailability
- **Eval-driven**: Built-in harness for retrieval impact measurement
- **LLM integration**: Qwen-3 embeddings via existing vLLM infrastructure
