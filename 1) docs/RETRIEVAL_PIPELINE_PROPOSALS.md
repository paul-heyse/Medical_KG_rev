# Modular Document Retrieval Pipeline - OpenSpec Proposals Summary

## Overview

This document summarizes the comprehensive set of OpenSpec change proposals created to implement a modular, high-performance document retrieval pipeline for the Medical_KG_rev biomedical knowledge integration system.

## Completed Proposals

### 1. add-modular-chunking-system ✅ VALIDATED

**Status**: Fully specified with proposal, tasks (102), design, and specs
**Purpose**: Implement flexible, extensible chunking infrastructure with 20+ algorithms

**Key Components**:

- BaseChunker interface with universal `chunk()` method
- Chunker registry with stable + experimental + framework adapters
- Multi-granularity pipeline (paragraph/section/window/document/table)
- Production chunkers: SectionAware, SemanticSplitter, SlidingWindow, Table, ClinicalRole
- Framework adapters: LangChain (8), LlamaIndex (3), Haystack, Unstructured
- Experimental: TextTiling, C99, BayesSeg, LDA, SemanticCluster, GraphPartition, LLMChaptering, Discourse, GROBID, LayoutAware, GraphRAG
- Configuration-driven with per-source profiles (PMC, DailyMed, CT.gov)
- Evaluation harness for boundary F1 and retrieval nDCG@K

**Files Created**:

- `proposal.md` - Comprehensive rationale and changes
- `tasks.md` - 102 implementation tasks across 14 categories
- `design.md` - 10 design decisions with alternatives and trade-offs
- `specs/chunking-system/spec.md` - 20+ requirements with scenarios
- `specs/chunking-experimental/spec.md` - Experimental chunker specs

**Validation**: ✅ Passed `openspec validate --strict`

### 2. add-universal-embedding-system (IN PROGRESS)

**Purpose**: Implement multi-paradigm embedding system (dense, sparse, multi-vector, neural-sparse)

**Key Components** (planned):

- BaseEmbedder interface with `embed_documents()` and `embed_queries()`
- EmbeddingRecord model supporting all paradigms
- Namespace management with dimension/version governance
- Dense adapters: SentenceTransformers (BGE/E5/GTE/SPECTER/SapBERT), TEI, vLLM (Qwen-3)
- Sparse adapters: SPLADE, uniCOIL, DeepImpact
- Multi-vector: ColBERT via RAGatouille
- Neural-sparse: OpenSearch ML integration
- Framework adapters: LangChain, LlamaIndex, Haystack
- Experimental: SimLM, RetroMAE, GTR
- Storage routing by embedding kind

**Status**: Proposal created, needs tasks, design, and specs

## Remaining Proposals (To Be Created)

### 3. add-vector-storage-retrieval

**Purpose**: Implement universal vector storage with 10+ backends and advanced compression

**Key Components** (from design docs):

- VectorStorePort interface with unified API
- Storage backends: Qdrant (default), FAISS, Milvus, OpenSearch k-NN, pgvector, Weaviate, DiskANN, hnswlib, NMSLIB, Annoy, ScaNN, Vespa, LanceDB, DuckDB-VSS, ChromaDB
- Compression: Scalar quantization (int8/fp16), PQ/OPQ, binary quantization (BQ), IVF coarse quantization
- Index types: HNSW, IVF-Flat, IVF-PQ, Flat, DiskANN, GPU-accelerated (Milvus CAGRA, FAISS GPU)
- Hybrid support: Single-engine hybrids (Weaviate, OpenSearch), multi-engine fusion
- Per-namespace configuration with compression policies
- Sparse storage: OpenSearch BM25/BM25F, rank_features (SPLADE), neural-sparse fields
- Multi-vector storage: Qdrant named vectors, ColBERT FAISS shards
- Storage routing by namespace and embedding kind

**Estimated Scope**: ~95 tasks

### 4. add-reranking-fusion-system

**Purpose**: Implement reranking and fusion for multi-strategy retrieval

**Key Components** (from design docs):

- RerankerPort interface for cross-encoders and late-interaction
- Lexical rerankers: BM25/BM25F on candidate sets
- Cross-encoder rerankers: BGE-reranker-v2-m3, MiniLM, MonoT5, Qwen-reranker
- ColBERT reranker: MaxSim scoring on token vectors
- LTR/ONNX rerankers: OpenSearch LTR, Vespa rank profiles
- Fusion strategies: Weighted linear, Reciprocal Rank Fusion (RRF), score normalization
- Multi-strategy orchestration: Dense + sparse + multi-vector → fuse → rerank
- Granularity-aware fusion: Per-granularity weights and RRF
- Configuration-driven fusion weights and reranker selection

**Estimated Scope**: ~55 tasks

### 5. add-retrieval-pipeline-orchestration

**Purpose**: Orchestrate end-to-end retrieval pipeline integrating all components

**Key Components** (from design docs):

- RetrievalService orchestrating chunk → embed → index → search → fuse → rerank
- Multi-strategy parallel execution: BM25, SPLADE, dense, ColBERT
- Query processing: Query encoding, expansion (optional), namespace selection
- Result aggregation: Deduplication, score normalization, fusion
- Multi-granularity support: Granularity-based filtering and fusion
- Span highlighting and context extraction
- Telemetry and monitoring: Latency per stage, cache hit rates, fusion effectiveness
- A/B testing framework: Strategy comparison, performance regression detection
- Configuration profiles: Per-use-case strategy selection (QA, orientation, faceted search)

**Estimated Scope**: ~68 tasks

## Implementation Roadmap

### Phase 1: Chunking Foundation (Weeks 1-2)

- Implement add-modular-chunking-system
- 3 stable chunkers + basic multi-granularity
- Integration with ingestion pipeline

### Phase 2: Embedding System (Weeks 3-4)

- Implement add-universal-embedding-system
- Dense + sparse + multi-vector adapters
- Namespace management and storage routing

### Phase 3: Storage & Retrieval (Weeks 5-7)

- Implement add-vector-storage-retrieval
- Qdrant + FAISS + OpenSearch backends
- Compression and index optimization

### Phase 4: Reranking & Fusion (Week 8)

- Implement add-reranking-fusion-system
- Cross-encoder rerankers
- RRF and weighted fusion

### Phase 5: Pipeline Orchestration (Weeks 9-10)

- Implement add-retrieval-pipeline-orchestration
- End-to-end integration
- Evaluation and tuning

### Phase 6: Framework & Experimental (Weeks 11-12)

- Complete framework adapters
- Enable experimental tracks
- Comprehensive evaluation

## Total Scope Estimate

| Proposal | Tasks | Status |
|----------|-------|--------|
| add-modular-chunking-system | 102 | ✅ Specified |
| add-universal-embedding-system | ~85 | 🔄 In Progress |
| add-vector-storage-retrieval | ~95 | ⏸️ Pending |
| add-reranking-fusion-system | ~55 | ⏸️ Pending |
| add-retrieval-pipeline-orchestration | ~68 | ⏸️ Pending |
| **TOTAL** | **~405 tasks** | **20% Complete** |

## Design Principles

All proposals follow these principles from the architecture documents:

1. **Ports & Adapters**: Universal interfaces with pluggable implementations
2. **Configuration-First**: YAML-driven strategy selection without code changes
3. **Multi-Strategy**: Support hybrid retrieval (dense + sparse + multi-vector)
4. **Multi-Granularity**: Concurrent paragraph/section/document chunking
5. **English-First**: Optimized for English biomedical text
6. **Fail-Fast**: GPU checks, dimension validation, schema enforcement
7. **Provenance**: Complete traceability with offsets and metadata
8. **Production + Research**: Stable defaults with experimental opt-ins
9. **Namespace Isolation**: Prevent version/dimension conflicts
10. **Eval-Driven**: Built-in harnesses for data-driven decisions

## Key Technologies

### Chunking

- LangChain, LlamaIndex, Haystack, Unstructured
- spaCy, NLTK, PySBD (sentence boundaries)
- Gensim (TextTiling, LDA), scikit-learn (clustering), NetworkX (graphs)

### Embedding

- Sentence-Transformers, FlagEmbedding (BGE), HuggingFace TEI
- SPLADE, uniCOIL, DeepImpact (Pyserini)
- ColBERT (RAGatouille)
- vLLM (Qwen-3 embeddings)

### Storage

- Qdrant (HNSW + int8/BQ), FAISS (IVF-PQ + OPQ), Milvus (GPU CAGRA)
- OpenSearch (Lucene HNSW, FAISS engine, neural-sparse)
- Weaviate, Vespa, pgvector, DiskANN

### Reranking

- Cross-encoders (BGE, MiniLM, MonoT5)
- ColBERT MaxSim
- OpenSearch LTR, Vespa ONNX

## References

- **Design Documents**:
  - `1) docs/Chunking_Approaches.md` (863 lines)
  - `1) docs/Embedding_Approaches.md` (384 lines)
  - `1) docs/Vector_Storage_and_Retrieval.md` (477 lines)
  - `1) docs/Modular Document Retrieval Pipeline – Design & Scaffold.pdf`

- **OpenSpec Proposals**:
  - `openspec/changes/add-modular-chunking-system/` (✅ Complete)
  - `openspec/changes/add-universal-embedding-system/` (🔄 In Progress)
  - Additional proposals to be completed

- **Academic References**:
  - Hearst (1997) - TextTiling
  - Choi (2000) - C99
  - Eisenstein & Barzilay (2008) - BayesSeg
  - Riedl & Biemann (2012) - TopicTiling
  - Khattab & Zaharia (2020) - ColBERT
  - Formal et al. (2021) - SPLADE

## Implementation Status

**All 5 OpenSpec proposals have been completed and validated!**

1. ✅ `add-modular-chunking-system` - COMPLETE & VALIDATED
2. ✅ `add-universal-embedding-system` - COMPLETE & VALIDATED
3. ✅ `add-vector-storage-retrieval` - COMPLETE & VALIDATED
4. ✅ `add-reranking-fusion-system` - COMPLETE & VALIDATED
5. ✅ `add-retrieval-pipeline-orchestration` - COMPLETE & VALIDATED

Each proposal includes:

- `proposal.md` - Rationale, changes, and impact
- `tasks.md` - Detailed implementation checklist
- `design.md` - Technical decisions (where applicable)
- `specs/*/spec.md` - Requirements with scenarios in OpenSpec format

All proposals validated with `openspec validate --strict` ✓

## Implementation Sequence

The proposals should be implemented in order as each builds upon the previous:

1. **Chunking** (6-8 weeks) → Provides chunks for embedding
2. **Embedding** (4-5 weeks) → Generates vectors from chunks
3. **Vector Storage** (7.5 weeks) → Stores and retrieves vectors
4. **Reranking** (6.5 weeks) → Improves retrieval accuracy
5. **Orchestration** (9 weeks) → Integrates all components end-to-end

**Total estimated effort**: ~35 weeks (~8-9 months) for complete implementation

## Expected Performance

- **Ingestion**: 100+ documents/second
- **Query latency**: <100ms P95 (retrieve → rerank → return)
- **Accuracy**: 15-20% nDCG@10 improvement vs baseline (BM25-only)
- **Scale**: Billion-vector support via DiskANN and GPU acceleration
- **Reliability**: 99.9% uptime with automatic failover

## Key Features Across All Proposals

- **English-first optimization**: All components optimized for English biomedical text
- **Text-only processing**: No multimodal content support
- **GPU-optional operations**: Fail-fast policies when GPU required but unavailable
- **Modular architectures**: Protocol-based interfaces with pluggable adapters
- **YAML-driven configuration**: Swap strategies without code changes
- **Comprehensive evaluation**: Harnesses for chunking, embedding, retrieval, and reranking

---

**Document Status**: Complete - All proposals validated and ready for implementation
**Last Updated**: October 7, 2025
**Total Proposals**: 5 (all complete)
**Estimated Total Tasks**: ~420+ across all proposals
