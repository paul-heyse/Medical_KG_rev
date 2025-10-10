## Why

The current PDF processing pipeline using MinerU + vLLM in Docker containers has served well but has limitations in document understanding, accuracy, and infrastructure complexity. The proposed Docling VLM retrieval pipeline addresses these issues by:

1. **Enhanced Document Understanding**: Docling's vision-language model provides superior semantic understanding compared to traditional OCR approaches
2. **Improved Retrieval Accuracy**: Hybrid retrieval with BM25, SPLADE-v3, and Qwen3 embeddings achieves higher precision and recall
3. **Simplified Infrastructure**: Single VLM model vs dual OCR+LLM pipeline reduces complexity and maintenance overhead
4. **Medical Domain Optimization**: Specialized handling for medical document structures, tables, and terminology
5. **Provenance and Reproducibility**: Comprehensive audit trails and deterministic processing for clinical trust

This change aligns with our goal to eliminate torch dependencies from the main codebase while maintaining GPU-accelerated processing in Docker containers.

## What Changes

This change proposal implements a comprehensive Docling VLM-based PDF processing and retrieval pipeline:

- **BREAKING**: Replace MinerU + vLLM with Docling VLM in Docker containers
- **BREAKING**: Implement hybrid retrieval system (BM25 + SPLADE-v3 + Qwen3 embeddings)
- **BREAKING**: Redesign storage model with chunk store + separate indexes + manifests
- **BREAKING**: Implement deterministic chunking with tokenizer-aligned segmentation
- **BREAKING**: Add SPLADE-v3 with Rep-Max aggregation for learned sparse retrieval
- **BREAKING**: Integrate Qwen3 4096-dimension dense embeddings with FAISS/Qdrant storage
- **BREAKING**: Implement structured BM25 indexing with medical corpus optimizations
- **BREAKING**: Add comprehensive provenance tracking with immutable identifiers
- Add orchestration pipeline with discrete, restartable stages
- Add comprehensive testing, monitoring, and validation framework
- Add medical domain-specific normalization and processing
- Add feature flag system for gradual migration and rollback

## Impact

- **Affected specs**: orchestration, services, retrieval, storage, config, gateway
- **Affected code**:
  - `src/Medical_KG_rev/services/mineru/` - Replace with Docling VLM integration
  - `src/Medical_KG_rev/orchestration/stages/` - Update PDF processing stages
  - `src/Medical_KG_rev/services/retrieval/` - Implement hybrid retrieval system
  - `src/Medical_KG_rev/services/vector_store/` - Add SPLADE and Qwen3 support
  - `src/Medical_KG_rev/chunking/` - Implement hybrid chunker with tokenizer alignment
  - `src/Medical_KG_rev/config/` - Add Docling VLM and retrieval configurations
  - `src/Medical_KG_rev/storage/` - Implement chunk store + index storage model
  - `src/Medical_KG_rev/gateway/` - Update retrieval endpoints for hybrid search
- **New dependencies**:
  - `docling-core>=2.0.0` - Core Docling functionality (repo)
  - `transformers>=4.36.0` - Model loading and inference
  - `torch>=2.1.0` - PyTorch with CUDA (Docker only)
  - `faiss-cpu>=1.12.0` - Vector similarity search
  - `pyserini>=1.2.0` - Information retrieval toolkit
  - `duckdb>=1.4.1` - Chunk store database
- **Docker-only dependencies**:
  - `docling[vlm]>=2.0.0` - Vision-language model PDF processing (Docker)
  - `vllm>=0.11.0` - High-performance LLM serving (Docker)
- **Migration**: Existing MinerU-processed documents remain accessible; new documents use Docling
- **Performance**: Expect 20-30% accuracy improvement with 15-25% processing time increase
- **Resource Requirements**: Docling VLM requires ~24GB VRAM vs ~8GB for current vLLM setup
