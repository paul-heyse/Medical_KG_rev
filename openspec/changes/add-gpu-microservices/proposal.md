# Change Proposal: GPU Microservices

## Why

Implement GPU-bound microservices for PDF parsing (MinerU), embedding generation (SPLADE + Qwen-3), and information extraction (LLM-based) exposed via gRPC. These services enforce fail-fast behavior (no CPU fallback), include health checks, and integrate with the orchestration system for heavy workloads.

## What Changes

- MinerU gRPC service for PDF layout analysis and OCR
- Embedding gRPC service for SPLADE sparse and Qwen-3 dense vectors
- Extraction gRPC service for LLM-based entity/fact extraction
- GPU availability detection and fail-fast enforcement
- CUDA device management and allocation
- Model loading and caching
- Batch processing for efficiency
- gRPC health check and readiness probes
- Prometheus metrics for GPU utilization
- Docker images with CUDA support

## Impact

- **Affected specs**: NEW capability `gpu-microservices`
- **Affected code**:
  - `src/Medical_KG_rev/services/mineru/` - MinerU gRPC service
  - `src/Medical_KG_rev/services/embedding/` - Embedding service
  - `src/Medical_KG_rev/services/extraction/` - Extraction service
  - `proto/mineru.proto`, `proto/embedding.proto`, `proto/extraction.proto`
  - `ops/Dockerfile.gpu` - GPU-enabled containers
  - `tests/services/` - Service integration tests with GPU mocks
