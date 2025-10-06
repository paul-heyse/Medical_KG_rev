# Implementation Tasks: GPU Microservices

## 1. MinerU Service

- [x] 1.1 Implement MinerU gRPC service (ProcessPDF RPC)
- [x] 1.2 Add GPU detection and fail-fast logic
- [x] 1.3 Integrate MinerU library for PDF parsing
- [x] 1.4 Add layout analysis and table extraction
- [x] 1.5 Return structured IR (Document with Blocks)
- [x] 1.6 Write MinerU service tests with sample PDFs

## 2. Embedding Service

- [x] 2.1 Implement Embedding gRPC service (EmbedChunks RPC)
- [x] 2.2 Load SPLADE model on GPU
- [x] 2.3 Load Qwen-3 dense embedding model on GPU
- [x] 2.4 Add batch processing for efficiency
- [x] 2.5 Return vector results with dimension info
- [x] 2.6 Write embedding service tests

## 3. Extraction Service

- [x] 3.1 Implement Extraction gRPC service (Extract RPC)
- [x] 3.2 Integrate LLM (GPT/Claude/local) for extraction
- [x] 3.3 Add extraction templates (PICO, effects, AE, dose)
- [x] 3.4 Implement span-grounding validation
- [x] 3.5 Return structured extraction results
- [x] 3.6 Write extraction service tests

## 4. GPU Management

- [x] 4.1 Add CUDA device detection
- [x] 4.2 Implement GPU memory management
- [x] 4.3 Add model caching to avoid reload
- [x] 4.4 Implement fail-fast on GPU unavailable
- [x] 4.5 Add GPU utilization metrics

## 5. gRPC Infrastructure

- [x] 5.1 Add gRPC health check to all services
- [x] 5.2 Implement readiness probes
- [x] 5.3 Add OpenTelemetry tracing to gRPC
- [x] 5.4 Add request/response logging
- [x] 5.5 Write gRPC middleware tests

## 6. Containerization

- [x] 6.1 Create Dockerfile.gpu with CUDA base
- [x] 6.2 Add NVIDIA runtime configuration
- [x] 6.3 Build and test GPU containers
- [x] 6.4 Add container health checks
- [x] 6.5 Write deployment documentation
