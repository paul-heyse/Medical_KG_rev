# Implementation Tasks: GPU Microservices

## 1. MinerU Service

- [ ] 1.1 Implement MinerU gRPC service (ProcessPDF RPC)
- [ ] 1.2 Add GPU detection and fail-fast logic
- [ ] 1.3 Integrate MinerU library for PDF parsing
- [ ] 1.4 Add layout analysis and table extraction
- [ ] 1.5 Return structured IR (Document with Blocks)
- [ ] 1.6 Write MinerU service tests with sample PDFs

## 2. Embedding Service

- [ ] 2.1 Implement Embedding gRPC service (EmbedChunks RPC)
- [ ] 2.2 Load SPLADE model on GPU
- [ ] 2.3 Load Qwen-3 dense embedding model on GPU
- [ ] 2.4 Add batch processing for efficiency
- [ ] 2.5 Return vector results with dimension info
- [ ] 2.6 Write embedding service tests

## 3. Extraction Service

- [ ] 3.1 Implement Extraction gRPC service (Extract RPC)
- [ ] 3.2 Integrate LLM (GPT/Claude/local) for extraction
- [ ] 3.3 Add extraction templates (PICO, effects, AE, dose)
- [ ] 3.4 Implement span-grounding validation
- [ ] 3.5 Return structured extraction results
- [ ] 3.6 Write extraction service tests

## 4. GPU Management

- [ ] 4.1 Add CUDA device detection
- [ ] 4.2 Implement GPU memory management
- [ ] 4.3 Add model caching to avoid reload
- [ ] 4.4 Implement fail-fast on GPU unavailable
- [ ] 4.5 Add GPU utilization metrics

## 5. gRPC Infrastructure

- [ ] 5.1 Add gRPC health check to all services
- [ ] 5.2 Implement readiness probes
- [ ] 5.3 Add OpenTelemetry tracing to gRPC
- [ ] 5.4 Add request/response logging
- [ ] 5.5 Write gRPC middleware tests

## 6. Containerization

- [ ] 6.1 Create Dockerfile.gpu with CUDA base
- [ ] 6.2 Add NVIDIA runtime configuration
- [ ] 6.3 Build and test GPU containers
- [ ] 6.4 Add container health checks
- [ ] 6.5 Write deployment documentation
