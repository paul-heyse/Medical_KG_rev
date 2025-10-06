# GPU Microservices Specification

## ADDED Requirements

### Requirement: MinerU PDF Processing Service

The system SHALL provide a gRPC service for GPU-accelerated PDF parsing with layout analysis and OCR.

#### Scenario: PDF processing RPC

- **WHEN** ProcessPDF RPC is called with PDF bytes
- **THEN** service MUST return structured IR with Document and Blocks

#### Scenario: GPU fail-fast

- **WHEN** GPU is unavailable
- **THEN** service MUST return error immediately without CPU fallback

### Requirement: Embedding Generation Service

The system SHALL provide a gRPC service for generating SPLADE sparse and Qwen-3 dense embeddings on GPU.

#### Scenario: Batch embedding generation

- **WHEN** EmbedChunks RPC is called with multiple chunk IDs
- **THEN** service MUST return vectors for all chunks efficiently

### Requirement: LLM Extraction Service

The system SHALL provide a gRPC service for span-grounded information extraction using LLMs.

#### Scenario: PICO extraction

- **WHEN** Extract RPC is called with kind="pico"
- **THEN** service MUST return Population, Intervention, Comparison, Outcome spans with validation
