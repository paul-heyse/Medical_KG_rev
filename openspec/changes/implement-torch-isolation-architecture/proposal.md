## Why

The current Medical KG system has torch dependencies scattered throughout the main API gateway codebase, creating deployment complexity and maintenance overhead. The proposed torch isolation architecture addresses these issues by:

1. **Eliminated Torch Dependencies**: Complete removal of torch from main API gateway for simplified deployment
2. **Isolated GPU Processing**: GPU-accelerated services moved to dedicated Docker containers
3. **Docling Integration**: Leverage Docling's superior chunking capabilities instead of torch-based semantic chunking
4. **Service-Oriented Architecture**: gRPC-based communication between torch-free core and GPU services (aligning with project standards)
5. **Operational Flexibility**: Deploy main gateway without GPU infrastructure when not needed

This change aligns with our architectural goal to maintain a torch-free main codebase while preserving GPU-accelerated processing capabilities through isolated Docker services.

## What Changes

This change proposal implements a comprehensive torch isolation architecture:

- **BREAKING**: Remove all torch dependencies from main API gateway codebase
- **BREAKING**: Move GPU management, embedding, reranking to dedicated Docker services
- **BREAKING**: Replace torch-based chunking with Docling's built-in chunking capabilities
- **BREAKING**: Implement gRPC API communication between core gateway and GPU services
- **NEW**: Create GPU services Docker container with torch ecosystem
- **NEW**: Create embedding services Docker container for GPU-accelerated embeddings
- **NEW**: Create reranking services Docker container for cross-encoder reranking
- **NEW**: Implement service discovery and health monitoring for GPU services
- **NEW**: Add circuit breaker patterns for service communication resilience
- **NEW**: Create torch-free main gateway Docker image and deployment
- **NEW**: Implement comprehensive service integration testing and monitoring

## Impact

- **Affected specs**: deployment, services, chunking, embeddings, reranking, observability, config, orchestration
- **Dependencies**: This change assumes Docling is already integrated (see `replace-mineru-with-docling-vlm` for prerequisite)
- **Affected code**:
  - `src/Medical_KG_rev/services/gpu/` - Move to GPU services Docker container
  - `src/Medical_KG_rev/embeddings/` - Move GPU functionality to Docker service
  - `src/Medical_KG_rev/services/reranking/` - Move to reranking Docker service
  - `src/Medical_KG_rev/chunking/` - Remove torch-based semantic checks
  - `src/Medical_KG_rev/observability/` - Move GPU metrics to Docker services
  - `src/Medical_KG_rev/gateway/` - Replace torch calls with gRPC service calls
  - `src/Medical_KG_rev/orchestration/` - Update Dagster assets to use gRPC service clients
  - `ops/docker/` - Add GPU services Docker configurations
- **New dependencies**:
  - gRPC infrastructure already present (grpcio, grpcio-tools, grpcio-health-checking)
  - Service discovery and circuit breaker libraries (tenacity, pybreaker)
  - Docker multi-stage build optimizations
  - OpenTelemetry for distributed tracing across services
- **Docker containers**:
  - `medical-kg-gateway` - Torch-free main API gateway
  - `medical-kg-gpu-services` - GPU management and monitoring
  - `medical-kg-embedding-services` - GPU-accelerated embedding generation
  - `medical-kg-reranking-services` - GPU-accelerated cross-encoder reranking
- **Performance**: Expect equivalent performance with improved deployment flexibility
- **Resource Requirements**: Reduced main gateway resource requirements, GPU services scale independently
