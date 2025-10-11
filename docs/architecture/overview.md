# Architecture Overview

The Medical KG platform is composed of multiple services connected through a multi-protocol gateway with a torch-free architecture.

## Core Architecture

- **Gateway** – FastAPI application exposing REST, GraphQL, gRPC, SOAP, and SSE protocols. Shared services and adapters live in `src/Medical_KG_rev/gateway`. **Torch-free** - no PyTorch dependencies in the main gateway.
- **Ingestion Pipeline** – Kafka-backed orchestration layer that coordinates adapters and workers defined in `src/Medical_KG_rev/orchestration`.
- **Storage** – Neo4j for the knowledge graph and OpenSearch for indexing and retrieval.
- **GPU Services** – Dedicated Docker containers for GPU-intensive workloads:
  - **GPU Management Service** – gRPC service for GPU resource allocation and monitoring
  - **Embedding Service** – gRPC service for embedding generation using transformer models
  - **Reranking Service** – gRPC service for cross-encoder reranking models
  - **Docling VLM Service** – gRPC service for document processing using vision-language models
- **Observability** – Prometheus, Grafana, Jaeger, Sentry, and Loki deliver unified telemetry across the stack.

## Service Communication

All GPU services communicate via gRPC with:

- **Circuit breaker patterns** for network resilience
- **Service discovery** for dynamic service location
- **Health checks** using gRPC health protocol
- **Mutual TLS (mTLS)** for service-to-service authentication
- **OpenTelemetry tracing** for distributed observability

## Torch Isolation Benefits

- **Reduced main gateway size** – No PyTorch dependencies in production gateway image
- **Independent scaling** – GPU services can scale independently based on workload
- **Resource isolation** – GPU memory and compute resources are isolated per service
- **Fail-fast behavior** – Services fail immediately if GPU unavailable (no CPU fallback)
- **Simplified deployment** – Main gateway can run on CPU-only infrastructure

Refer to `docs/architecture` and the Engineering Blueprint PDF for deeper diagrams and sequence flows.
