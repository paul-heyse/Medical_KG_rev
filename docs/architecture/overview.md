# Architecture Overview

The Medical KG platform is composed of multiple services connected through a multi-protocol gateway.

- **Gateway** – FastAPI application exposing REST, GraphQL, gRPC, SOAP, and SSE protocols. Shared services and adapters live in `src/Medical_KG_rev/gateway`.
- **Ingestion Pipeline** – Kafka-backed orchestration layer that coordinates adapters and workers defined in `src/Medical_KG_rev/orchestration`.
- **Storage** – Neo4j for the knowledge graph and OpenSearch for indexing and retrieval.
- **ML/GPU Services** – Embedding and extraction workloads offload to GPU-enabled microservices.
- **Observability** – Prometheus, Grafana, Jaeger, Sentry, and Loki deliver unified telemetry across the stack.

Refer to `docs/architecture` and the Engineering Blueprint PDF for deeper diagrams and sequence flows.
