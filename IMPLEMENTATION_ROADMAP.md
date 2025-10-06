# Medical_KG_rev Implementation Roadmap

## Overview

This document provides a comprehensive roadmap for implementing the multi-protocol API gateway and orchestration system for biomedical knowledge integration, based on the Engineering Blueprint and Public Biomedical APIs addendum.

## Architecture Summary

The system is a sophisticated multi-protocol API gateway that:

- Ingests data from 10+ biomedical sources (clinical trials, literature, drug databases, ontologies)
- Processes content using GPU-accelerated services (PDF parsing, embeddings, extraction)
- Stores data in a federated knowledge graph (Neo4j) with provenance tracking
- Provides multi-strategy retrieval (BM25 + SPLADE + dense vectors)
- Exposes functionality via 5 protocols: REST (OpenAPI/JSON:API/OData), GraphQL, gRPC, SOAP, AsyncAPI/SSE
- Enforces OAuth 2.0 authentication with fine-grained scopes and multi-tenant isolation
- Provides production-grade observability (Prometheus, OpenTelemetry, Grafana)

## Implementation Sequence

The implementation is divided into 9 major change proposals that should be implemented sequentially:

### 1. Foundation Infrastructure (`add-foundation-infrastructure`)

**Purpose**: Core data models, configuration, utilities, and adapter SDK

**Key Deliverables**:

- Pydantic models for federated IR (Document, Block, Entity, Claim)
- Domain overlays (medical/FHIR, finance/XBRL, legal/LegalDocML)
- Adapter SDK base classes with YAML config support
- Shared utilities (HTTP client, logging, validation, errors)
- Storage abstractions (object store, ledger, cache)
- Provenance tracking models

**Dependencies**: None (foundation)

**Estimated Complexity**: Medium (40-50 tasks)

---

### 2. Multi-Protocol API Gateway (`add-multi-protocol-gateway`)

**Purpose**: Unified API gateway exposing REST, GraphQL, gRPC, SOAP, and AsyncAPI/SSE

**Key Deliverables**:

- FastAPI application with all REST endpoints (ingest, chunk, embed, retrieve, extract, kg/write)
- JSON:API response formatting + OData query support
- GraphQL schema and resolvers auto-generated from Pydantic
- gRPC service definitions (.proto files with Buf)
- Server-Sent Events for job status streaming
- AsyncAPI specification
- Minimal SOAP adapter for legacy support
- Interactive documentation portal (Swagger UI, GraphQL Playground, AsyncAPI UI)

**Dependencies**: Foundation Infrastructure

**Estimated Complexity**: High (40+ tasks)

---

### 3. Biomedical Data Source Adapters (`add-biomedical-adapters`)

**Purpose**: Plug-in adapters for ingesting from 10+ biomedical APIs

**Key Deliverables**:

- ClinicalTrials.gov API v2 adapter
- OpenFDA adapters (drug labels, adverse events, devices)
- Literature adapters (OpenAlex/pyalex, PubMed Central, Unpaywall, Crossref, CORE)
- Ontology adapters (RxNorm, ICD-11, MeSH)
- ChEMBL chemistry adapter
- Semantic Scholar citation adapter
- YAML adapter configurations
- Rate limiting and resilience (retry, backoff, circuit breaking)

**Dependencies**: Foundation Infrastructure

**Estimated Complexity**: High (45+ tasks across 11 adapters)

---

### 4. Ingestion & Orchestration System (`add-ingestion-orchestration`)

**Purpose**: Kafka-based orchestration for multi-step pipelines

**Key Deliverables**:

- Apache Kafka setup with topics (ingest.requests.v1, ingest.results.v1, mapping.events.v1)
- Job ledger for state tracking and idempotency
- Orchestrator service with auto-pipeline and two-phase pipeline
- Multi-adapter chaining for literature enrichment (OpenAlex → Unpaywall → CORE → MinerU)
- Background workers for consuming Kafka messages
- Job status API and SSE streaming
- Dead letter queue for failed jobs

**Dependencies**: Foundation Infrastructure, Multi-Protocol Gateway, Biomedical Adapters

**Estimated Complexity**: High (30+ tasks)

---

### 5. GPU Microservices (`add-gpu-microservices`)

**Purpose**: GPU-accelerated services for heavy processing

**Key Deliverables**:

- MinerU gRPC service for PDF parsing (layout analysis + OCR)
- Embedding gRPC service (SPLADE sparse + Qwen-3 dense vectors)
- Extraction gRPC service (LLM-based span-grounded extraction)
- GPU fail-fast enforcement (no CPU fallback)
- Model loading and caching
- Batch processing for efficiency
- Docker images with CUDA support

**Dependencies**: Foundation Infrastructure (gRPC protos from Gateway)

**Estimated Complexity**: High (30+ tasks, GPU infrastructure)

---

### 6. Knowledge Graph & Retrieval System (`add-knowledge-graph-retrieval`)

**Purpose**: Graph storage and multi-strategy retrieval

**Key Deliverables**:

- Neo4j graph database with schema (Entity, Claim, Evidence, ExtractionActivity nodes)
- Cypher MERGE operations with idempotency
- SHACL validation for graph constraints
- Semantic chunking service (paragraph, section, table-aware)
- OpenSearch integration (BM25 + SPLADE)
- FAISS integration (dense vector similarity)
- Multi-strategy retrieval with fusion ranking (RRF)
- Reranking with cross-encoder
- Span highlighting

**Dependencies**: Foundation Infrastructure, GPU Microservices (for embeddings)

**Estimated Complexity**: Very High (35+ tasks)

---

### 7. Security & Authentication (`add-security-auth`)

**Purpose**: OAuth 2.0, multi-tenancy, and security hardening

**Key Deliverables**:

- OAuth 2.0 client credentials flow with JWT validation
- Scope definitions and enforcement (ingest:write, kg:read, etc.)
- Multi-tenant isolation (tenant_id in all queries)
- API key generation and rotation
- Rate limiting with token bucket algorithm
- Input validation and sanitization
- Secrets management (Vault or env vars)
- Audit logging for all mutations
- CORS, TLS/HTTPS enforcement

**Dependencies**: Multi-Protocol Gateway

**Estimated Complexity**: High (45+ tasks)

---

### 8. DevOps & Observability (`add-devops-observability`)

**Purpose**: Production deployment infrastructure and monitoring

**Key Deliverables**:

- CI/CD pipeline (GitHub Actions) with lint, test, build, deploy
- Contract tests (Schemathesis, GraphQL Inspector, Buf)
- Performance tests (k6 with P95 latency assertions)
- Prometheus metrics exposition
- OpenTelemetry distributed tracing (Jaeger)
- Grafana dashboards (system health, API latencies, GPU utilization)
- Structured logging with correlation IDs
- Error tracking (Sentry)
- Docker Compose for local development
- Kubernetes manifests (deployments, services, HPA, ingress)
- Documentation site (MkDocs) with API docs

**Dependencies**: All previous changes

**Estimated Complexity**: Very High (65+ tasks)

---

## Total Scope

- **9 Major Change Proposals**
- **16+ Capabilities** (foundation, rest-api, graphql-api, grpc-services, asyncapi-events, biomedical-adapters, ingestion-orchestration, gpu-microservices, knowledge-graph, retrieval-system, security-auth, devops-observability, domain-validation-caching)
- **462 Implementation Tasks** (added 73 for domain validation & caching)
- **10+ Biomedical Data Sources**
- **5 API Protocols**
- **3 GPU Services**
- **4 Storage Systems** (Neo4j, OpenSearch, FAISS, MinIO/S3)

## Technology Stack

- **Language**: Python 3.12 with strict typing
- **API Frameworks**: FastAPI, Strawberry GraphQL, gRPC
- **Data Processing**: Apache Kafka, Pydantic
- **ML/AI**: MinerU (PDF), SPLADE (sparse), Qwen-3 (dense), LLMs (extraction)
- **Storage**: Neo4j (graph), OpenSearch (search), FAISS (vectors), MinIO/S3 (objects)
- **Auth**: OAuth 2.0 with JWT
- **Monitoring**: Prometheus, OpenTelemetry, Grafana, Jaeger
- **Deployment**: Docker Compose, Kubernetes

## Standards Compliance

- HL7 FHIR (medical domain)
- OpenAPI 3.1 (REST)
- JSON:API v1.1 (response format)
- OData (query syntax)
- GraphQL (typed queries)
- Protocol Buffers (gRPC)
- AsyncAPI (events)
- RFC 7807 (Problem Details)
- OAuth 2.0 (authentication)

## Next Steps

1. **Review all proposals**: Read through each change proposal in `openspec/changes/`
2. **Prioritize**: Determine which changes to implement first (recommended order above)
3. **Seek approval**: Get stakeholder sign-off on each proposal before implementation
4. **Begin implementation**: Start with Change 1 (Foundation Infrastructure)
5. **Track progress**: Update tasks.md files as you complete work
6. **Validate continuously**: Run `openspec validate --strict` to ensure compliance

## Getting Help

- **OpenSpec Commands**: `openspec list`, `openspec show <change>`, `openspec validate <change>`
- **Documentation**: See `openspec/AGENTS.md` for AI agent instructions
- **Project Context**: See `openspec/project.md` for conventions and constraints

## References

- Engineering Blueprint: `1) docs/Engineering Blueprint_ Multi-Protocol API Gateway & Orchestration System.pdf`
- Biomedical APIs Addendum: `1) docs/Section A_ Public Biomedical APIs for Integration.pdf`
