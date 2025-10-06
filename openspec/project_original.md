# Project Context

## Purpose

Medical_KG_rev is a multi-protocol API gateway and orchestration system for biomedical knowledge integration. The system ingests, processes, and unifies data from clinical trials, research literature, drug databases, and medical ontologies into a federated knowledge graph with advanced retrieval capabilities.

## Tech Stack

- **Language**: Python 3.12
- **API Frameworks**: FastAPI (REST/GraphQL), gRPC (Protocol Buffers)
- **Data Processing**: Apache Kafka, Pydantic, MinerU (PDF parsing)
- **ML/AI**: SPLADE embeddings, Qwen-3 dense embeddings, LLM extraction
- **Storage**: Neo4j (graph), OpenSearch (search), FAISS (vector), MinIO/S3 (object)
- **Auth**: OAuth 2.0 with scope-based authorization
- **Monitoring**: Prometheus, OpenTelemetry, Grafana
- **Deployment**: Docker Compose, Kubernetes

## Project Conventions

### Code Style

- Line length: 100 characters (Black/Ruff)
- Type hints: Required (strict mypy)
- Imports: Sorted with isort via Ruff
- Naming: snake_case for functions/variables, PascalCase for classes
- Docstrings: Google style with type annotations

### Architecture Patterns

- **Adapter SDK Pattern**: Plug-in architecture for data source connectors
- **Two-Phase Pipeline**: Metadata fetch → content processing for heavy operations
- **Federated Data Model**: Core entities extended with domain-specific overlays (FHIR, XBRL)
- **Fail-Fast**: GPU operations fail immediately if resources unavailable (no CPU fallback)
- **Event-Driven**: Kafka topics for async orchestration, SSE for client updates
- **Multi-Protocol Façade**: Single backend exposed via REST, GraphQL, gRPC, SOAP, AsyncAPI

### Testing Strategy

- **Contract Tests**: Schemathesis (OpenAPI), GraphQL Inspector, Buf (gRPC)
- **Performance Tests**: k6 for load testing (P95 latency SLOs)
- **Integration Tests**: Docker Compose test environment
- **Coverage**: Minimum 80% for core business logic

### Git Workflow

- Main branch: `main` (production-ready)
- Feature branches: `feature/<change-id>` matching OpenSpec change IDs
- Commits: Conventional commits format
- PR required for all changes with CI validation

## Domain Context

### Biomedical Data Sources

- **Clinical Trials**: ClinicalTrials.gov API (NCT identifiers)
- **Literature**: OpenAlex, PubMed Central, Unpaywall, Crossref
- **Drug Data**: OpenFDA (labels, adverse events, devices), DailyMed (SPL)
- **Ontologies**: RxNorm, MeSH, SNOMED CT, ICD-11, LOINC, UCUM
- **Chemistry**: ChEMBL for drug/molecule data

### Standards Compliance

- **HL7 FHIR**: Medical domain alignment (Evidence, ResearchStudy resources)
- **JSON:API v1.1**: REST response formatting
- **OData**: Query filtering syntax
- **OpenAPI 3.1**: REST API specification
- **GraphQL**: Typed query language with introspection
- **AsyncAPI**: Event-driven API documentation
- **RFC 7807**: Problem Details for HTTP errors
- **OAuth 2.0**: Client credentials flow with scopes

### Key Concepts

- **IR (Intermediate Representation)**: Document/Block/Section unified format
- **Span-Grounded Extraction**: All extracted facts linked to source text spans
- **PICO**: Population, Intervention, Comparison, Outcome (clinical evidence structure)
- **Provenance**: Every node/edge in KG tracks source and extraction method

## Important Constraints

### Performance

- Retrieval queries: P95 < 500ms
- Embedding generation: GPU-only (fail-fast if unavailable)
- Large documents: Chunked for processing (max tokens per chunk)
- Rate limiting: Per-client/per-endpoint (429 Too Many Requests)

### Security

- Multi-tenant isolation: All data scoped by tenant_id
- Scope-based authorization: Fine-grained OAuth2 scopes (ingest:write, kg:read, etc.)
- No PII in logs: Structured logging with sensitive data scrubbed
- API keys: Secure storage (HashiCorp Vault or environment variables)

### Quality

- Validation: Fail closed on invalid data (reject rather than partial save)
- Idempotency: Repeated operations safe (same input → same result)
- Deterministic: No random behavior in extraction/mapping
- SHACL/Pydantic validation: Schema enforcement at all layers

### Data Retention

- Object storage: 365 days minimum
- Audit logs: Permanent retention for compliance
- Embeddings: Versioned (model updates don't orphan vectors)

## External Dependencies

### Public APIs (with rate limits)

- ClinicalTrials.gov API: v2, no key required, reasonable use
- OpenFDA: No key for <1000 req/day, keyed for higher
- OpenAlex: 100k requests/day (polite pool with email)
- Unpaywall: Rate limited, requires email in User-Agent
- Crossref: 50 req/sec with "plus" service
- Europe PMC: SOAP and REST APIs
- Semantic Scholar: API key required, rate limits apply
- CORE: API key required for PDF access
- ChEMBL: Rate limited, REST API
- WHO ICD-11 API: OAuth token required
- RxNorm/NLM APIs: Public, rate limits apply

### Infrastructure Services

- Neo4j: v5.x graph database
- OpenSearch: Full-text search and SPLADE indexing
- FAISS: Dense vector similarity search
- Apache Kafka: Message broker for orchestration
- MinIO or S3: Object storage for PDFs and large files
- Prometheus: Metrics collection
- Jaeger/Zipkin: Distributed tracing

### ML Models

- MinerU: GPU-based PDF parsing (layout analysis, OCR)
- SPLADE: Sparse learned embeddings (GPU)
- Qwen-3: Dense embeddings (GPU)
- LLM for extraction: Configurable (GPT, Claude, local models)
