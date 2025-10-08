# Project Context: Medical_KG_rev

## Purpose

Medical_KG_rev is an enterprise-grade, multi-protocol API gateway and orchestration system designed for comprehensive biomedical knowledge integration. The platform ingests, processes, normalizes, and unifies heterogeneous data from diverse sources—clinical trials, peer-reviewed literature, drug safety databases, regulatory filings, and standardized medical ontologies—into a federated knowledge graph with advanced multi-strategy retrieval.

### Core Value Proposition

**Problem**: Biomedical researchers, clinicians, and analysts face fragmented data across dozens of incompatible APIs, formats, and standards. Manual integration is time-consuming, error-prone, and doesn't scale.

**Solution**: A unified platform that:

- **Ingests** from 11+ biomedical sources automatically with resilient adapters
- **Transforms** raw data into a standardized Intermediate Representation (IR) with full provenance
- **Enriches** content using GPU-accelerated AI (PDF parsing, embeddings, extraction)
- **Unifies** knowledge in a graph database aligned with industry standards (FHIR, SNOMED, RxNorm)
- **Retrieves** information using multi-strategy search (BM25 + sparse + dense vectors)
- **Exposes** via 5 API protocols (REST, GraphQL, gRPC, SOAP, AsyncAPI) for maximum compatibility

### Target Users

- **Pharmaceutical researchers** analyzing clinical trial outcomes across studies
- **Medical informaticists** building clinical decision support systems
- **Regulatory affairs teams** tracking drug safety signals and adverse events
- **Healthcare data scientists** training ML models on comprehensive medical literature
- **Clinical trial coordinators** identifying patient cohorts and trial feasibility
- **Evidence synthesis teams** conducting systematic reviews and meta-analyses

### System Capabilities (High Level)

1. **Multi-Source Ingestion**: Automated data collection from clinical trials (ClinicalTrials.gov), literature (OpenAlex, PMC), drug databases (OpenFDA), and ontologies (RxNorm, ICD-11)

2. **GPU-Accelerated Processing**: PDF parsing (MinerU), embedding generation (SPLADE + Qwen-3), and LLM-based extraction for structured data

3. **Knowledge Graph Construction**: Neo4j graph with provenance tracking, SHACL validation, and FHIR-aligned schema

4. **Advanced Retrieval**: Multi-strategy search combining lexical (BM25), learned sparse (SPLADE), and dense semantic vectors with fusion ranking

5. **Multi-Protocol API**: Single backend exposed via REST (OpenAPI + JSON:API + OData), GraphQL, gRPC, SOAP, and AsyncAPI/SSE

6. **Enterprise Security**: OAuth 2.0 authentication, multi-tenant isolation, scope-based authorization, rate limiting, audit logging

7. **Production Observability**: Prometheus metrics, OpenTelemetry distributed tracing, Grafana dashboards, structured logging, Sentry error tracking

8. **Automated CI/CD**: Contract tests (Schemathesis, GraphQL Inspector, Buf), performance tests (k6), Docker/Kubernetes deployment

## Tech Stack

### Core Technologies

- **Language**: Python 3.12 with strict type hints (mypy)
- **API Framework**: FastAPI (REST/SSE), Strawberry GraphQL, gRPC (Protocol Buffers)
- **Data Processing**: Apache Kafka, Pydantic v2 (validation), MinerU (PDF)
- **ML/AI**: PyTorch, Transformers, SPLADE, Qwen-3, Sentence Transformers
- **Storage**: Neo4j 5.x (graph), OpenSearch (search), FAISS (vectors), MinIO/S3 (objects), Redis (cache)
- **Auth**: OAuth 2.0 with JWT (python-jose)
- **Monitoring**: Prometheus, OpenTelemetry, Grafana, Jaeger, Sentry, structlog
- **Deployment**: Docker Compose, Kubernetes
- **Testing**: pytest, Schemathesis, k6, GraphQL Inspector, Buf

### Development Tools

- **Code Quality**: Black, Ruff, mypy, pre-commit
- **API Specs**: OpenAPI 3.1, GraphQL SDL, Protocol Buffers (.proto)
- **Documentation**: MkDocs Material
- **Version Control**: Git with conventional commits
- **CI/CD**: GitHub Actions

### External Libraries

- **HTTP**: httpx, aiohttp, tenacity (retry)
- **Biomedical**: pyalex, biopython
- **Data**: pandas, numpy, pyyaml
- **CLI**: click, tqdm
- **Security**: cryptography, passlib, python-dotenv, hvac (Vault)
- **Validation**: pint (UCUM), jsonschema, pyshacl, rdflib
- **Tokenization**: tiktoken

## Project Conventions

### Code Style

- Line length: 100 characters (Black/Ruff)
- Type hints: Required (strict mypy enforcement)
- Imports: Sorted with isort via Ruff (relative imports allowed within package)
- Naming: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- Docstrings: Google style with type annotations
- Async/await: Preferred for I/O operations

### Architecture Patterns

#### Multi-Protocol Façade

- Single backend exposed through 5 protocols: REST, GraphQL, gRPC, SOAP, AsyncAPI/SSE
- Protocol handlers share common service layer - no duplicate business logic
- Always implement protocol-agnostic logic first, then add protocol-specific wrappers

#### Adapter SDK Pattern

- Data sources plug in via BaseAdapter interface: fetch() → parse() → validate() → write()
- Simple REST APIs: Define in YAML (Singer/Airbyte-inspired)
- Complex sources: Implement Python adapter class
- Each adapter manages its own rate limits and retry logic

#### Two-Phase Pipeline

- **Auto-pipeline**: Fast sources (metadata → chunk → embed → index) in one pass
- **Manual pipeline**: GPU-bound operations (metadata → fetch PDF → MinerU → postpdf → chunk → embed → index)
- Ledger tracks document processing stage for idempotency and fault tolerance

#### Federated Data Model

- **Core entities**: Document, Block, Section, Entity, Claim, Organization
- **Domain overlays**: Medical (FHIR-aligned), Financial (XBRL), Legal (LegalDocML)
- Use discriminated unions for domain-specific extensions
- All models use Pydantic v2 with strict validation

#### Fail-Fast Philosophy

- GPU operations: Fail immediately if GPU unavailable (no CPU fallback)
- Validation: Reject at entry points (don't propagate bad data)
- External APIs: Validate IDs before making requests
- Contracts: Schemathesis/GraphQL Inspector/Buf prevent spec drift

#### Event-Driven Orchestration

- Kafka topics for async job processing (ingest.requests.v1, ingest.results.v1, mapping.events.v1)
- SSE for client-side real-time updates
- Worker processes consume messages and execute pipelines
- Dead letter queue for failed jobs with retry logic

### Testing Strategy

#### Contract Tests (Required - Block PRs)

- **REST**: Schemathesis generates tests from OpenAPI spec
- **GraphQL**: GraphQL Inspector detects breaking changes
- **gRPC**: Buf breaking change detection on .proto files
- Run in CI on every PR - failing contract tests block merge

#### Performance Tests (Required - Nightly)

- k6 scripts with thresholds (P95 < 500ms for retrieval)
- Test concurrent job processing (5+ simultaneous ingests)
- GPU service load testing (batch sizes, memory limits)
- Run nightly or on release branches

#### Integration Tests (Required - CI)

- Docker Compose test environment with all services
- Test multi-adapter chaining (OpenAlex → Unpaywall → MinerU)
- Test two-phase pipeline end-to-end
- Test multi-tenant isolation

#### Unit Tests (Required - 80%+ Coverage)

- pytest with pytest-cov
- Mock external APIs with pytest-mock
- Async tests with pytest-asyncio
- Parametrized tests for edge cases

### Git Workflow

- Main branch: `main` (production-ready, protected)
- Feature branches: `feature/<change-id>` matching OpenSpec change IDs
- Commits: Conventional commits format (feat:, fix:, docs:, test:, refactor:)
- PR required for all changes with CI validation
- Squash merge to keep history clean

## Domain Context

### Biomedical Data Sources (11+ Adapters)

#### Clinical Research

- **ClinicalTrials.gov API v2**: 450k+ interventional/observational studies
  - Fields: phase, status, interventions, eligibility, outcomes, sponsors, locations
  - Real-time updates, no API key required

#### Research Literature (Open Access - 6 sources)

- **OpenAlex**: 250M+ scholarly works with citations, authors, institutions
- **PubMed Central**: 8M+ full-text articles (XML format via Europe PMC)
- **Unpaywall**: 40M+ OA articles with legal full-text links (gold, green, hybrid, bronze)
- **Crossref**: 140M+ DOI metadata with citations
- **CORE**: 200M+ OA research papers with PDF access
- **Semantic Scholar**: Citation counts, references, influential papers

#### Drug Safety & Regulatory

- **OpenFDA Drug Labels**: FDA-approved SPL documents (indications, contraindications, dosing)
- **OpenFDA Adverse Events**: FAERS post-market drug safety reports
- **OpenFDA Medical Devices**: Device registrations, recalls, adverse events
- **DailyMed**: Current drug labeling with NDC codes (daily updates)

#### Medical Ontologies & Standards

- **RxNorm**: ~200k drug names normalized to RxCUI codes (NDC, SNOMED mapping)
- **ICD-11 (WHO)**: 55k+ disease classification codes with OAuth API
- **MeSH (NLM)**: 30k+ controlled vocabulary with tree structures
- **SNOMED CT**: 350k+ clinical terminology concepts
- **LOINC**: Laboratory and observation codes
- **UCUM**: Units of measure for medical quantities

#### Chemistry & Pharmacology

- **ChEMBL**: 2.3M+ compounds, 20M+ bioactivity measurements with drug targets

### Standards Compliance & Interoperability

#### Medical Standards

- **HL7 FHIR R5**: Fast Healthcare Interoperability Resources
  - Evidence, ResearchStudy, MedicationStatement, Observation resources
  - CodeableConcept, Identifier, Reference patterns
- **SNOMED CT**: Clinical terminology for diagnoses, procedures, findings
- **RxNorm**: Drug normalization and ingredient-level mapping
- **ICD-11**: WHO disease classification codes
- **LOINC**: Laboratory test identifiers
- **UCUM**: Unified Code for Units of Measure

#### API & Protocol Standards

- **OpenAPI 3.1**: REST API specification with JSON Schema
- **JSON:API v1.1**: Standardized REST response format (data, attributes, relationships, included)
- **OData v4**: Query syntax ($filter, $select, $expand, $top, $skip, $orderby)
- **GraphQL**: Strongly-typed query language with introspection
- **gRPC & Protocol Buffers**: High-performance RPC with HTTP/2
- **AsyncAPI 3.0**: Event-driven API specification with channels and messages
- **SOAP 1.2**: Legacy integration with WSDL

#### Security Standards

- **OAuth 2.0**: Client credentials flow, JWT (RS256), scope-based authorization
- **RFC 7807**: Problem Details for HTTP APIs with structured errors
- **TLS 1.3**: Transport encryption
- **CORS**: Cross-Origin Resource Sharing

#### Data Quality Standards

- **SHACL**: Shapes Constraint Language for RDF graph validation
- **JSON Schema**: Request/response validation
- **Pydantic**: Runtime data validation with type hints

### Key Concepts & Terminology

#### Data Model

- **IR (Intermediate Representation)**: Document/Block/Section unified format
- **Federated Model**: Core entities + domain overlays (medical, finance, legal)
- **Document**: Top-level container (trial, paper, drug label) with metadata
- **Block**: Semantic chunk (paragraph, section, table) with type
- **Span**: Text coordinate (start, end) for grounding extractions

#### Knowledge Graph

- **Entity**: Identified real-world object normalized to codes (RxCUI, ICD-11, SNOMED)
- **Claim**: Extracted statement (subject-predicate-object triples) with confidence
- **Evidence**: Text spans supporting claims with quality assessment
- **ExtractionActivity**: Provenance (model, version, prompt, timestamp)
- **Relationship**: Typed edges (TREATS, CAUSES, INTERACTS_WITH, etc.)

#### Clinical Informatics

- **PICO**: Population, Intervention, Comparison, Outcome (evidence structure)
- **Eligibility Criteria**: Inclusion/exclusion for clinical trials
- **Adverse Events**: Unwanted drug effects with severity grading
- **Dosing Regimen**: Dose, route, frequency, duration (UCUM units)

#### Information Extraction

- **Span-Grounded Extraction**: Facts link to exact source text (character offsets)
- **Named Entity Recognition (NER)**: Identifying mentions (drugs, diseases, genes)
- **Entity Linking (EL)**: Mapping mentions to canonical IDs
- **Relation Extraction**: Identifying relationships with confidence scores

#### Retrieval

- **Multi-Strategy**: BM25 (sparse lexical) + SPLADE (learned sparse) + Dense (semantic)
- **Fusion Ranking**: Reciprocal Rank Fusion (RRF) combines results
- **Reranking**: Cross-encoder for top-k results
- **Chunking**: Splitting documents into retrievable units (paragraph, section, sliding window)

#### Orchestration

- **Two-Phase Pipeline**: Metadata → Content processing pattern
- **Adapter SDK**: Plug-in architecture (fetch → parse → validate → write)
- **Idempotency**: Same input → same output, safe retries
- **Fail-Fast**: Reject invalid inputs immediately

#### Provenance & Trust

- **Provenance Tracking**: Complete lineage (source, timestamp, processing method)
- **Versioning**: Document, model, and ontology versions over time
- **Audit Trail**: Immutable log (user, action, resource, timestamp)

## Important Constraints

### Performance

- Retrieval queries: **P95 < 500ms** (SLO)
- Embedding generation: **GPU-only** (fail-fast if unavailable)
- Large documents: Chunked (max tokens per chunk based on model)
- Rate limiting: Per-client/per-endpoint with 429 responses
- Database queries: Indexed, optimized Cypher (Neo4j), avoid full scans

### Security

- **Multi-tenant isolation**: All data scoped by tenant_id from JWT
- **Scope-based authorization**: Fine-grained OAuth2 scopes (ingest:write, kg:read, retrieve:read, etc.)
- **No PII in logs**: Structured logging with sensitive data scrubbing
- **Secrets management**: HashiCorp Vault or secure environment variables
- **TLS/HTTPS**: Enforced in production
- **Input validation**: At all entry points (API, gRPC, Kafka)
- **SQL injection prevention**: Parameterized queries, ORM usage
- **XSS prevention**: Output encoding, CSP headers

### Quality

- **Validation**: Fail closed on invalid data (reject rather than partial save)
- **Idempotency**: Repeated operations safe (MERGE in Neo4j, unique doc_ids)
- **Deterministic**: No random behavior in extraction/mapping (fixed seeds if needed)
- **SHACL/Pydantic validation**: Schema enforcement at all layers
- **Span validation**: All spans within document bounds, start < end
- **Provenance**: Mandatory for all extracted data

### Data Retention

- **Object storage**: 365 days minimum (PDFs, large files)
- **Audit logs**: Permanent retention for compliance (HIPAA, GDPR)
- **Embeddings**: Versioned (model updates don't orphan vectors)
- **Graph data**: Soft deletes (mark as inactive, don't physically delete)
- **Backups**: Daily automated backups with 30-day retention

### Scalability

- **Horizontal scaling**: Stateless gateway (scale with load balancer)
- **Kafka partitioning**: Distribute job processing across workers
- **Neo4j clustering**: High availability and read replicas
- **OpenSearch sharding**: Distribute index across nodes
- **FAISS indexing**: Support billions of vectors with HNSW

## External Dependencies

### Public APIs (with rate limits)

- **ClinicalTrials.gov API**: v2, no key, reasonable use (~10 req/sec)
- **OpenFDA**: No key for <1000 req/day, keyed for higher limits
- **OpenAlex**: 100k req/day (polite pool with email in User-Agent)
- **Unpaywall**: Rate limited, requires email in User-Agent
- **Crossref**: 50 req/sec with "plus" service
- **Europe PMC**: SOAP and REST APIs, reasonable use
- **Semantic Scholar**: API key required, rate limits apply
- **CORE**: API key required for PDF access
- **ChEMBL**: Rate limited REST API
- **WHO ICD-11 API**: OAuth token required
- **RxNorm/NLM APIs**: Public, rate limits apply

### Infrastructure Services

- **Neo4j**: v5.x graph database (Community or Enterprise)
- **OpenSearch**: Full-text search and SPLADE indexing
- **FAISS**: Dense vector similarity search (CPU or GPU)
- **Apache Kafka**: Message broker with Zookeeper
- **MinIO or S3**: Object storage for PDFs and large files
- **Redis**: Cache and rate limiting (Hiredis for performance)
- **PostgreSQL** (optional): Job ledger and audit logs
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Jaeger or Zipkin**: Distributed tracing
- **Sentry** (optional): Error tracking

### ML Models

- **MinerU**: GPU-based PDF parsing (layout analysis, OCR)
- **SPLADE**: Sparse learned embeddings (GPU, ~1GB VRAM)
- **Qwen-3**: Dense embeddings (GPU, ~3GB VRAM for 1.5B model)
- **LLM for extraction**: Configurable (GPT-4, Claude, Llama, Mixtral)
- **Cross-encoder reranker**: ms-marco-MiniLM or similar

### Development Dependencies

- **Docker**: Containerization (20.10+)
- **Docker Compose**: Local development stack (v2.x)
- **Kubernetes** (optional): Production deployment (1.28+)
- **Buf**: Protocol Buffer management (1.28+)
- **k6**: Performance testing (0.48+)
- **pre-commit**: Git hooks (3.6+)
- **act** (optional): Test GitHub Actions locally

## Project Structure

```
Medical_KG_rev/
├── src/Medical_KG_rev/       # Main package
│   ├── models/               # Pydantic data models
│   │   ├── ir.py            # Intermediate Representation
│   │   ├── entities.py      # Entity, Claim, Evidence
│   │   ├── organization.py  # Organization, Tenant
│   │   ├── provenance.py    # ExtractionActivity
│   │   └── overlays/        # Domain-specific extensions
│   ├── adapters/            # Data source adapters
│   │   ├── base.py          # BaseAdapter abstract class
│   │   ├── biomedical.py    # 11+ biomedical adapters
│   │   ├── registry.py      # Adapter discovery
│   │   ├── yaml_parser.py   # YAML config parser
│   │   └── config/          # YAML adapter configs
│   ├── gateway/             # API Gateway
│   │   ├── app.py           # FastAPI application
│   │   ├── rest/            # REST endpoints
│   │   ├── graphql/         # Strawberry GraphQL schema
│   │   ├── grpc/            # gRPC server
│   │   ├── soap/            # SOAP adapter
│   │   ├── sse/             # Server-Sent Events
│   │   ├── services.py      # Protocol-agnostic service layer
│   │   ├── models.py        # Request/response models
│   │   └── middleware.py    # Logging, correlation IDs
│   ├── services/            # gRPC microservices
│   │   ├── mineru/          # PDF parsing service
│   │   ├── embedding/       # Embedding generation
│   │   ├── extraction/      # LLM extraction
│   │   ├── retrieval/       # Multi-strategy retrieval
│   │   ├── gpu/             # GPU manager
│   │   └── grpc/            # gRPC server utilities
│   ├── orchestration/       # Job orchestration
│   │   ├── kafka.py         # Kafka producer/consumer
│   │   ├── ledger.py        # State tracking
│   │   ├── orchestrator.py  # Pipeline definitions
│   │   └── worker.py        # Background workers
│   ├── kg/                  # Knowledge Graph
│   │   ├── neo4j_client.py  # Neo4j driver wrapper
│   │   ├── cypher_templates.py # Query templates
│   │   ├── shacl.py         # SHACL validation
│   │   ├── schema.py        # Graph schema
│   │   └── shapes.ttl       # SHACL shapes
│   ├── storage/             # Storage abstractions
│   │   ├── object_store.py  # MinIO/S3
│   │   ├── cache.py         # Redis
│   │   ├── ledger.py        # Job ledger
│   │   └── base.py          # Base interfaces
│   ├── auth/                # OAuth & JWT
│   │   ├── jwt.py           # JWT validation
│   │   ├── api_keys.py      # API key management
│   │   ├── rate_limit.py    # Rate limiting
│   │   ├── audit.py         # Audit logging
│   │   ├── scopes.py        # Scope definitions
│   │   └── dependencies.py  # FastAPI dependencies
│   ├── validation/          # Domain validation
│   │   ├── ucum.py          # UCUM unit validation
│   │   └── fhir.py          # FHIR validation
│   ├── observability/       # Monitoring
│   │   ├── metrics.py       # Prometheus metrics
│   │   ├── tracing.py       # OpenTelemetry
│   │   └── sentry.py        # Error tracking
│   ├── config/              # Configuration
│   │   ├── settings.py      # Pydantic settings
│   │   └── domains.py       # Domain configs
│   ├── utils/               # Shared utilities
│   │   ├── http_client.py   # HTTP client with retry
│   │   ├── logging.py       # Structured logging
│   │   ├── identifiers.py   # ID generation
│   │   ├── errors.py        # Error classes
│   │   ├── validation.py    # Validation utilities
│   │   ├── spans.py         # Span utilities
│   │   ├── time.py          # Time utilities
│   │   ├── versioning.py    # Version management
│   │   └── metadata.py      # Metadata utilities
│   ├── proto/               # gRPC Protocol Buffers
│   │   ├── ingestion.proto
│   │   ├── embedding.proto
│   │   ├── extraction.proto
│   │   └── mineru.proto
│   └── scripts/             # Utility scripts (reserved for future helpers)
├── tests/                   # Test suites
│   ├── unit/
│   ├── integration/
│   ├── contract/
│   └── performance/
├── docs/                    # Documentation
│   ├── index.md
│   ├── architecture/
│   ├── guides/
│   ├── devops/
│   ├── openapi.yaml
│   ├── schema.graphql
│   └── asyncapi.yaml
├── openspec/                # OpenSpec proposals
│   ├── AGENTS.md
│   ├── project.md
│   ├── changes/             # 9 change proposals
│   └── specs/               # Capability specs (after archiving)
├── ops/                     # Deployment
│   ├── docker-compose.yml
│   ├── Dockerfile.gpu
│   ├── k8s/                 # Kubernetes manifests
│   └── monitoring/          # Grafana dashboards, alerts
├── scripts/                 # Utility scripts
│   ├── init.sh
│   ├── generate_api_docs.py
│   ├── update_graphql_schema.py
│   └── run_buf_checks.sh
├── pyproject.toml           # Project metadata & dependencies
├── buf.yaml                 # Buf configuration
├── buf.gen.yaml             # Buf generation config
├── docker-compose.yml       # Local development
├── Dockerfile               # Gateway container
├── mkdocs.yml               # Documentation config
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- GPU (optional, for ML services)
- API keys for external services (see `.env.example`)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/Medical_KG_rev.git
cd Medical_KG_rev

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Running Locally

```bash
# Start infrastructure services
docker-compose up -d neo4j opensearch kafka redis

# Run database migrations (if applicable)
python -m Medical_KG_rev.kg.migrate

# Start API gateway
python -m Medical_KG_rev.gateway.main

# Start Dagster services (webserver + daemon)
dagster dev -m Medical_KG_rev.orchestration.dagster.runtime

# API available at http://localhost:8000
# Swagger UI: http://localhost:8000/docs/openapi
# GraphQL Playground: http://localhost:8000/docs/graphql
```

### Running Tests

```bash
# Unit tests
pytest tests/unit

# Integration tests (requires Docker)
pytest tests/integration

# Contract tests
pytest tests/contract

# Performance tests (requires k6)
k6 run tests/performance/retrieval_test.js

# All tests with coverage
pytest --cov
```

## Documentation

- **API Documentation**: <http://localhost:8000/docs>
- **Architecture**: `1) docs/System Architecture & Design Rationale.md`
- **Implementation Roadmap**: `IMPLEMENTATION_ROADMAP.md`
- **OpenSpec Proposals**: `openspec/changes/`
- **Engineering Blueprint**: `1) docs/Engineering Blueprint_.pdf`
- **Biomedical APIs**: `1) docs/Section A_ Public Biomedical APIs.pdf`

## Support

- **Issues**: <https://github.com/your-org/Medical_KG_rev/issues>
- **Discussions**: <https://github.com/your-org/Medical_KG_rev/discussions>
- **Documentation**: <https://your-org.github.io/Medical_KG_rev>

## License

MIT License - see LICENSE file for details
