# Medical_KG_rev

**Multi-Protocol API Gateway & Orchestration System for Biomedical Knowledge Integration**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Overview

Medical_KG_rev is a production-ready, enterprise-grade platform that unifies fragmented biomedical data from 10+ diverse sources into a coherent knowledge graph with advanced retrieval capabilities. The system addresses the critical challenge faced by healthcare researchers, pharmaceutical companies, and medical informaticists: **data fragmentation across incompatible APIs, formats, and standards**.

### Key Features

- 🔌 **Multi-Protocol API**: Single backend accessible via REST (OpenAPI/JSON:API/OData), GraphQL, gRPC, SOAP, and AsyncAPI/SSE
- 📊 **Federated Data Model**: Unified Intermediate Representation with domain-specific overlays (medical/FHIR, financial/XBRL, legal/LegalDocML)
- 🔌 **Plug-in Adapters**: YAML-based connector SDK for 11+ biomedical data sources
- 🚀 **GPU-Accelerated AI**: PDF parsing (MinerU), embeddings (SPLADE + Qwen-3), and LLM extraction
- 🔍 **Multi-Strategy Retrieval**: Hybrid search combining BM25, SPLADE, and dense vectors with fusion ranking
- 🔐 **Enterprise Security**: OAuth 2.0 with JWT, multi-tenant isolation, scope-based authorization, rate limiting
- 📈 **Production Observability**: Prometheus metrics, OpenTelemetry tracing, Grafana dashboards, Sentry error tracking
- ✅ **Standards Compliance**: HL7 FHIR, OpenAPI 3.1, JSON:API, GraphQL, gRPC, AsyncAPI, RFC 7807

### Architecture Highlights

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIENT APPLICATIONS                           │
│  Web Apps │ Mobile │ Desktop │ Legacy Systems │ ML Pipelines     │
└──────┬─────────┬─────────┬─────────┬──────────┬─────────────────┘
       │ REST    │ GraphQL │ gRPC    │ SOAP     │ AsyncAPI/SSE
┌──────▼─────────▼─────────▼─────────▼──────────▼─────────────────┐
│                 MULTI-PROTOCOL API GATEWAY                       │
│  FastAPI │ Strawberry GraphQL │ gRPC Services │ SOAP │ SSE      │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│         ORCHESTRATION & EVENT BUS (Kafka + Job Ledger)          │
└──────┬──────────────┬────────────────┬─────────────────┬────────┘
       │              │                │                 │
┌──────▼────────┐  ┌──▼─────────┐  ┌──▼─────────┐  ┌───▼─────────┐
│  BIOMEDICAL   │  │   GPU      │  │  STORAGE   │  │ RETRIEVAL   │
│   ADAPTERS    │  │ SERVICES   │  │  LAYER     │  │  ENGINES    │
│ 11+ Sources   │  │ MinerU     │  │ Neo4j      │  │ OpenSearch  │
│               │  │ Embeddings │  │ MinIO/S3   │  │ FAISS       │
└───────────────┘  └────────────┘  └────────────┘  └─────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose (for infrastructure services)
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
# Edit .env with your API keys and configuration
```

### Running Locally

```bash
# Start infrastructure services (Neo4j, OpenSearch, Kafka, Redis)
docker-compose up -d

# Run database migrations (if applicable)
python -m Medical_KG_rev.kg.migrate

# Start API gateway
python -m Medical_KG_rev.gateway.main

# Start background workers (in separate terminal)
python -m Medical_KG_rev.orchestration.workers

# API available at:
# - REST: http://localhost:8000
# - Swagger UI: http://localhost:8000/docs/openapi
# - GraphQL Playground: http://localhost:8000/docs/graphql
# - gRPC: localhost:50051
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

---

## Data Sources

Medical_KG_rev integrates with 11+ biomedical data sources:

### Clinical Research

- **ClinicalTrials.gov API v2**: 450k+ interventional/observational studies

### Research Literature (6 sources)

- **OpenAlex**: 250M+ scholarly works with citations
- **PubMed Central**: 8M+ full-text articles
- **Unpaywall**: 40M+ open access articles
- **Crossref**: 140M+ DOI metadata
- **CORE**: 200M+ OA research papers
- **Semantic Scholar**: Citation analysis

### Drug Safety & Regulatory

- **OpenFDA Drug Labels**: FDA-approved SPL documents
- **OpenFDA Adverse Events**: FAERS post-market reports
- **OpenFDA Medical Devices**: Device registrations and recalls

### Medical Ontologies

- **RxNorm**: ~200k drug names normalized to RxCUI codes
- **ICD-11 (WHO)**: 55k+ disease classification codes
- **ChEMBL**: 2.3M+ compounds with bioactivity data

---

## API Protocols

### REST API (OpenAPI 3.1 + JSON:API + OData)

```bash
# Ingest clinical trials
curl -X POST http://localhost:8000/v1/ingest/clinicaltrials \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/vnd.api+json" \
  -d '{"data": {"type": "ingestion", "attributes": {"nct_ids": ["NCT04267848"]}}}'

# Retrieve with OData filters
curl "http://localhost:8000/v1/retrieve?query=pembrolizumab&\$filter=year gt 2020&\$top=10"
```

### GraphQL API

```graphql
query SearchTrials {
  search(
    query: "pembrolizumab melanoma"
    filters: { documentType: CLINICAL_TRIAL, status: RECRUITING }
    limit: 10
  ) {
    document {
      title
      nctId
    }
    score
    highlights
  }
}
```

### gRPC API

```python
import grpc
from Medical_KG_rev.proto.gen import ingestion_pb2, ingestion_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = ingestion_pb2_grpc.IngestionServiceStub(channel)

request = ingestion_pb2.IngestionJobRequest(
    tenant_id="tenant-123",
    dataset="clinicaltrials",
    item_ids=["NCT04267848"]
)
response = stub.Submit(request)
```

### Server-Sent Events (SSE)

```javascript
const eventSource = new EventSource('/v1/jobs/job-123/events');
eventSource.addEventListener('jobs.progress', (e) => {
  const data = JSON.parse(e.data);
  console.log(`Progress: ${data.progress}%`);
});
```

---

## Technology Stack

- **Language**: Python 3.12 with strict type hints
- **API Frameworks**: FastAPI, Strawberry GraphQL, gRPC
- **Data Processing**: Apache Kafka, Pydantic v2
- **ML/AI**: PyTorch, Transformers, SPLADE, Qwen-3, MinerU
- **Storage**: Neo4j 5.x (graph), OpenSearch (search), FAISS (vectors), MinIO/S3 (objects), Redis (cache)
- **Auth**: OAuth 2.0 with JWT (python-jose)
- **Monitoring**: Prometheus, OpenTelemetry, Grafana, Jaeger, Sentry
- **Deployment**: Docker Compose, Kubernetes
- **Testing**: pytest, Schemathesis, k6, GraphQL Inspector, Buf

---

## Project Structure

```
Medical_KG_rev/
├── src/Medical_KG_rev/       # Main package
│   ├── adapters/             # Data source adapters (11+ sources)
│   ├── auth/                 # OAuth 2.0, JWT, rate limiting
│   ├── config/               # Configuration management
│   ├── gateway/              # Multi-protocol API gateway
│   │   ├── rest/             # FastAPI REST endpoints
│   │   ├── graphql/          # Strawberry GraphQL schema
│   │   ├── grpc/             # gRPC server
│   │   ├── soap/             # SOAP adapter
│   │   └── sse/              # Server-Sent Events
│   ├── kg/                   # Knowledge Graph (Neo4j)
│   ├── models/               # Pydantic data models
│   │   ├── ir.py             # Intermediate Representation
│   │   ├── entities.py       # Entity, Claim, Evidence
│   │   └── overlays/         # Domain-specific extensions
│   ├── observability/        # Metrics, tracing, logging
│   ├── orchestration/        # Kafka, job ledger, workers
│   ├── proto/                # gRPC Protocol Buffers
│   ├── services/             # GPU microservices
│   │   ├── mineru/           # PDF parsing
│   │   ├── embedding/        # SPLADE + Qwen-3
│   │   └── extraction/       # LLM extraction
│   ├── storage/              # Object store, cache, ledger
│   ├── utils/                # Shared utilities
│   └── validation/           # UCUM, FHIR validation
├── tests/                    # Test suites
│   ├── unit/
│   ├── integration/
│   ├── contract/
│   └── performance/
├── docs/                     # Documentation (MkDocs)
├── openspec/                 # OpenSpec proposals
├── ops/                      # Deployment configs
│   ├── docker-compose.yml
│   ├── k8s/                  # Kubernetes manifests
│   └── monitoring/           # Grafana dashboards, alerts
└── pyproject.toml            # Project metadata & dependencies
```

---

## Documentation

- **API Documentation**: <http://localhost:8000/docs>
- **Architecture**: [`1) docs/System Architecture & Design Rationale.md`](1)%20docs/System%20Architecture%20&%20Design%20Rationale.md)
- **Implementation Roadmap**: [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)
- **OpenSpec Proposals**: [openspec/changes/](openspec/changes/)
- **Engineering Blueprint**: [`1) docs/Engineering Blueprint_.pdf`](1)%20docs/Engineering%20Blueprint_%20Multi-Protocol%20API%20Gateway%20&%20Orchestration%20System.pdf)
- **Biomedical APIs**: [`1) docs/Section A_ Public Biomedical APIs.pdf`](1)%20docs/Section%20A_%20Public%20Biomedical%20APIs%20for%20Integration.pdf)

---

## Development

### Code Quality

```bash
# Format code
black .

# Lint
ruff check .

# Type checking
mypy src

# Run all quality checks
pre-commit run --all-files
```

### Adding New Data Sources

1. Check if RESTAdapter + YAML config sufficient
2. If complex (SOAP, PDF, special auth): Implement Python adapter
3. Add to adapter registry with source name
4. Define rate limits in config
5. Add comprehensive tests with mocked responses
6. Update OpenAPI endpoints if user-facing

See [guides/adapter-sdk.md](docs/guides/adapter-sdk.md) for details.

### Performance Requirements

- **Retrieval queries**: P95 < 500ms (SLO)
- **Ingestion throughput**: 100+ documents/second
- **Concurrent users**: 1000+ simultaneous API clients
- **GPU services**: Fail-fast if GPU unavailable (no CPU fallback)

---

## Deployment

### Docker Compose (Development)

```bash
docker-compose up -d
```

### Kubernetes (Production)

```bash
kubectl apply -k ops/k8s/overlays/production
```

See [docs/devops/kubernetes.md](docs/devops/kubernetes.md) for details.

---

## Standards Compliance

- **HL7 FHIR R5**: Medical domain alignment (Evidence, ResearchStudy, MedicationStatement)
- **OpenAPI 3.1**: REST API specification
- **JSON:API v1.1**: REST response format
- **OData v4**: Query syntax ($filter, $select, $expand, $top, $skip, $orderby)
- **GraphQL**: Strongly-typed query language
- **gRPC & Protocol Buffers**: High-performance RPC
- **AsyncAPI 3.0**: Event-driven API documentation
- **OAuth 2.0**: Authentication & authorization
- **RFC 7807**: Problem Details for HTTP APIs
- **UCUM**: Units of measure for medical quantities

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Support

- **Issues**: <https://github.com/your-org/Medical_KG_rev/issues>
- **Discussions**: <https://github.com/your-org/Medical_KG_rev/discussions>
- **Documentation**: <https://your-org.github.io/Medical_KG_rev>

---

## Acknowledgments

Built with support from the biomedical research community and powered by:

- OpenAlex, PubMed Central, ClinicalTrials.gov, OpenFDA
- FastAPI, Strawberry GraphQL, Neo4j, OpenSearch
- PyTorch, Hugging Face Transformers
- And many other open-source projects

---

**Medical_KG_rev** - Unifying biomedical knowledge, one API at a time.
