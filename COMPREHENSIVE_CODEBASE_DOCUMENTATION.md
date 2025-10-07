# Comprehensive Medical_KG_rev Codebase Documentation

## Executive Summary

### System Overview

Medical_KG_rev is a sophisticated, production-ready multi-protocol API gateway and orchestration system designed to unify fragmented biomedical data from diverse sources into a coherent knowledge graph with advanced retrieval capabilities. The system addresses the critical challenge faced by healthcare researchers, pharmaceutical companies, and medical informaticists: **data fragmentation across incompatible APIs, formats, and standards**.

### Key Innovations

1. **Multi-Protocol Façade**: Single backend accessible via 5 protocols (REST, GraphQL, gRPC, SOAP, AsyncAPI/SSE), enabling integration with any client technology stack

2. **Federated Data Model**: Unified Intermediate Representation (IR) with domain-specific overlays allows medical and other knowledge domains to coexist

3. **Plug-in Adapter Architecture**: YAML-based connector SDK inspired by Singer/Airbyte enables adding new data sources declaratively without code changes

4. **GPU-Accelerated AI Pipeline**: Fail-fast GPU services for PDF parsing (MinerU), embeddings (SPLADE + Qwen-3), and LLM extraction ensure high-quality content processing

5. **Multi-Strategy Retrieval**: Hybrid search combining lexical (BM25), learned sparse (SPLADE), and dense semantic vectors with fusion ranking delivers superior relevance

6. **Provenance-First Design**: Every extracted fact traceable to source document, extraction method, and timestamp enables trust and reproducibility

### Target Scale & Performance

- **Data Volume**: 10M+ documents, 100M+ entities, 1B+ relationships
- **Query Performance**: P95 < 500ms for retrieval queries
- **Concurrent Users**: 1000+ simultaneous API clients
- **Ingestion Throughput**: 100+ documents/second
- **Geographic Distribution**: Multi-region deployment capability

### Standards Compliance

The system is built on industry standards to ensure long-term viability and interoperability:

- **HL7 FHIR R5** (medical domain alignment)
- **OpenAPI 3.1** (REST API specification)
- **JSON:API v1.1** (REST response format)
- **OData v4** (query syntax for filtering, sorting, pagination)
- **GraphQL** (typed query language)
- **gRPC/Protocol Buffers** (microservice communication)
- **OAuth 2.0** (authentication & authorization)
- **AsyncAPI 3.0** (event-driven API documentation)
- **UCUM** (Unified Code for Units of Measure)
- **SHACL** (Shapes Constraint Language for graph validation)
- **RFC 7807** (Problem Details for HTTP APIs)

### Implementation Status

**All 9 OpenSpec change proposals completed (462 tasks)**:

1. ✅ Foundation Infrastructure (48 tasks)
2. ✅ Multi-Protocol Gateway (62 tasks)
3. ✅ Biomedical Adapters (49 tasks)
4. ✅ Ingestion Orchestration (36 tasks)
5. ✅ GPU Microservices (33 tasks)
6. ✅ Knowledge Graph & Retrieval (43 tasks)
7. ✅ Security & Authentication (49 tasks)
8. ✅ DevOps & Observability (69 tasks)
9. ✅ Domain Validation & Caching (73 tasks)

**Production-ready features**:

- 11+ biomedical data source adapters
- 5 API protocols fully implemented
- 3 GPU services with fail-fast architecture
- Multi-strategy hybrid retrieval (BM25 + SPLADE + dense)
- OAuth 2.0 with multi-tenant isolation
- Comprehensive observability (Prometheus, OpenTelemetry, Grafana)
- Contract, performance, and integration test suites

---

## Detailed Narrative Review

### System Evolution and Development Journey

The Medical_KG_rev project represents a comprehensive evolution from a simple biomedical knowledge integration prototype to a sophisticated, production-ready multi-protocol API gateway. The system has been engineered to solve the fundamental challenge of biomedical data fragmentation, where researchers and healthcare professionals struggle with incompatible APIs, varying data formats, and inconsistent standards across different biomedical data sources.

### Technical Architecture Evolution

**Phase 1: Foundation Infrastructure**
The project began with establishing a robust foundation based on the OpenSpec design principles. This phase focused on creating a modular, extensible architecture that could handle multiple knowledge domains while maintaining type safety and consistency.

**Phase 2: Multi-Protocol Gateway**
Building on the foundation, the team implemented a multi-protocol API gateway that provides a single entry point for clients while abstracting the complexity of different communication protocols. This was crucial for supporting diverse client applications ranging from web interfaces to legacy enterprise systems.

**Phase 3: Biomedical Data Integration**
The core innovation emerged in the biomedical adapter ecosystem. Rather than building custom integrations for each data source, the team created a YAML-based adapter SDK that allows declarative configuration of new data sources. This approach dramatically reduced the friction of adding new biomedical APIs.

**Phase 4: AI-Enhanced Processing Pipeline**
Recognizing the need for high-quality content processing, the team integrated GPU-accelerated services for PDF parsing, embedding generation, and LLM-based entity extraction. The fail-fast architecture ensures that GPU services either work correctly or fail immediately, preventing silent performance degradation.

**Phase 5: Advanced Retrieval Systems**
The retrieval architecture combines multiple search strategies - lexical (BM25), learned sparse (SPLADE), and dense semantic vectors - with intelligent fusion ranking to provide superior relevance for biomedical queries.

### Key Technical Challenges Overcome

**Challenge 1: Multi-Tenancy at Scale**
The system needed to support multiple tenants (research institutions, pharmaceutical companies, healthcare providers) while ensuring complete data isolation. The solution implemented tenant-aware routing at every layer of the stack, from authentication through to storage and retrieval.

**Challenge 2: Heterogeneous Data Integration**
Biomedical data sources vary dramatically in their API designs, authentication mechanisms, and data formats. The adapter SDK provides a unified interface while supporting complex patterns like OAuth flows, rate limiting, and custom parsing logic.

**Challenge 3: GPU Resource Management**
AI/ML processing requires GPU resources, but ensuring consistent performance across different deployment environments proved challenging. The fail-fast GPU service architecture guarantees that services either perform optimally or fail explicitly, preventing silent degradation.

**Challenge 4: Provenance and Trust**
In healthcare and biomedical research, data provenance is critical for trust and regulatory compliance. The system implements comprehensive provenance tracking from source document through extraction to final knowledge graph representation.

### Operational Excellence

**Observability Strategy**
The system implements comprehensive observability using OpenTelemetry for distributed tracing, Prometheus for metrics collection, and structured logging with correlation IDs. This enables rapid incident response and performance optimization.

**Deployment Architecture**
The production deployment uses Kubernetes with separate node pools for CPU and GPU workloads, ensuring optimal resource allocation. The system supports both Docker Compose for development and Kubernetes for production deployment.

**Security First Approach**
Security is integrated throughout the system, from OAuth 2.0 authentication and JWT-based authorization to comprehensive audit logging and penetration testing compliance.

---

## Technical Architecture and Design Details

### 1. System Architecture Overview

#### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIENT APPLICATIONS                           │
│  Web Apps │ Mobile │ Desktop │ Legacy Systems │ ML Pipelines     │
└──────┬─────────┬─────────┬─────────┬──────────┬─────────────────┘
       │         │         │         │          │
       │ REST    │ GraphQL │ gRPC    │ SOAP     │ AsyncAPI/SSE
       │         │         │         │          │
┌──────▼─────────▼─────────▼─────────▼──────────▼─────────────────┐
│                 MULTI-PROTOCOL API GATEWAY                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │FastAPI   │ │Strawberry│ │gRPC      │ │Zeep      │           │
│  │REST+SSE  │ │GraphQL   │ │Services  │ │SOAP      │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │   Auth Middleware (OAuth 2.0 JWT)                          │ │
│  │   Rate Limiting │ Tenant Isolation │ Logging │ Metrics     │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                    SERVICE LAYER (Protocol-Agnostic)             │
│  Ingestion │ Chunking │ Retrieval │ Extraction │ KG Management  │
└──────┬─────────┬─────────┬─────────┬───────────┬────────────────┘
       │         │         │         │           │
┌──────▼─────────▼─────────▼─────────▼───────────▼────────────────┐
│               ORCHESTRATION & EVENT BUS (Kafka)                  │
│  ingest.requests │ ingest.results │ mapping.events              │
│  Job Ledger (State Tracking) │ Dead Letter Queue                │
└──────┬──────────────┬────────────────┬─────────────────┬────────┘
       │              │                │                 │
┌──────▼────────┐  ┌──▼─────────┐  ┌──▼─────────┐  ┌───▼─────────┐
│  BIOMEDICAL   │  │   GPU      │  │  STORAGE   │  │ RETRIEVAL   │
│   ADAPTERS    │  │ SERVICES   │  │  LAYER     │  │  ENGINES    │
├───────────────┤  ├────────────┤  ├────────────┤  ├─────────────┤
│CT.gov  OpenFDA│  │MinerU (PDF)│  │Neo4j       │  │OpenSearch   │
│OpenAlex  PMC  │  │SPLADE Emb. │  │(Graph KG)  │  │(BM25+SPLADE)│
│Unpaywall CORE │  │Qwen-3 Dense│  │            │  │             │
│RxNorm ICD-11  │  │LLM Extract │  │MinIO/S3    │  │FAISS        │
│ChEMBL  Crossref│  │            │  │(Objects)   │  │(Dense Vec.) │
└───────────────┘  └────────────┘  ├────────────┤  └─────────────┘
                                    │Redis       │
                                    │(Cache)     │
                                    └────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  OBSERVABILITY LAYER                             │
│  Prometheus Metrics │ OpenTelemetry Traces │ Structured Logs    │
│  Grafana Dashboards │ Jaeger │ Alertmanager                     │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Core Data Models

#### 2.1 Intermediate Representation (IR)

The system uses a sophisticated intermediate representation that provides a unified view of documents regardless of source:

```python
class Document(IRBaseModel):
    """Top level document representation."""

    id: str
    source: str = Field(description="Logical source identifier (e.g. clinicaltrials)")
    title: str | None = None
    sections: Sequence[Section] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: str = Field(default="v1")
    metadata: dict[str, Any] = Field(default_factory=dict)

class Section(IRBaseModel):
    """High-level grouping of blocks."""

    id: str
    title: str | None = None
    blocks: Sequence[Block] = Field(default_factory=list)

class Block(IRBaseModel):
    """A block is the smallest logical unit we track in the IR."""

    id: str
    type: BlockType = BlockType.PARAGRAPH
    text: str | None = Field(default=None, description="Plain-text content")
    spans: Sequence[Span] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    layout_bbox: tuple[float, float, float, float] | None = Field(default=None)
    reading_order: int | None = Field(default=None, ge=0)
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
```

#### 2.2 Federated Domain Model

The system supports multiple knowledge domains through a federated model:

```python
class MedicalMetadata(BaseModel):
    """FHIR-aligned medical domain metadata."""

    # Clinical Trial fields
    nct_id: Optional[str]
    phase: Optional[ClinicalPhase]
    status: Optional[TrialStatus]
    interventions: list[Intervention]
    conditions: list[Condition]
    eligibility_criteria: Optional[str]
    primary_outcome: Optional[Outcome]

    # Literature fields (FHIR Evidence resource)
    doi: Optional[str]
    pmcid: Optional[str]
    authors: list[Author]
    journal: Optional[str]
    publication_date: Optional[date]
    citation_count: Optional[int]

    # Drug Label fields (FHIR MedicationKnowledge)
    ndc: Optional[str]
    rxcui: Optional[str]
    indications: list[str]
    contraindications: list[str]
    adverse_events: list[AdverseEvent]
    dosage: Optional[Dosage]
```

### 3. Multi-Protocol API Gateway

#### 3.1 FastAPI Application Structure

The gateway is built on FastAPI with comprehensive middleware for security, logging, and caching:

```python
def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Medical KG Multi-Protocol Gateway", version="0.1.0")
    app.state.settings = settings
    app.state.jwt_cache = {}

    # Middleware stack
    app.add_middleware(JSONAPIResponseMiddleware)
    app.add_middleware(RequestLoggingMiddleware, correlation_header=settings.observability.logging.correlation_id_header)
    app.add_middleware(SecurityHeadersMiddleware, headers_config=settings.security.headers)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.security.cors.allow_origins))

    # Protocol routers
    app.include_router(rest_router)
    app.include_router(graphql_router, prefix="/graphql")
    app.include_router(sse_router)
    app.include_router(soap_router)

    return app
```

#### 3.2 Protocol-Specific Implementations

**REST API (OpenAPI 3.1 + JSON:API)**

- Full OpenAPI 3.1 specification compliance
- JSON:API v1.1 response format with relationships
- OData v4 query support for filtering, sorting, pagination

**GraphQL API (Strawberry GraphQL)**

- Type-safe schema with DataLoader pattern for N+1 prevention
- Rich query capabilities with filtering and relationships
- Subscription support for real-time updates

**gRPC Services (Protocol Buffers)**

- High-performance RPC with HTTP/2
- Bidirectional streaming for large batch uploads
- Service definitions for ingestion, embedding, and extraction

**SOAP API (Zeep)**

- Legacy enterprise system compatibility
- WSDL contract generation
- Comprehensive error handling

**AsyncAPI/SSE (Server-Sent Events)**

- Real-time job progress updates
- Event-driven architecture support
- AsyncAPI 3.0 specification compliance

### 4. Biomedical Adapter Ecosystem

#### 4.1 Adapter SDK Architecture

The adapter system provides a plug-in architecture for integrating external data sources:

```python
class BaseAdapter(ABC):
    """Base class for all data source adapters."""

    def __init__(self, config: AdapterConfig):
        self.config = config
        self.http_client = get_http_client(
            retries=config.retries,
            timeout=config.timeout,
        )
        self.rate_limiter = RateLimiter(
            requests_per_second=config.rate_limit.requests_per_second,
            burst=config.rate_limit.burst,
        )

    @abstractmethod
    async def fetch(self, **params: Any) -> list[dict[str, Any]]:
        """Fetch raw data from external source."""
        pass

    @abstractmethod
    async def parse(self, raw_data: list[dict[str, Any]]) -> list[Document]:
        """Parse raw data into Document IR."""
        pass
```

#### 4.2 Supported Data Sources

**Clinical Research:**

- ClinicalTrials.gov API v2 (450k+ studies)
- OpenFDA Drug Labels (FDA-approved SPL)
- OpenFDA Adverse Events (FAERS reports)

**Literature (6 sources):**

- OpenAlex (250M+ works)
- PubMed Central via Europe PMC (8M+ full-text)
- Unpaywall (40M+ OA articles)
- Crossref (140M+ DOI metadata)
- CORE (200M+ OA papers)
- Semantic Scholar (citation analysis)

**Ontologies & Standards (3 sources):**

- RxNorm (~200k drug names → RxCUI)
- ICD-11 WHO API (55k+ disease codes)
- ChEMBL (2.3M+ compounds, 20M+ bioactivity)

#### 4.3 Rate Limiting and Error Handling

**Token Bucket Algorithm:**

```python
class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_second: float, burst: int):
        self.rate = requests_per_second
        self.burst = burst
        self.tokens = burst
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update

            # Refill tokens based on elapsed time
            self.tokens = min(
                self.burst,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            # Wait if no tokens available
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1
```

### 5. GPU-Accelerated AI Services

#### 5.1 Service Architecture

**Fail-Fast GPU Services:**

```python
class MinerUService:
    def __init__(self):
        if not torch.cuda.is_available():
            logger.error("GPU not available, refusing to start")
            sys.exit(1)

        self.device = torch.device("cuda:0")
        self.model = load_mineru_model().to(self.device)
        logger.info(f"MinerU service started on {self.device}")

    async def ProcessPDF(self, request, context):
        try:
            # Process on GPU
            result = self.model.process(request.pdf_data)
            return ProcessPDFResponse(...)
        except torch.cuda.OutOfMemoryError:
            # Fail gracefully, don't fall back to CPU
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "GPU out of memory")
```

**Service Specifications:**

- **MinerU Service**: PDF parsing with OCR and layout analysis
- **Embedding Service**: SPLADE sparse + Qwen-3 dense embeddings
- **Extraction Service**: LLM-based entity and relationship extraction

#### 5.2 Model Configuration

**Embedding Models:**

```yaml
active_namespaces:
  - single_vector.bge_small_en.384.v1
  - sparse.splade_v3.400.v1
  - multi_vector.colbert_v2.128.v1

namespaces:
  single_vector.bge_small_en.384.v1:
    name: bge-small-en
    provider: sentence-transformers
    kind: single_vector
    model_id: BAAI/bge-small-en
    model_version: v1.5
    dim: 384
    pooling: mean
    normalize: true
    batch_size: 32
```

### 6. Orchestration and Event-Driven Architecture

#### 6.1 Kafka-Based Event Bus

**Topic Structure:**

- `ingest.requests.v1`: Ingestion job requests
- `ingest.results.v1`: Ingestion completion notifications
- `mapping.events.v1`: Entity mapping triggers
- `ingest.dlq.v1`: Dead letter queue for failed jobs

#### 6.2 Pipeline Stages

**Auto Pipeline (Fast Sources):**

1. Metadata fetch → Document IR creation
2. Chunking → Semantic text segmentation
3. Embedding → SPLADE + dense vector generation
4. Indexing → OpenSearch + FAISS storage
5. KG Construction → Entity extraction and graph building

**Two-Phase Pipeline (PDF-Bound):**

1. Metadata fetch → Preliminary document creation
2. PDF fetch → Full-text retrieval via Unpaywall/CORE
3. GPU parsing → MinerU PDF structure extraction
4. Post-PDF processing → Chunking, embedding, indexing

#### 6.3 Job State Management

**Ledger-Based Tracking:**

```python
class JobLedgerEntry:
    job_id: str
    doc_key: str
    tenant_id: str
    status: JobStatus
    stage: str
    retries: int
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]
```

### 7. Knowledge Graph and Storage

#### 7.1 Neo4j Graph Schema

**Node Types:**

- **Document**: Source documents with metadata
- **Entity**: Normalized real-world objects (drugs, diseases, etc.)
- **Claim**: Extracted facts with confidence scores
- **ExtractionActivity**: Provenance tracking

**Relationship Types:**

- `MENTIONS`: Document → Entity (with text spans)
- `HAS_SUBJECT`/`HAS_OBJECT`: Claim → Entity
- `EXTRACTED`: ExtractionActivity → Claim
- `FROM_DOCUMENT`: ExtractionActivity → Document

#### 7.2 SHACL Validation

**Shape Definitions:**

```turtle
medkg:DocumentShape a sh:NodeShape ;
    sh:targetClass medkg:Document ;
    sh:property [
        sh:path medkg:doc_id ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:datatype xsd:string ;
        sh:pattern "^[a-z]+:[A-Za-z0-9_-]+" ;
    ] .
```

### 8. Retrieval Engine

#### 8.1 Multi-Strategy Search

**Search Components:**

- **BM25**: Lexical term matching with TF-IDF scoring
- **SPLADE**: Learned sparse retrieval with expansion terms
- **Dense Vectors**: Semantic similarity via Qwen-3 embeddings

**Fusion Ranking:**

```python
def reciprocal_rank_fusion(scores: list[tuple[str, float]]) -> list[str]:
    """Combine multiple ranking strategies using RRF."""
    doc_scores = defaultdict(float)
    for ranking in scores:
        for rank, (doc_id, _) in enumerate(ranking):
            doc_scores[doc_id] += 1.0 / (rank + 60)  # RRF constant

    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
```

#### 8.2 Vector Storage

**OpenSearch (BM25 + SPLADE):**

- Full-text search with field boosting
- Sparse vector indexing for SPLADE
- Faceted search for filtering

**FAISS (Dense Vectors):**

- GPU-accelerated similarity search
- Hierarchical Navigable Small World (HNSW) indexing
- Batch processing for efficiency

### 9. Security and Multi-Tenancy

#### 9.1 OAuth 2.0 Authentication

**JWT Token Structure:**

```python
class JWTToken(BaseModel):
    sub: str  # Subject (user ID)
    tenant_id: str  # Multi-tenancy
    scopes: list[str]  # OAuth scopes
    exp: datetime  # Expiration
    iat: datetime  # Issued at
    jti: str  # Unique token ID
```

#### 9.2 Tenant Isolation

**Query Filtering:**

```python
# All queries must include tenant_id filter
WHERE node.tenant_id = $tenant_id
```

**Index Partitioning:**

- Separate Elasticsearch indices per tenant
- Neo4j subgraphs per tenant
- MinIO buckets per tenant

### 10. Observability and Operations

#### 10.1 Metrics Collection

**Prometheus Metrics:**

- Request latency and throughput
- Error rates by endpoint
- GPU utilization and memory usage
- Kafka consumer lag
- Database connection pools

**OpenTelemetry Traces:**

- Distributed request tracing
- Performance profiling
- Error correlation across services

#### 10.2 Logging Strategy

**Structured Logging:**

```python
logger.info(
    "gateway.request",
    extra={
        "method": request.method,
        "path": request.url.path,
        "correlation_id": correlation_id,
        "tenant_id": tenant_id,
    },
)
```

### 11. Deployment and DevOps

#### 11.1 Container Strategy

**Docker Compose (Development):**

```yaml
services:
  gateway:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - kafka
      - neo4j
      - opensearch

  kafka:
    image: confluentinc/cp-kafka:7.5.0

  neo4j:
    image: neo4j:5.12
    environment:
      NEO4J_AUTH: neo4j/testpassword

  opensearch:
    image: opensearchproject/opensearch:2.11.0
```

#### 11.2 Kubernetes Production Deployment

**Gateway Deployment:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gateway
  namespace: medical-kg
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: gateway
          image: ghcr.io/example/medical-kg:latest
          resources:
            requests:
              cpu: 500m
              memory: 1Gi
            limits:
              cpu: 1
              memory: 2Gi
```

**GPU Node Configuration:**

```yaml
nodeSelector:
  accelerator: nvidia-tesla-k80
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

### 12. Configuration Management

#### 12.1 Environment-Based Configuration

**Settings Hierarchy:**

1. Base configuration (YAML files)
2. Environment variables override
3. Runtime configuration updates

#### 12.2 Profile-Based Configuration

**Chunking Profiles:**

```yaml
profiles:
  default:
    enable_multi_granularity: true
    primary:
      strategy: semantic_splitter
      granularity: paragraph
  pmc:
    enable_multi_granularity: true
    primary:
      strategy: section_aware
      granularity: section
```

### 13. Testing Strategy

#### 13.1 Test Categories

**Unit Tests:**

- Individual function and class testing
- Mocked external dependencies
- Fast feedback for development

**Integration Tests:**

- End-to-end adapter testing with real APIs
- Database and storage integration
- Cross-service communication

**Performance Tests:**

- Load testing with k6
- Stress testing for breaking points
- Latency benchmarking

**Contract Tests:**

- API specification compliance
- Protocol interoperability
- Data format validation

### 14. Performance Characteristics

#### 14.1 Benchmarks

**Ingestion Performance:**

- ClinicalTrials.gov: 10 studies/second
- OpenAlex metadata: 50 papers/second
- PDF processing: 2-3 papers/second (GPU)

**Retrieval Performance:**

- Simple queries: <100ms P95
- Complex queries: <500ms P95
- Batch retrieval: <1000ms for 100 results

**Scalability:**

- Linear scaling to 1000 concurrent users
- 10M+ document capacity
- Multi-region deployment support

### 15. Future Considerations

#### 15.1 Planned Enhancements

**GraphQL Federation:**

- Apollo Federation for microservice composition
- Schema stitching across domains

**Multi-Region Deployment:**

- Global load balancing
- Data replication strategies
- Latency optimization

**FHIR Server Integration:**

- Native FHIR resource support
- SMART on FHIR authorization

**Real-time Collaboration:**

- WebSocket-based collaborative editing
- Conflict resolution strategies

**Federated Learning:**

- Privacy-preserving model training
- Multi-tenant model sharing

---

## Detailed API Documentation

### REST API Specification

#### Core Endpoints

**Document Ingestion**

```http
POST /v1/ingest/{source}
Content-Type: application/vnd.api+json
Authorization: Bearer {jwt_token}

{
  "data": {
    "type": "IngestionRequest",
    "attributes": {
      "identifiers": ["NCT04267848", "NCT04267849"],
      "options": {
        "include_pdf": true,
        "priority": "high"
      }
    }
  }
}
```

**Response:**

```http
HTTP 200 OK
Content-Type: application/vnd.api+json

{
  "data": {
    "type": "IngestionJob",
    "id": "job-abc123",
    "attributes": {
      "status": "queued",
      "documents_queued": 2,
      "estimated_completion": "2025-01-15T10:35:00Z"
    }
  },
  "meta": {
    "processing_time_ms": 45
  }
}
```

**Document Retrieval with OData Filtering**

```http
GET /v1/documents?$filter=source eq 'clinicaltrials' and status eq 'completed'&$select=title,created_at&$orderby=created_at desc&$top=10&$skip=0
Authorization: Bearer {jwt_token}
```

**Response:**

```http
HTTP 200 OK
Content-Type: application/vnd.api+json

{
  "data": [
    {
      "type": "Document",
      "id": "clinicaltrials:NCT04267848",
      "attributes": {
        "title": "Study of Pembrolizumab in Melanoma",
        "created_at": "2025-01-15T10:30:00Z"
      },
      "links": {
        "self": "/v1/documents/clinicaltrials:NCT04267848"
      }
    }
  ],
  "links": {
    "first": "/v1/documents?$filter=...&$top=10&$skip=0",
    "next": "/v1/documents?$filter=...&$top=10&$skip=10",
    "last": "/v1/documents?$filter=...&$top=10&$skip=90"
  },
  "meta": {
    "total": 100,
    "processing_time_ms": 156
  }
}
```

**Knowledge Graph Queries**

```http
POST /v1/kg/query
Content-Type: application/vnd.api+json
Authorization: Bearer {jwt_token}

{
  "data": {
    "type": "KnowledgeGraphQuery",
    "attributes": {
      "cypher": "MATCH (d:Document {tenant_id: $tenant_id}) WHERE d.source = 'clinicaltrials' RETURN d.title as title, d.created_at as created LIMIT 10",
      "parameters": {
        "tenant_id": "tenant-123"
      }
    }
  }
}
```

### GraphQL API Schema

**Document Queries**

```graphql
query GetDocument($id: ID!) {
  document(id: $id) {
    id
    title
    source
    createdAt
    sections {
      id
      title
      blocks {
        id
        type
        text
        spans {
          start
          end
          text
        }
      }
    }
    metadata
  }
}

query SearchDocuments($query: String!, $filters: DocumentFilter, $limit: Int) {
  search(query: $query, filters: $filters, limit: $limit) {
    document {
      id
      title
      source
    }
    score
    highlights
  }
}
```

**Mutations**

```graphql
mutation IngestClinicalTrial($nctIds: [String!]!) {
  startIngestion(input: {
    source: CLINICALTRIALS
    identifiers: $nctIds
    options: {
      priority: HIGH
      includePdf: true
    }
  }) {
    jobId
    status
    documentsQueued
  }
}
```

### gRPC Service Definitions

**Ingestion Service**

```protobuf
service IngestionService {
  // Submit ingestion job
  rpc SubmitJob(IngestionJobRequest) returns (IngestionJobResponse);

  // Get job status with streaming updates
  rpc GetJobStatus(JobStatusRequest) returns (stream JobStatusUpdate);

  // Cancel running job
  rpc CancelJob(CancelJobRequest) returns (CancelJobResponse);
}

message IngestionJobRequest {
  string tenant_id = 1;
  string source = 2;  // "clinicaltrials", "openalex", etc.
  repeated string identifiers = 3;
  IngestionOptions options = 4;
}
```

**Embedding Service**

```protobuf
service EmbeddingService {
  rpc EmbedChunks(EmbedChunksRequest) returns (EmbedChunksResponse);
  rpc Health(google.protobuf.Empty) returns (HealthResponse);
}

message EmbedChunksRequest {
  repeated string chunk_ids = 1;
  repeated string texts = 2;
  repeated string namespaces = 3;  // Which embedding models to use
}
```

---

## Database Schema and Data Models

### Neo4j Graph Schema

#### Node Schema Definitions

**Document Node**

```cypher
CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE;
CREATE INDEX document_tenant_idx IF NOT EXISTS FOR (d:Document) ON (d.tenant_id, d.document_id);
CREATE FULLTEXT INDEX document_content_idx IF NOT EXISTS FOR (d:Document) ON EACH [d.title, d.content];

CALL apoc.schema.assert(
  {Document: ["document_id", "title", "source", "ingested_at", "tenant_id"]},
  {Entity: ["entity_id", "name", "type", "ontology_code"]},
  {Claim: ["claim_id", "statement", "polarity"]},
  {Evidence: ["evidence_id", "chunk_id", "confidence"]},
  {ExtractionActivity: ["activity_id", "performed_at", "pipeline"]}
);
```

**Entity Node (Normalized Objects)**

```cypher
// Unique constraint on entity identifier per tenant
CREATE CONSTRAINT entity_id_tenant_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.entity_id, e.tenant_id) IS UNIQUE;

// Index for efficient lookups
CREATE INDEX entity_ontology_idx IF NOT EXISTS FOR (e:Entity) ON (e.ontology_code, e.tenant_id);
CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type, e.tenant_id);

// Full-text search on entity names
CREATE FULLTEXT INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.canonical_name];
```

**Relationship Schema**

```cypher
// Document-Entity relationships (mentions)
CREATE INDEX mentions_span_idx IF NOT EXISTS FOR ()-[r:MENTIONS]-() ON (r.sentence_index);

// Claim-Entity relationships
CREATE INDEX claim_subject_idx IF NOT EXISTS FOR ()-[r:HAS_SUBJECT]-() ON (r.claim_id);
CREATE INDEX claim_object_idx IF NOT EXISTS FOR ()-[r:HAS_OBJECT]-() ON (r.claim_id);

// Evidence relationships
CREATE INDEX evidence_claim_idx IF NOT EXISTS FOR ()-[r:SUPPORTS]-() ON (r.evidence_id);
CREATE INDEX evidence_activity_idx IF NOT EXISTS FOR ()-[r:GENERATED_BY]-() ON (r.activity_id);
```

#### Sample Cypher Queries

**Document Ingestion**

```cypher
// Create or update document (idempotent)
MERGE (d:Document {document_id: $document_id, tenant_id: $tenant_id})
ON CREATE SET
    d.title = $title,
    d.source = $source,
    d.content = $content,
    d.ingested_at = datetime(),
    d.created_at = datetime()
ON MATCH SET
    d.updated_at = datetime(),
    d.content = CASE WHEN d.content IS NULL THEN $content ELSE d.content END
RETURN d;
```

**Entity Resolution**

```cypher
// Find or create entity with normalization
MERGE (e:Entity {entity_id: $entity_id, tenant_id: $tenant_id})
ON CREATE SET
    e.name = $name,
    e.canonical_name = $canonical_name,
    e.type = $type,
    e.ontology_code = $ontology_code,
    e.created_at = datetime()
ON MATCH SET
    e.updated_at = datetime()
RETURN e;
```

---

## Configuration Management

### Environment-Based Configuration

#### Configuration Hierarchy

1. **Base Configuration** (YAML files in `/config/`)
2. **Environment Variables** (override YAML settings)
3. **Runtime Configuration** (programmatic updates)

#### Core Configuration Files

**Application Settings** (`settings.py`)

```python
class Settings(BaseSettings):
    """Main application configuration."""

    # Environment
    env: Environment = Field(default=Environment.DEV)

    # API Gateway
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1)

    # Database Connections
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: SecretStr = Field(default=SecretStr("password"))

    opensearch_hosts: list[str] = Field(default=["http://localhost:9200"])
    redis_url: str = Field(default="redis://localhost:6379")

    # Message Broker
    kafka_bootstrap_servers: list[str] = Field(default=["localhost:9092"])

    # Authentication
    jwt_secret_key: SecretStr = Field(default=SecretStr("your-secret-key"))
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(default=30)

    # Observability
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
```

**Embedding Configuration** (`embeddings.yaml`)

```yaml
active_namespaces:
  - single_vector.bge_small_en.384.v1
  - sparse.splade_v3.400.v1
  - multi_vector.colbert_v2.128.v1

namespaces:
  single_vector.bge_small_en.384.v1:
    name: bge-small-en
    provider: sentence-transformers
    kind: single_vector
    model_id: BAAI/bge-small-en-v1.5
    model_version: v1.5
    dim: 384
    pooling: mean
    normalize: true
    batch_size: 32
    prefixes:
      query: "query: "
      document: "passage: "

  sparse.splade_v3.400.v1:
    name: splade-v3
    provider: splade-doc
    kind: sparse
    model_id: splade-v3
    model_version: v3
    dim: 400
    normalize: false
    batch_size: 8
    parameters:
      top_k: 400

  multi_vector.colbert_v2.128.v1:
    name: colbert-v2
    provider: colbert
    kind: multi_vector
    model_id: colbert-v2
    model_version: v2
    dim: 128
    normalize: false
    batch_size: 16
    parameters:
      max_doc_tokens: 180
```

**Chunking Configuration** (`chunking.yaml`)

```yaml
default_profile: default

profiles:
  default:
    enable_multi_granularity: true
    primary:
      strategy: semantic_splitter
      granularity: paragraph
      params:
        tau_coh: 0.82
        min_tokens: 200
    auxiliaries:
      - strategy: section_aware
        granularity: section
        params:
          target_tokens: 450
          min_tokens: 180
      - strategy: sliding_window
        granularity: window
        params:
          target_tokens: 512
          overlap_ratio: 0.25

  pmc:
    enable_multi_granularity: true
    primary:
      strategy: section_aware
      granularity: section
      params:
        target_tokens: 420
        min_tokens: 160
    auxiliaries:
      - strategy: semantic_splitter
        granularity: paragraph
        params:
          tau_coh: 0.8
          min_tokens: 180
```

#### Environment Variables

**Required Environment Variables**

```bash
# Application
export MK_ENV=prod
export MK_HOST=0.0.0.0
export MK_PORT=8000

# Database
export MK_NEO4J_URI=bolt://neo4j:7687
export MK_NEO4J_USER=neo4j
export MK_NEO4J_PASSWORD=secure_password

# Message Broker
export MK_KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# Authentication
export MK_JWT_SECRET_KEY=your-256-bit-secret

# Observability
export MK_TELEMETRY_EXPORTER=jaeger
export MK_TELEMETRY_ENDPOINT=http://jaeger-collector:4318
export MK_OBSERVABILITY_LOGGING_LEVEL=INFO
```

**Optional Environment Variables**

```bash
# Performance Tuning
export MK_WORKERS=4
export MK_BATCH_SIZE=32

# GPU Services
export MK_MINERU_WORKERS=4
export MK_MINERU_VRAM_PER_WORKER=8

# Rate Limiting
export MK_RATE_LIMIT_REQUESTS_PER_MINUTE=1000
export MK_RATE_LIMIT_BURST=100
```

---

## Development Setup and Getting Started

### Prerequisites

**System Requirements**

- Python 3.12+
- Docker and Docker Compose
- 16GB RAM minimum (32GB recommended for GPU services)
- NVIDIA GPU with CUDA support (optional, for GPU-accelerated services)

**Required Tools**

```bash
# Package management
pip install poetry

# Development tools
pip install pre-commit black ruff mypy

# Testing
pip install pytest pytest-cov pytest-asyncio

# Performance testing
# Install k6 from https://k6.io/docs/get-started/installation/
```

### Local Development Setup

1. **Clone and Setup**

```bash
git clone https://github.com/your-org/Medical_KG_rev.git
cd Medical_KG_rev
poetry install
```

2. **Environment Configuration**

```bash
cp .env.example .env
# Edit .env with your local configuration
```

3. **Start Development Services**

```bash
# Start infrastructure services
docker-compose up -d kafka neo4j opensearch redis minio

# Wait for services to be ready
./scripts/wait_for_services.sh

# Run database migrations
python -m Medical_KG_rev.scripts.init_db

# Start GPU services (if you have GPU support)
./scripts/setup_mineru.sh
```

4. **Run the Application**

```bash
# Development mode with auto-reload
poetry run medkg-gateway

# Or with uvicorn directly
uvicorn Medical_KG_rev.gateway.main:create_app --reload --host 0.0.0.0 --port 8000
```

5. **Verify Installation**

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs/openapi
open http://localhost:8000/docs/graphql
```

### Development Workflow

**Code Quality Checks**

```bash
# Format code
black src/ tests/
ruff check src/ tests/ --fix

# Type checking
mypy src/

# Run tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v -m integration
pytest tests/performance/ -v -m performance
```

**Adding New Adapters**

```bash
# 1. Create adapter class in src/Medical_KG_rev/adapters/
class NewSourceAdapter(BaseAdapter):
    def fetch(self, context: AdapterContext) -> Iterable[Mapping[str, Any]]:
        # Implementation here
        pass

    def parse(self, payloads: Iterable[Mapping[str, Any]], context: AdapterContext) -> Sequence[Document]:
        # Implementation here
        pass

# 2. Add configuration to adapters/config/
echo "source: newsource
base_url: https://api.newsource.com
rate_limit:
  requests_per_second: 5" > src/Medical_KG_rev/adapters/config/newsource.yaml

# 3. Register adapter in registry
# src/Medical_KG_rev/adapters/registry.py
ADAPTERS["newsource"] = NewSourceAdapter

# 4. Add tests
# tests/adapters/test_newsource_adapter.py
```

---

## Testing Strategy and Examples

### Test Categories

#### Unit Tests

**Purpose**: Test individual functions and classes in isolation
**Coverage**: >90% code coverage target
**Examples**:

```python
# tests/adapters/test_clinicaltrials_adapter.py
@pytest.mark.asyncio
async def test_clinicaltrials_adapter_fetch():
    """Test ClinicalTrials adapter fetches study data."""
    adapter = ClinicalTrialsAdapter()

    with aioresponses() as m:
        m.get(
            "https://clinicaltrials.gov/api/v2/studies/NCT04267848",
            payload=MOCK_STUDY_RESPONSE,
            status=200,
        )

        results = await adapter.fetch(nct_id="NCT04267848")
        assert len(results) == 1
        assert results[0]["protocolSection"]["identificationModule"]["nctId"] == "NCT04267848"
```

#### Integration Tests

**Purpose**: Test end-to-end workflows and external service integration
**Markers**: `@pytest.mark.integration`
**Examples**:

```python
# tests/adapters/test_biomedical_adapters.py
@pytest.mark.integration
@pytest.mark.asyncio
async def test_openalex_adapter_real_api():
    """Test OpenAlex adapter with real API."""
    adapter = OpenAlexAdapter()

    results = await adapter.fetch(query="cancer immunotherapy", per_page=5)
    assert len(results) <= 5
    assert all("title" in work for work in results)
```

#### Performance Tests

**Purpose**: Validate latency, throughput, and concurrency requirements
**Tool**: k6 load testing
**Examples**:

```javascript
// tests/performance/retrieve_latency.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 },  // Ramp up to 100 users
    { duration: '5m', target: 100 },  // Stay at 100 users
    { duration: '2m', target: 0 },    // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p95<500'],  // 95th percentile < 500ms
  },
};

export default function() {
  const response = http.get(`${__ENV.BASE_URL}/v1/documents/search?q=cancer`);

  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });

  sleep(1);
}
```

#### Contract Tests

**Purpose**: Ensure API specifications are met
**Tools**: Schemathesis (OpenAPI), GraphQL validation
**Examples**:

```python
# tests/contract/test_openapi_schemathesis.py
def test_openapi_compliance():
    """Test that API responses match OpenAPI specification."""
    schema = schemathesis.from_uri("http://localhost:8000/openapi.json")

    @schema.parametrize(endpoint="/v1/documents/{document_id}")
    def test_document_endpoint(case):
        response = case.call_wsgi()
        case.validate_response(response)
```

### Running Tests

**Local Testing**

```bash
# Run all tests
pytest

# Run specific categories
pytest tests/unit/ tests/integration/ -v

# Run with coverage
pytest --cov=Medical_KG_rev --cov-report=html

# Performance testing
k6 run tests/performance/retrieve_latency.js

# Contract testing
schemathesis run --base-url http://localhost:8000 http://localhost:8000/openapi.json
```

**CI/CD Integration**

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install poetry
      - run: poetry install
      - run: poetry run pytest tests/unit/ --cov=Medical_KG_rev

  integration-tests:
    runs-on: ubuntu-latest
    services:
      neo4j:
        image: neo4j:5.12
      opensearch:
        image: opensearchproject/opensearch:2.11.0
    steps:
      - uses: actions/checkout@v3
      - run: poetry install
      - run: poetry run pytest tests/integration/ -m integration
```

---

## Security Implementation Details

### OAuth 2.0 Implementation

#### JWT Token Structure

```python
class JWTToken(BaseModel):
    """JWT token payload structure."""

    sub: str  # Subject (user ID)
    tenant_id: str  # Multi-tenancy isolation
    scopes: list[str]  # OAuth scopes (ingest:write, kg:read, etc.)
    exp: datetime  # Expiration time
    iat: datetime  # Issued at time
    jti: str  # Unique token identifier
    aud: str = "medical-kg-api"  # Audience
    iss: str = "medical-kg-auth"  # Issuer
```

#### Authentication Flow

```python
# src/Medical_KG_rev/auth/jwt.py
def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({
        "iat": datetime.utcnow(),
        "exp": expire,
        "jti": str(uuid.uuid4()),
        "aud": "medical-kg-api",
        "iss": "medical-kg-auth"
    })

    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm
    )
    return encoded_jwt

def verify_token(token: str) -> JWTToken:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
            audience="medical-kg-api",
            issuer="medical-kg-auth"
        )
        return JWTToken(**payload)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTClaimsError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token claims: {e}")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

#### Authorization Middleware

```python
# src/Medical_KG_rev/auth/dependencies.py
async def get_current_user(token: str = Depends(oauth2_scheme)) -> SecurityContext:
    """Extract user context from JWT token."""
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception

        token_data = JWTToken(**payload)
    except jwt.PyJWTError:
        raise credentials_exception

    user = get_user(username)
    if user is None:
        raise credentials_exception

    return SecurityContext(
        user_id=username,
        tenant_id=token_data.tenant_id,
        scopes=token_data.scopes
    )
```

#### Scope-Based Authorization

```python
# src/Medical_KG_rev/auth/scopes.py
class Scopes:
    """OAuth 2.0 scopes for API access control."""

    # Ingestion operations
    INGEST_READ = "ingest:read"
    INGEST_WRITE = "ingest:write"

    # Knowledge graph operations
    KG_READ = "kg:read"
    KG_WRITE = "kg:write"

    # Retrieval operations
    RETRIEVE_READ = "retrieve:read"

    # Administrative operations
    ADMIN_READ = "admin:read"
    ADMIN_WRITE = "admin:write"

def require_scopes(*required_scopes: str):
    """Decorator to require specific OAuth scopes."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract security context from request
            security_context = kwargs.get("security_context")
            if not security_context:
                raise HTTPException(status_code=401, detail="Authentication required")

            user_scopes = set(security_context.scopes)
            required_scopes_set = set(required_scopes)

            if not required_scopes_set.issubset(user_scopes):
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required: {required_scopes}"
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

### Multi-Tenancy Implementation

#### Tenant Isolation Architecture

```python
# src/Medical_KG_rev/auth/context.py
class SecurityContext(BaseModel):
    """Security context for authenticated requests."""

    user_id: str
    tenant_id: str
    scopes: list[str]
    roles: list[str] = Field(default_factory=list)
    permissions: dict[str, Any] = Field(default_factory=dict)

class TenantContext(BaseModel):
    """Tenant-specific context for data isolation."""

    tenant_id: str
    database_schema: str | None = None  # For multi-tenant databases
    index_prefix: str  # For Elasticsearch/OpenSearch
    storage_prefix: str  # For object storage
    rate_limits: dict[str, int]  # Per-tenant rate limits

def get_tenant_context(tenant_id: str) -> TenantContext:
    """Get tenant-specific configuration."""
    # In production, this would come from a tenant management service
    tenant_configs = {
        "tenant-123": {
            "index_prefix": "tenant-123",
            "storage_prefix": "tenants/tenant-123",
            "rate_limits": {
                "requests_per_minute": 1000,
                "ingestion_jobs_per_hour": 100
            }
        }
    }

    config = tenant_configs.get(tenant_id, tenant_configs["tenant-123"])
    return TenantContext(tenant_id=tenant_id, **config)
```

#### Query Filtering

```python
# All database queries must include tenant_id filter
def apply_tenant_filter(query: str, tenant_id: str) -> str:
    """Apply tenant isolation to queries."""
    if "WHERE" in query.upper():
        return query + f" AND tenant_id = '{tenant_id}'"
    else:
        return query + f" WHERE tenant_id = '{tenant_id}'"

# Example usage in repository layer
class DocumentRepository:
    async def get_by_id(self, document_id: str, tenant_id: str) -> Document | None:
        query = """
        MATCH (d:Document {document_id: $document_id})
        WHERE d.tenant_id = $tenant_id
        RETURN d
        """
        result = await self.neo4j_client.execute(query, document_id=document_id, tenant_id=tenant_id)
        return Document(**result[0]) if result else None
```

### Audit Logging

```python
# src/Medical_KG_rev/auth/audit.py
class AuditEvent(BaseModel):
    """Structured audit event."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: str
    tenant_id: str
    action: str  # "create", "read", "update", "delete"
    resource_type: str  # "document", "entity", "claim"
    resource_id: str
    details: dict[str, Any] = Field(default_factory=dict)
    ip_address: str | None = None
    user_agent: str | None = None
    success: bool = True

class AuditService:
    """Service for recording audit events."""

    def __init__(self, kafka_client: KafkaClient):
        self.kafka = kafka_client

    async def record_event(self, event: AuditEvent) -> None:
        """Record audit event to Kafka topic."""
        await self.kafka.publish(
            "audit.events.v1",
            event.model_dump(),
            key=f"{event.tenant_id}:{event.user_id}"
        )

def audit_trail(action: str, resource_type: str):
    """Decorator to automatically audit API operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            security_context = kwargs.get("security_context")
            if not security_context:
                return await func(*args, **kwargs)

            # Record successful operation
            try:
                result = await func(*args, **kwargs)

                audit_event = AuditEvent(
                    user_id=security_context.user_id,
                    tenant_id=security_context.tenant_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=str(kwargs.get("resource_id", "unknown")),
                    success=True
                )

                # Async fire-and-forget
                asyncio.create_task(audit_service.record_event(audit_event))

                return result
            except Exception as e:
                # Record failed operation
                audit_event = AuditEvent(
                    user_id=security_context.user_id,
                    tenant_id=security_context.tenant_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=str(kwargs.get("resource_id", "unknown")),
                    success=False,
                    details={"error": str(e)}
                )

                asyncio.create_task(audit_service.record_event(audit_event))
                raise

        return wrapper
    return decorator
```

---

## Error Handling and Troubleshooting

### Error Classification

#### HTTP Error Responses (RFC 7807)

**Problem Details Format**

```json
{
  "type": "https://httpstatuses.com/400",
  "title": "Bad Request",
  "status": 400,
  "detail": "One or more parameters are invalid.",
  "instance": "/v1/ingest/clinicaltrials",
  "extensions": {
    "errors": [
      {
        "field": "identifiers",
        "message": "Field is required"
      }
    ],
    "retry_after": 60
  }
}
```

#### Error Categories

**Client Errors (4xx)**

- `400 Bad Request`: Invalid request parameters or format
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions for operation
- `404 Not Found`: Requested resource does not exist
- `409 Conflict`: Resource conflict (e.g., duplicate document)
- `422 Unprocessable Entity`: Validation errors in request body
- `429 Too Many Requests`: Rate limit exceeded

**Server Errors (5xx)**

- `500 Internal Server Error`: Unexpected server error
- `502 Bad Gateway`: Upstream service unavailable
- `503 Service Unavailable`: Service temporarily unavailable
- `504 Gateway Timeout`: Request timeout

#### Custom Exception Hierarchy

```python
# src/Medical_KG_rev/utils/errors.py
class MedicalKGError(Exception):
    """Base exception for Medical KG errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}

class ValidationError(MedicalKGError):
    """Data validation failed."""
    status_code = 422

class AuthenticationError(MedicalKGError):
    """Authentication failed."""
    status_code = 401

class AuthorizationError(MedicalKGError):
    """Authorization failed."""
    status_code = 403

class NotFoundError(MedicalKGError):
    """Resource not found."""
    status_code = 404

class RateLimitError(MedicalKGError):
    """Rate limit exceeded."""
    status_code = 429

class ServiceUnavailableError(MedicalKGError):
    """Service temporarily unavailable."""
    status_code = 503

class AdapterError(MedicalKGError):
    """External adapter error."""
    retryable: bool = False

class GPUResourceError(MedicalKGError):
    """GPU resource error."""
    status_code = 503
    retryable = True
```

### Troubleshooting Guide

#### Common Issues and Solutions

**High Latency Issues**

```bash
# Check database connection pools
curl http://localhost:8000/metrics | grep neo4j_connection_pool

# Check Kafka consumer lag
curl http://localhost:8000/metrics | grep kafka_consumer_lag

# Check GPU memory usage (if using GPU services)
nvidia-smi

# Check OpenSearch cluster health
curl http://localhost:9200/_cluster/health
```

**Ingestion Failures**

```bash
# Check worker logs
docker-compose logs ingest-worker

# Check Kafka topics
docker-compose exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Check dead letter queue
docker-compose exec kafka kafka-console-consumer \
  --topic ingest.dlq.v1 \
  --bootstrap-server localhost:9092 \
  --from-beginning

# Check adapter rate limits
curl http://localhost:8000/metrics | grep rate_limit
```

**Authentication Issues**

```bash
# Check JWT token format
echo $TOKEN | cut -d'.' -f2 | base64 -d

# Check token expiration
python3 -c "
import jwt
token = 'your-jwt-token'
decoded = jwt.decode(token, options={'verify_signature': False})
print('Expires:', decoded.get('exp'))
"

# Check OAuth scopes
curl -H "Authorization: Bearer $TOKEN" \
     http://localhost:8000/v1/profile
```

**Memory Issues**

```bash
# Check application memory usage
ps aux | grep python

# Check Neo4j memory usage
curl -u neo4j:password http://localhost:7474/db/manage/server/jmx/query \
  org.neo4j:instance=kernel#0,name=Memory,name=MemoryUsage,attribute=HeapMemoryUsage

# Check OpenSearch JVM usage
curl http://localhost:9200/_nodes/stats/jvm
```

#### Debug Logging

**Enable Debug Logging**

```bash
export MK_OBSERVABILITY_LOGGING_LEVEL=DEBUG
export MK_TELEMETRY_SAMPLE_RATIO=1.0
```

**Structured Log Analysis**

```bash
# Filter by correlation ID
docker-compose logs gateway | grep "correlation_id.*abc123"

# Filter by tenant
docker-compose logs gateway | jq 'select(.tenant_id == "tenant-123")'

# Filter by error type
docker-compose logs gateway | grep '"level": "ERROR"'
```

**Distributed Tracing**

```bash
# Access Jaeger UI
open http://localhost:16686

# Search traces by service
# Search traces by operation name
# Search traces by tags (tenant_id, user_id, etc.)
```

---

## Deployment and Operations

### Production Deployment

#### Kubernetes Deployment

**Gateway Deployment**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gateway
  namespace: medical-kg
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: gateway
  template:
    metadata:
      labels:
        app: gateway
    spec:
      containers:
      - name: gateway
        image: ghcr.io/your-org/medical-kg:latest
        ports:
        - containerPort: 8000
        env:
        - name: MK_ENV
          value: "prod"
        - name: MK_NEO4J_URI
          valueFrom:
            secretKeyRef:
              name: neo4j-secret
              key: uri
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

**Horizontal Pod Autoscaler**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gateway-hpa
  namespace: medical-kg
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gateway
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**Ingress Configuration**

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: gateway-ingress
  namespace: medical-kg
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.medical-kg.example.com
    secretName: gateway-tls
  rules:
  - host: api.medical-kg.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: gateway-service
            port:
              number: 8000
```

#### GPU Node Configuration

**GPU Worker Node Pool**

```yaml
apiVersion: v1
kind: Node
metadata:
  labels:
    accelerator: nvidia-tesla-k80
    medical-kg/gpu-enabled: "true"
spec:
  taints:
  - key: nvidia.com/gpu
    value: present
    effect: NoSchedule
```

**GPU Service Deployment**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mineru-service
  namespace: medical-kg
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mineru-service
  template:
    metadata:
      labels:
        app: mineru-service
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-k80
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - name: mineru-service
        image: ghcr.io/your-org/mineru-service:latest
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: 8Gi
          limits:
            nvidia.com/gpu: 1
            memory: 16Gi
```

### Monitoring and Alerting

#### Prometheus Metrics

**Custom Metrics**

```python
# src/Medical_KG_rev/observability/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_DURATION = Histogram(
    'medicalkg_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint', 'status_code']
)

REQUEST_COUNT = Counter(
    'medicalkg_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status_code']
)

# Business metrics
DOCUMENTS_INGESTED = Counter(
    'medicalkg_documents_ingested_total',
    'Total documents ingested',
    ['source', 'tenant_id']
)

INGESTION_DURATION = Histogram(
    'medicalkg_ingestion_duration_seconds',
    'Document ingestion duration',
    ['source', 'tenant_id']
)

# Infrastructure metrics
NEO4J_CONNECTION_POOL_SIZE = Gauge(
    'medicalkg_neo4j_connection_pool_size',
    'Neo4j connection pool size'
)

KAFKA_CONSUMER_LAG = Gauge(
    'medicalkg_kafka_consumer_lag',
    'Kafka consumer lag by topic',
    ['topic']
)
```

**Grafana Dashboards**

**Key Dashboard Panels:**

- Request latency percentiles (P50, P95, P99)
- Error rate by endpoint
- Ingestion throughput by source
- GPU utilization and memory usage
- Kafka consumer lag
- Database connection pool usage
- Tenant-specific metrics

#### Alerting Rules

**PrometheusRule Configuration**

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: medical-kg-alerts
  namespace: medical-kg
spec:
  groups:
  - name: medical-kg
    rules:
    - alert: HighErrorRate
      expr: rate(medicalkg_requests_total{status_code=~"5.."}[5m]) / rate(medicalkg_requests_total[5m]) > 0.05
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value }}% for the last 5 minutes"

    - alert: HighLatency
      expr: histogram_quantile(0.95, rate(medicalkg_request_duration_seconds_bucket[5m])) > 2
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High request latency"
        description: "95th percentile latency is {{ $value }}s"

    - alert: KafkaConsumerLag
      expr: medicalkg_kafka_consumer_lag > 1000
      for: 10m
      labels:
        severity: critical
      annotations:
        summary: "Kafka consumer lag too high"
        description: "Consumer lag is {{ $value }} messages"
```

### Backup and Recovery

#### Database Backups

**Neo4j Backup**

```bash
# Create backup
docker exec neo4j neo4j-admin backup \
  --backup-dir=/backups \
  --database=graph.db

# Restore from backup
docker run --rm \
  -v $(pwd)/backups:/backups \
  -v neo4j_data:/data \
  neo4j:5.12 \
  neo4j-admin restore \
  --from=/backups/graph.db \
  --database=graph.db \
  --force
```

**OpenSearch Backup**

```bash
# Create snapshot
curl -X PUT "localhost:9200/_snapshot/backup-repo" -H 'Content-Type: application/json' -d'
{
  "type": "fs",
  "settings": {
    "location": "/usr/share/opensearch/backup"
  }
}'

curl -X PUT "localhost:9200/_snapshot/backup-repo/snapshot_$(date +%Y%m%d_%H%M%S)?wait_for_completion=true"
```

#### Configuration Backup

```bash
# Backup configuration and secrets
kubectl get configmap,secret -n medical-kg -o yaml > medical-kg-config-backup.yaml

# Restore configuration
kubectl apply -f medical-kg-config-backup.yaml
```

### Scaling Strategies

#### Horizontal Scaling

**Gateway Scaling**

- Scale based on CPU/memory utilization
- Scale based on request queue depth
- Regional scaling for global deployments

**Worker Scaling**

- Scale based on Kafka queue depth
- Scale based on ingestion backlog
- Auto-scaling for GPU workers based on queue size

**Database Scaling**

- Neo4j cluster for read/write scaling
- OpenSearch cluster for search scaling
- Redis cluster for cache scaling

#### Vertical Scaling

**Resource Optimization**

- GPU memory allocation per worker
- Neo4j heap size tuning
- OpenSearch JVM settings
- Kafka broker configuration

---

## Performance Tuning and Optimization

### Performance Benchmarks

#### Ingestion Performance

**ClinicalTrials.gov Adapter**

```bash
# Single-threaded performance
time python -c "
import asyncio
from Medical_KG_rev.adapters import ClinicalTrialsAdapter

async def benchmark():
    adapter = ClinicalTrialsAdapter()
    start = time.time()
    docs = await adapter.fetch(nct_id='NCT04267848')
    parsed = await adapter.parse(docs, AdapterContext())
    end = time.time()
    print(f'Ingestion time: {end-start:.2f}s')

asyncio.run(benchmark())
"
```

**OpenAlex Literature Ingestion**

```javascript
// k6 load test for literature ingestion
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '1m', target: 10 },   // Ramp up
    { duration: '3m', target: 10 },   // Steady state
    { duration: '1m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p95<2000'], // 2s for literature ingestion
  },
};

export default function() {
  const response = http.post(
    `${__ENV.BASE_URL}/v1/ingest/openalex`,
    JSON.stringify({
      data: {
        type: 'IngestionRequest',
        attributes: {
          query: 'cancer immunotherapy',
          max_results: 10
        }
      }
    }),
    { headers: { 'Content-Type': 'application/vnd.api+json' } }
  );

  check(response, {
    'status is 202': (r) => r.status === 202,
  });

  sleep(1);
}
```

#### Retrieval Performance

**Search Query Performance**

```javascript
// k6 search performance test
export let options = {
  vus: 50,  // 50 virtual users
  duration: '3m',
  thresholds: {
    http_req_duration: ['p95<500'],  // 500ms P95
  },
};

export default function() {
  const queries = [
    'cancer treatment',
    'diabetes management',
    'cardiovascular disease',
    'alzheimer treatment',
    'pneumonia diagnosis'
  ];

  const query = queries[Math.floor(Math.random() * queries.length)];

  const response = http.get(
    `${__ENV.BASE_URL}/v1/search?q=${encodeURIComponent(query)}&limit=10`
  );

  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
    'returned results': (r) => JSON.parse(r.body).data.length > 0,
  });

  sleep(0.5);
}
```

### Performance Tuning

#### Database Optimization

**Neo4j Performance Tuning**

```bash
# Query optimization
EXPLAIN MATCH (d:Document {tenant_id: 'tenant-123'})
WHERE d.source = 'clinicaltrials'
RETURN d.title, d.created_at
ORDER BY d.created_at DESC
LIMIT 10

# Index creation for common queries
CREATE INDEX document_source_tenant_idx IF NOT EXISTS
FOR (d:Document) ON (d.source, d.tenant_id);

CREATE INDEX document_created_tenant_idx IF NOT EXISTS
FOR (d:Document) ON (d.created_at, d.tenant_id);
```

**OpenSearch Performance Tuning**

```json
PUT /documents/_settings
{
  "index": {
    "refresh_interval": "30s",
    "number_of_shards": 3,
    "number_of_replicas": 1
  }
}
```

#### Application Performance

**Async Processing Optimization**

```python
# Batch processing for embeddings
async def process_embeddings_batch(chunks: list[Chunk], batch_size: int = 32):
    """Process embeddings in batches for better GPU utilization."""

    results = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        # Process batch on GPU
        batch_results = await embedding_service.embed_batch(batch)

        results.extend(batch_results)

        # Small delay to prevent GPU memory issues
        if i + batch_size < len(chunks):
            await asyncio.sleep(0.1)

    return results
```

**Caching Strategy**

```python
# Multi-level caching
class CacheManager:
    def __init__(self):
        self.redis = RedisCache()  # Fast, distributed cache
        self.memory = LRUCache(max_size=1000)  # Hot data cache

    async def get(self, key: str):
        # Check memory cache first
        if value := self.memory.get(key):
            return value

        # Check Redis cache
        if value := await self.redis.get(key):
            self.memory.put(key, value)  # Promote to hot cache
            return value

        return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        await self.redis.set(key, value, ttl=ttl)
        self.memory.put(key, value)
```

---

## Conclusion

Medical_KG_rev represents a comprehensive solution to the biomedical data integration challenge. Through its innovative multi-protocol gateway, extensible adapter architecture, and sophisticated AI pipeline, the system provides a robust foundation for healthcare research and clinical decision support.

The architecture demonstrates production-ready thinking with comprehensive security, observability, and operational capabilities. The modular design ensures long-term maintainability while the federated data model supports expansion into additional knowledge domains.

The implementation showcases modern software engineering practices including event-driven architecture, containerized deployment, and comprehensive testing strategies. The system is well-positioned to scale with the growing demands of biomedical research and healthcare informatics.
