# System Architecture & Design Rationale

## Medical_KG_rev: Multi-Protocol API Gateway for Biomedical Knowledge Integration

**Version**: 1.1
**Date**: October 6, 2025
**Status**: Architecture Design Document - IMPLEMENTATION COMPLETE

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Architecture Principles](#3-architecture-principles)
4. [Core Components](#4-core-components)
5. [Data Flow & Pipelines](#5-data-flow--pipelines)
6. [Multi-Protocol API Design](#6-multi-protocol-api-design)
7. [Adapter SDK & Extensibility](#7-adapter-sdk--extensibility)
8. [Knowledge Graph Schema](#8-knowledge-graph-schema)
9. [Retrieval Architecture](#9-retrieval-architecture)
10. [Security & Multi-Tenancy](#10-security--multi-tenancy)
11. [Observability & Operations](#11-observability--operations)
12. [Deployment Architecture](#12-deployment-architecture)
13. [Design Decisions & Trade-offs](#13-design-decisions--trade-offs)
14. [Future Considerations](#14-future-considerations)

---

## 1. Executive Summary

### 1.1 Purpose

Medical_KG_rev is a production-ready, multi-protocol API gateway and orchestration system designed to unify fragmented biomedical data from diverse sources into a coherent knowledge graph with advanced retrieval capabilities. The system addresses the critical challenge faced by healthcare researchers, pharmaceutical companies, and medical informaticists: **data fragmentation across incompatible APIs, formats, and standards**.

### 1.2 Key Innovations

1. **Multi-Protocol Façade**: Single backend accessible via 5 protocols (REST, GraphQL, gRPC, SOAP, AsyncAPI/SSE), enabling integration with any client technology stack

2. **Federated Data Model**: Unified Intermediate Representation (IR) with domain-specific overlays (medical/FHIR, financial/XBRL, legal/LegalDocML) allows medical and other knowledge domains to coexist

3. **Plug-in Adapter Architecture**: YAML-based connector SDK inspired by Singer/Airbyte enables adding new data sources declaratively without code changes

4. **GPU-Accelerated AI Pipeline**: Fail-fast GPU services for PDF parsing (MinerU), embeddings (SPLADE + Qwen-3), and LLM extraction ensure high-quality content processing

5. **Multi-Strategy Retrieval**: Hybrid search combining lexical (BM25), learned sparse (SPLADE), and dense semantic vectors with fusion ranking delivers superior relevance

6. **Provenance-First Design**: Every extracted fact traceable to source document, extraction method, and timestamp enables trust and reproducibility

### 1.3 Target Scale

- **Data Volume**: 10M+ documents, 100M+ entities, 1B+ relationships
- **Query Performance**: P95 < 500ms for retrieval queries
- **Concurrent Users**: 1000+ simultaneous API clients
- **Ingestion Throughput**: 100+ documents/second
- **Geographic Distribution**: Multi-region deployment capability

### 1.4 Standards Compliance

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

### 1.5 Implementation Status

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

## 2. System Overview

### 2.1 High-Level Architecture

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

### 2.2 Component Responsibilities

| Component | Responsibility | Technology Choices | Implementation Status |
|-----------|---------------|-------------------|----------------------|
| **API Gateway** | Protocol translation, auth, rate limiting | FastAPI, Strawberry GraphQL, gRPC, Zeep | ✅ Complete (5 protocols) |
| **Service Layer** | Business logic, protocol-agnostic | Python 3.12, Pydantic v2 | ✅ Complete |
| **Orchestration** | Job coordination, state management | Apache Kafka, Redis (ledger) | ✅ Complete |
| **Adapters** | External API integration | httpx, pyalex, zeep, biopython | ✅ Complete (11+ sources) |
| **GPU Services** | AI/ML processing | PyTorch, Transformers, MinerU | ✅ Complete (3 services) |
| **Storage** | Persistence | Neo4j 5.x, OpenSearch, FAISS, MinIO, Redis | ✅ Complete |
| **Validation** | Domain-specific validation | pint (UCUM), jsonschema (FHIR), pyshacl (SHACL) | ✅ Complete |
| **Observability** | Monitoring, tracing, logging | Prometheus, OpenTelemetry, structlog, Sentry | ✅ Complete |

### 2.2.1 Data Sources (11+ Adapters Implemented)

**Clinical Research**:

- ClinicalTrials.gov API v2 (450k+ studies)

**Literature (6 sources)**:

- OpenAlex (250M+ works)
- PubMed Central via Europe PMC (8M+ full-text)
- Unpaywall (40M+ OA articles)
- Crossref (140M+ DOI metadata)
- CORE (200M+ OA papers)
- Semantic Scholar (citation analysis)

**Drug Safety & Regulatory (3 sources)**:

- OpenFDA Drug Labels (FDA-approved SPL)
- OpenFDA Adverse Events (FAERS reports)
- OpenFDA Medical Devices (registrations, recalls)

**Ontologies & Standards (3 sources)**:

- RxNorm (~200k drug names → RxCUI)
- ICD-11 WHO API (55k+ disease codes)
- ChEMBL (2.3M+ compounds, 20M+ bioactivity)

### 2.3 Request Flow Example: Literature Ingestion

```
1. Client → POST /ingest/openalex {query: "SGLT2 heart failure"}
2. Gateway → Validate JWT, check scope (ingest:write)
3. Gateway → Publish to Kafka topic: ingest.requests.v1
4. Worker → Consume message, call OpenAlexAdapter
5. OpenAlexAdapter → Query OpenAlex API, parse JSON
6. OpenAlexAdapter → Create Document IR, write to ledger
7. Orchestrator → Detect OA papers, trigger PDF fetch
8. UnpaywallAdapter → Get PDF URLs
9. COREAdapter → Download PDFs to MinIO
10. GPU Worker → Call MinerU service via gRPC
11. MinerU → Parse PDF, extract layout, OCR text
12. MinerU → Return structured blocks
13. PostPDF Pipeline → Chunk → Embed → Index
14. ChunkingService → Semantic chunking (paragraph-aware)
15. EmbeddingService → Call SPLADE + Qwen-3 (gRPC)
16. IndexingService → Write to OpenSearch + FAISS
17. KGService → Extract entities, create graph nodes
18. Orchestrator → Publish to ingest.results.v1
19. Gateway → Emit SSE event: jobs.completed
20. Client → Receive completion notification via SSE
```

---

## 3. Architecture Principles

### 3.1 Design Principles

#### 1. **Protocol Agnostic Core**

- **Principle**: Business logic independent of API protocol
- **Implementation**: Service layer callable from REST, GraphQL, gRPC, SOAP equally
- **Benefit**: Add new protocols without rewriting logic; test once, use everywhere

#### 2. **Fail-Fast Validation**

- **Principle**: Reject invalid inputs at system boundaries
- **Implementation**: Pydantic validation on all inputs; ID format checks before API calls; GPU availability checks on service startup
- **Benefit**: Prevent cascading failures; easier debugging; better error messages

#### 3. **Provenance by Default**

- **Principle**: Every data point traceable to its origin
- **Implementation**: ExtractionActivity nodes in graph; source URLs in metadata; timestamps on all writes
- **Benefit**: Trust, reproducibility, compliance (HIPAA, GDPR)

#### 4. **Idempotency Everywhere**

- **Principle**: Same input produces same output; safe to retry
- **Implementation**: MERGE operations in Neo4j; unique doc_ids; ledger tracks completed jobs
- **Benefit**: Fault tolerance; simplifies retry logic; prevents duplicates

#### 5. **Event-Driven Orchestration**

- **Principle**: Loose coupling via message broker
- **Implementation**: Kafka topics for async job processing; workers subscribe to topics
- **Benefit**: Scalability; fault isolation; easy to add new pipeline stages

#### 6. **Multi-Tenancy from Day One**

- **Principle**: Data isolation enforced at all layers
- **Implementation**: tenant_id in JWT; all queries filtered by tenant; separate indices per tenant
- **Benefit**: Single deployment serves multiple customers; compliance; security

#### 7. **Observability Built-In**

- **Principle**: System behavior visible in production
- **Implementation**: OpenTelemetry spans on all operations; Prometheus metrics everywhere; structured logs with correlation IDs
- **Benefit**: Fast incident response; performance optimization; capacity planning

### 3.2 Non-Functional Requirements

| Requirement | Target | Measurement |
|-------------|--------|-------------|
| **Availability** | 99.9% uptime | Prometheus up/down checks |
| **Latency (Retrieval)** | P95 < 500ms | k6 performance tests |
| **Throughput (Ingestion)** | 100 docs/sec | Kafka consumer lag |
| **Scalability** | 10M documents | Load testing |
| **Data Freshness** | < 24hr from source | Monitoring dashboards |
| **Error Rate** | < 0.1% | Error rate alerts |
| **Security** | Zero data breaches | Penetration testing |

---

## 4. Core Components

### 4.1 Federated Data Model

#### 4.1.1 Design Rationale

**Problem**: Biomedical data (clinical trials, literature, drug labels) and potential future domains (finance, law) have overlapping structure but domain-specific fields.

**Solution**: Federated model with:

- **Core entities** (Document, Block, Entity, Claim): Shared across all domains
- **Domain overlays** (Medical, Financial, Legal): Extend core with domain-specific metadata
- **Discriminated unions**: Pydantic allows type-safe domain dispatch

**Benefits**:

- Single codebase handles multiple knowledge domains
- Medical investment (adapters, pipelines, retrieval) reusable for finance/law
- Type safety ensures domain-specific logic only applies to correct documents

#### 4.1.2 Core Entity Schemas

**Document** (Pydantic model):

```python
class Document(BaseModel):
    id: str  # Format: {source}:{source_id}#{version}:{hash12}
    type: Literal["Document"] = "Document"
    title: str
    content: Optional[str]  # Full text or abstract
    status: DocumentStatus  # draft, published, retracted, etc.
    source: str  # "clinicaltrials", "openalex", "openfda", etc.
    meta: dict[str, Any]  # Domain-specific metadata
    blocks: list[Block]  # Semantic chunks
    created_at: datetime
    updated_at: datetime
    tenant_id: str  # Multi-tenancy
```

**Block** (semantic chunk):

```python
class Block(BaseModel):
    id: str
    doc_id: str  # Foreign key to Document
    type: BlockType  # paragraph, section, table, figure
    text: str
    order: int  # Position in document
    section_title: Optional[str]
    meta: dict[str, Any]
    spans: list[Span]  # Entity mentions
```

**Entity** (normalized real-world object):

```python
class Entity(BaseModel):
    id: str  # RxCUI, ICD-11 code, SNOMED, etc.
    type: EntityType  # drug, disease, gene, organization
    name: str
    canonical_name: str  # Normalized form
    ontology: str  # "rxnorm", "icd11", "snomed"
    synonyms: list[str]
    properties: dict[str, Any]
```

**Claim** (extracted fact):

```python
class Claim(BaseModel):
    id: str
    subject: Entity
    predicate: str  # "treats", "causes", "interacts_with"
    object: Entity
    confidence: float  # [0, 1]
    evidence_spans: list[Span]
    extraction_activity: ExtractionActivity
```

**ExtractionActivity** (provenance):

```python
class ExtractionActivity(BaseModel):
    id: str
    model_name: str  # "gpt-4-2024-01", "qwen-3-1.5b"
    prompt_version: str
    timestamp: datetime
    parameters: dict[str, Any]  # temperature, top_p, etc.
```

#### 4.1.3 Medical Domain Overlay

**MedicalMetadata** (FHIR-aligned):

```python
class MedicalMetadata(BaseModel):
    # Clinical Trial fields
    nct_id: Optional[str]
    phase: Optional[ClinicalPhase]  # phase1, phase2, phase3, phase4
    status: Optional[TrialStatus]  # recruiting, completed, etc.
    interventions: list[Intervention]
    conditions: list[Condition]  # ICD-11 codes
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

This allows:

```python
# Type-safe dispatch
if document.meta isinstance MedicalMetadata:
    trial = extract_trial_info(document.meta.nct_id)
elif document.meta isinstance FinancialMetadata:
    filing = extract_xbrl(document.meta.sec_cik)
```

### 4.2 Adapter SDK

#### 4.2.1 Design Rationale

**Problem**: Integrating new data sources requires understanding API authentication, pagination, rate limits, response parsing—high friction for adding sources.

**Solution**: Adapter SDK inspired by Singer taps and Airbyte connectors:

- **YAML configs** for simple REST APIs (declarative)
- **Python classes** for complex sources (imperative)
- **Registry pattern** for dynamic discovery
- **Base lifecycle**: fetch() → parse() → validate() → write()

**Benefits**:

- Add new sources without modifying core code
- YAML configs enable non-developers to add sources
- Uniform testing and error handling
- Rate limiting and retry logic reusable across adapters

#### 4.2.2 Base Adapter Interface

```python
class BaseAdapter(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.http_client = get_http_client(
            retries=config.get("retries", 3),
            timeout=config.get("timeout", 30),
        )
        self.rate_limiter = RateLimiter(
            requests_per_second=config.get("rate_limit", 10)
        )

    @abstractmethod
    async def fetch(self, identifier: str) -> Any:
        """Fetch raw data from external source"""
        pass

    @abstractmethod
    def parse(self, raw_data: Any) -> Document:
        """Parse raw data into IR Document"""
        pass

    def validate(self, document: Document) -> Document:
        """Validate document against schema"""
        return Document.model_validate(document)

    async def write(self, document: Document) -> str:
        """Write to storage, return doc_id"""
        doc_id = await storage.write_document(document)
        await ledger.mark_complete(doc_id)
        return doc_id
```

#### 4.2.3 YAML Adapter Example

**clinicaltrials.yaml**:

```yaml
source: "ClinicalTrialsAPI"
base_url: "https://clinicaltrials.gov/api/v2"
rate_limit:
  requests: 10
  per_seconds: 1
auth:
  type: "none"
endpoints:
  get_study:
    path: "/studies/{nct_id}"
    method: "GET"
    params:
      format: "json"
    response:
      type: "json"
      extract: "$.protocolSection"
    mapping:
      document:
        id: "$.identificationModule.nctId"
        title: "$.identificationModule.briefTitle"
        status: "$.statusModule.overallStatus"
        meta:
          nct_id: "$.identificationModule.nctId"
          phase: "$.designModule.phases[0]"
          interventions: "$.armsInterventionsModule.interventions"
          conditions: "$.conditionsModule.conditions"
```

Auto-generates:

```python
class ClinicalTrialsAdapter(RESTAdapter):
    def __init__(self):
        config = load_yaml("adapters/clinicaltrials.yaml")
        super().__init__(config)

    async def fetch(self, nct_id: str) -> dict:
        url = self.config["base_url"] + f"/studies/{nct_id}"
        response = await self.http_client.get(url, params={"format": "json"})
        return response.json()["protocolSection"]

    def parse(self, data: dict) -> Document:
        # Auto-generated from mapping config
        return Document(
            id=f"clinicaltrials:{data['identificationModule']['nctId']}",
            title=data['identificationModule']['briefTitle'],
            ...
        )
```

#### 4.2.4 Complex Adapter Example (SOAP)

**Europe PMC** (requires manual implementation):

```python
class EuropePMCAdapter(SOAPAdapter):
    def __init__(self):
        self.client = zeep.Client("https://www.ebi.ac.uk/europepmc/webservices/soap?wsdl")

    async def fetch(self, pmcid: str) -> str:
        """Fetch XML via SOAP"""
        response = self.client.service.searchPMC(query=f"ext_id:{pmcid}")
        return response.fullTextUrlList.fullTextUrl[0].url

    def parse(self, xml_url: str) -> Document:
        """Parse XML to IR"""
        xml = await download_xml(xml_url)
        tree = etree.fromstring(xml)

        # Extract structured sections
        title = tree.find(".//article-title").text
        abstract = tree.find(".//abstract").text
        sections = [
            Block(
                type="section",
                text=section.find(".//p").text,
                section_title=section.find(".//title").text
            )
            for section in tree.findall(".//sec")
        ]

        return Document(
            id=f"pmc:{pmcid}",
            title=title,
            blocks=[Block(type="abstract", text=abstract)] + sections,
            meta=MedicalMetadata(pmcid=pmcid, ...)
        )
```

### 4.3 GPU Microservices Architecture

#### 4.3.1 Design Rationale

**Problem**: PDF parsing, embedding generation, and LLM extraction require GPUs. Running on CPU is 10-100x slower and produces lower quality.

**Solution**: Dedicated GPU microservices accessible via gRPC:

- **Fail-fast**: Services check GPU availability on startup, refuse to start on CPU
- **gRPC**: High-performance RPC with HTTP/2, efficient for large payloads (PDFs, vectors)
- **Batch processing**: Amortize GPU overhead across multiple requests
- **Model caching**: Load models once, reuse across requests

**Benefits**:

- Guaranteed performance (no silent CPU fallback degrading quality)
- Independent scaling (can scale GPU services separately from gateway)
- Technology flexibility (services can be Python, C++, Rust—gateway doesn't care)
- Cost optimization (GPU nodes only for services that need them)

#### 4.3.2 Service Specifications

**MinerU Service** (PDF Parsing):

```protobuf
service MinerUService {
  rpc ProcessPDF(ProcessPDFRequest) returns (ProcessPDFResponse);
  rpc Health(Empty) returns (HealthResponse);
}

message ProcessPDFRequest {
  string document_id = 1;
  bytes pdf_data = 2;  // Or S3 URI
  ProcessOptions options = 3;
}

message ProcessOptions {
  bool extract_tables = 1;
  bool extract_figures = 2;
  int32 dpi = 3;  // For OCR
}

message ProcessPDFResponse {
  string document_id = 1;
  repeated Block blocks = 2;
  repeated Table tables = 3;
  ProcessingStats stats = 4;
}
```

**Embedding Service** (SPLADE + Qwen-3):

```protobuf
service EmbeddingService {
  rpc EmbedChunks(EmbedChunksRequest) returns (EmbedChunksResponse);
  rpc Health(Empty) returns (HealthResponse);
}

message EmbedChunksRequest {
  repeated string chunk_ids = 1;
  repeated string texts = 2;
  bool use_splade = 3;
  bool use_dense = 4;
}

message EmbedChunksResponse {
  map<string, EmbedResult> results = 1;
}

message EmbedResult {
  string chunk_id = 1;
  SparseVector splade_vector = 2;  // ~30k dims, sparse
  repeated float dense_vector = 3;  // 768 or 1536 dims
}
```

**Extraction Service** (LLM):

```protobuf
service ExtractionService {
  rpc ExtractPICO(ExtractPICORequest) returns (ExtractPICOResponse);
  rpc ExtractAdverseEvents(ExtractAERequest) returns (ExtractAEResponse);
  // ... other extraction types
}

message ExtractPICORequest {
  string document_id = 1;
  repeated Block blocks = 2;  // Context
  ExtractionOptions options = 3;
}

message ExtractPICOResponse {
  Population population = 1;
  repeated Intervention interventions = 2;
  Comparison comparison = 3;
  repeated Outcome outcomes = 4;
  float confidence = 5;
}
```

#### 4.3.3 Fail-Fast Implementation

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

---

## 5. Data Flow & Pipelines

### 5.1 Ingestion Pipeline Architecture

#### 5.1.1 Auto-Pipeline (Fast Sources)

**Flow**:

```
1. Adapter.fetch() → Raw JSON
2. Adapter.parse() → Document IR
3. Adapter.validate() → Schema check
4. Adapter.write() → Storage + Ledger
5. Orchestrator → Detect completion
6. ChunkingService → Semantic chunks
7. EmbeddingService → SPLADE + Dense vectors
8. IndexingService → OpenSearch + FAISS
9. KGService → Extract entities, create graph
10. Orchestrator → Emit ingest.results.v1
```

**Use Cases**: Structured APIs (ClinicalTrials.gov, OpenFDA drug labels, OpenAlex metadata)

**Ledger States**: `queued` → `fetching` → `parsing` → `validating` → `writing` → `chunking` → `embedding` → `indexing` → `kg_writing` → `completed`

#### 5.1.2 Two-Phase Pipeline (PDF/GPU-Bound)

**Phase 1 - Metadata**:

```
1. OpenAlexAdapter → Fetch paper metadata
2. Write preliminary Document (title, authors, abstract)
3. Check is_oa (open access) flag
4. If is_oa: Trigger Phase 2
```

**Phase 2 - Full Text**:

```
5. UnpaywallAdapter → Get PDF URL
6. If no URL: COREAdapter → Try PDF access
7. Download PDF → MinIO object storage
8. Ledger → Mark pdf_downloaded
9. MinerU Worker → Consume Kafka message
10. Call MinerU gRPC → Parse PDF
11. MinerU → Return structured blocks
12. Merge with Phase 1 metadata
13. Continue to chunking/embedding/indexing
```

**Use Cases**: Academic papers, clinical trial documents, drug label PDFs

**Ledger States**: `metadata_fetched` → `pdf_pending` → `pdf_downloaded` → `pdf_parsing` → `pdf_parsed` → `postpdf_chunking` → ... → `completed`

**Rationale**: Separates fast metadata fetch from slow GPU processing. If PDF fetch fails, we still have metadata. Can retry PDF independently.

### 5.2 Multi-Adapter Chaining

#### 5.2.1 Literature Enrichment Workflow

```
┌──────────────┐   Query    ┌──────────────┐
│ OpenAlex     │────────────>│ Metadata +   │
│ Adapter      │<────────────│ DOI + OA URL │
└──────────────┘   Papers    └──────┬───────┘
                                     │
                              For each OA paper
                                     │
┌──────────────┐   DOI      ┌───────▼───────┐
│ Unpaywall    │────────────>│ PDF URL       │
│ Adapter      │<────────────│ (if available)│
└──────────────┘             └──────┬────────┘
                                     │
                              If no PDF URL
                                     │
┌──────────────┐   DOI      ┌───────▼────────┐
│ CORE         │────────────>│ PDF URL or     │
│ Adapter      │<────────────│ Repository Link│
└──────────────┘             └──────┬─────────┘
                                     │
┌──────────────┐  Download  ┌───────▼─────────┐
│ HTTP         │────────────>│ PDF Bytes       │
│ Downloader   │<────────────│                 │
└──────────────┘             └──────┬──────────┘
                                     │
┌──────────────┐   Parse    ┌───────▼──────────┐
│ MinerU       │────────────>│ Structured       │
│ Service      │<────────────│ Blocks + Tables  │
└──────────────┘             └──────────────────┘
```

**Orchestration Logic**:

```python
async def enrich_literature(query: str):
    # Step 1: Get metadata
    papers = await openalex_adapter.search(query)

    for paper in papers:
        doc_id = await openalex_adapter.write(paper)

        # Step 2: If open access, get full text
        if paper.is_oa:
            pdf_url = await unpaywall_adapter.get_pdf_url(paper.doi)

            if not pdf_url:
                pdf_url = await core_adapter.get_pdf_url(paper.doi)

            if pdf_url:
                # Step 3: Download and parse
                pdf_bytes = await download_pdf(pdf_url)
                await storage.write_pdf(doc_id, pdf_bytes)

                # Step 4: Trigger GPU pipeline
                await kafka.publish(
                    "ingest.requests.v1",
                    {"doc_id": doc_id, "type": "pdf_parse"}
                )
```

#### 5.2.2 Ontology Mapping Workflow

**After initial ingestion**:

```
1. KGService extracts entity mentions (drugs, diseases)
2. Publish to mapping.events.v1 topic
3. MappingWorker consumes events
4. For each drug mention:
   - RxNormAdapter → Get RxCUI
   - Update Entity node with rxcui property
5. For each disease mention:
   - ICD11Adapter → Get ICD-11 code
   - Update Entity node with icd11_code property
6. Link Entity to standardized ontology nodes
```

**Example**:

```
Mention: "atorvastatin 20mg"
├─> RxNormAdapter.normalize("atorvastatin") → RxCUI:83367
├─> Create/Merge Entity node:
│   {name: "atorvastatin", rxcui: "83367", type: "drug"}
└─> Link to Document with relationship: MENTIONED_IN(span=[45, 62])
```

### 5.3 Idempotency & Fault Tolerance

#### 5.3.1 Ledger-Based State Tracking

**Schema**:

```python
class JobLedger:
    job_id: str  # UUID
    doc_key: str  # Source identifier (NCT ID, DOI, etc.)
    tenant_id: str
    status: JobStatus  # queued, processing, completed, failed
    stage: str  # Current pipeline stage
    retries: int
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]
```

**Operations**:

```python
# Check if already processed
existing = await ledger.get_by_doc_key(doc_key, tenant_id)
if existing and existing.status == "completed":
    return existing.doc_id  # Skip, already done

# Create job
job_id = await ledger.create(doc_key, tenant_id, stage="queued")

# Update stage
await ledger.update_stage(job_id, stage="pdf_parsing")

# Mark complete
await ledger.mark_complete(job_id, doc_id=result_doc_id)

# Handle failure
await ledger.mark_failed(job_id, error="GPU OOM", retryable=True)
```

#### 5.3.2 Retry Logic

**Transient Failures** (retry):

- Network timeouts
- Rate limit exceeded (429)
- Temporary service unavailability (503)
- GPU out of memory (with smaller batch)

**Permanent Failures** (don't retry):

- Invalid ID format (400)
- Not found (404)
- Authentication failure (401)
- Schema validation error

**Implementation**:

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    reraise=True
)
async def fetch_with_retry(adapter, identifier):
    return await adapter.fetch(identifier)
```

#### 5.3.3 Dead Letter Queue

**Failed Jobs** → Moved to DLQ after max retries:

```
ingest.requests.v1 (main topic)
    ↓ (failed 3x)
ingest.dlq.v1 (dead letter queue)
```

**DLQ Processing**:

- Manual inspection
- Fix underlying issue (e.g., update adapter config)
- Re-publish to main topic

---

## 6. Multi-Protocol API Design

### 6.1 Protocol Selection Matrix

| Protocol | Use Case | Clients | Pros | Cons |
|----------|----------|---------|------|------|
| **REST** | Web apps, scripts | Broad compatibility | Simple, cacheable, stateless | Over-fetching, multiple round-trips |
| **GraphQL** | Rich UIs, mobile | React, Vue, iOS, Android | Precise queries, single round-trip | Complexity, caching harder |
| **gRPC** | Microservices | Internal services | High performance, streaming | Binary format, limited browser support |
| **SOAP** | Legacy enterprise | Old systems | WSDL contract, tooling | Verbose XML, outdated |
| **AsyncAPI/SSE** | Real-time updates | Web dashboards | Server push, simple | Unidirectional, limited by HTTP |

### 6.2 REST API Design

#### 6.2.1 OpenAPI 3.1 Specification

**Structure**:

```yaml
openapi: 3.1.0
info:
  title: Medical_KG_rev API
  version: 1.0.0
  description: Multi-protocol biomedical knowledge integration

servers:
  - url: https://api.medical-kg.example.com/v1
    description: Production

security:
  - OAuth2:
    - ingest:write
    - kg:read

paths:
  /ingest/clinicaltrials:
    post:
      summary: Ingest clinical trials by NCT IDs
      operationId: ingestClinicalTrials
      security:
        - OAuth2: [ingest:write]
      requestBody:
        required: true
        content:
          application/vnd.api+json:
            schema:
              $ref: '#/components/schemas/IngestClinicalTrialsRequest'
      responses:
        '200':
          description: Successful ingestion
          content:
            application/vnd.api+json:
              schema:
                $ref: '#/components/schemas/IngestResponse'
        '207':
          description: Multi-Status (partial success)
          content:
            application/vnd.api+json:
              schema:
                $ref: '#/components/schemas/IngestBatchResponse'
        '400':
          description: Validation error
          content:
            application/problem+json:
              schema:
                $ref: '#/components/schemas/ProblemDetails'
```

#### 6.2.2 JSON:API Compliance

**Resource Object Structure**:

```json
{
  "data": {
    "type": "Document",
    "id": "clinicaltrials:NCT04267848",
    "attributes": {
      "title": "Study of Pembrolizumab in Melanoma",
      "status": "recruiting",
      "createdAt": "2024-01-15T10:30:00Z"
    },
    "relationships": {
      "organization": {
        "data": {"type": "Organization", "id": "nci"}
      },
      "claims": {
        "data": [
          {"type": "Claim", "id": "claim-001"},
          {"type": "Claim", "id": "claim-002"}
        ]
      }
    },
    "links": {
      "self": "/documents/clinicaltrials:NCT04267848"
    }
  },
  "included": [
    {
      "type": "Organization",
      "id": "nci",
      "attributes": {
        "name": "National Cancer Institute"
      }
    }
  ],
  "meta": {
    "total": 1,
    "processingTime": "234ms"
  }
}
```

**Pagination**:

```
GET /documents?page[limit]=25&page[offset]=50
```

Response includes:

```json
{
  "links": {
    "first": "/documents?page[limit]=25&page[offset]=0",
    "prev": "/documents?page[limit]=25&page[offset]=25",
    "next": "/documents?page[limit]=25&page[offset]=75",
    "last": "/documents?page[limit]=25&page[offset]=975"
  },
  "meta": {
    "total": 1000
  }
}
```

#### 6.2.3 OData Query Support

**Filtering**:

```
GET /documents?$filter=status eq 'active' and year gt 2020
```

**Field Selection**:

```
GET /documents?$select=title,status,createdAt
```

**Expansion** (include relationships):

```
GET /documents?$expand=organization,claims
```

**Sorting**:

```
GET /documents?$orderby=createdAt desc,title asc
```

**Pagination**:

```
GET /documents?$top=50&$skip=100
```

**Combined**:

```
GET /documents?$filter=status eq 'recruiting'&$select=title,nctId&$orderby=startDate desc&$top=20
```

### 6.3 GraphQL API Design

#### 6.3.1 Schema Definition

```graphql
type Query {
  document(id: ID!): Document
  documents(
    filter: DocumentFilter
    orderBy: DocumentOrderBy
    limit: Int = 25
    offset: Int = 0
  ): DocumentConnection!

  search(
    query: String!
    filters: SearchFilters
    limit: Int = 10
  ): [SearchResult!]!

  organization(id: ID!): Organization
}

type Mutation {
  startIngestion(input: StartIngestionInput!): IngestionJob!
  chunkDocument(documentId: ID!): ChunkResult!
  embedChunks(chunkIds: [ID!]!): EmbedResult!
}

type Document {
  id: ID!
  title: String!
  content: String
  status: DocumentStatus!
  createdAt: DateTime!

  # Relationships (resolved via DataLoader)
  organization: Organization
  claims(filter: ClaimFilter): [Claim!]!
  entities: [Entity!]!
}

type Organization {
  id: ID!
  name: String!
  tenantId: String!
  documents(limit: Int): [Document!]!
}

type Claim {
  id: ID!
  subject: Entity!
  predicate: String!
  object: Entity!
  confidence: Float!
  evidenceSpans: [Span!]!
}

input DocumentFilter {
  status: DocumentStatus
  createdAfter: DateTime
  organizationId: ID
  search: String
}

enum DocumentStatus {
  DRAFT
  PUBLISHED
  RETRACTED
}
```

#### 6.3.2 DataLoader Pattern (N+1 Prevention)

**Problem**: Without batching, fetching 10 documents with their organizations makes 11 queries (1 for documents, 10 for organizations).

**Solution**: DataLoader batches requests:

```python
class OrganizationLoader:
    async def batch_load_fn(self, org_ids: list[str]) -> list[Organization]:
        # Single query for all org_ids
        orgs = await db.get_organizations(org_ids)
        return [org_map[org_id] for org_id in org_ids]

organization_loader = DataLoader(batch_load_fn)

# In resolver
async def resolve_organization(document, info):
    return await info.context.organization_loader.load(document.org_id)
```

**Result**: 10 documents → 2 queries total (1 for documents, 1 batched for organizations).

#### 6.3.3 Example Queries

**Single resource with relationships**:

```graphql
query GetDocument {
  document(id: "clinicaltrials:NCT04267848") {
    title
    status
    organization {
      name
    }
    claims(filter: {type: EFFICACY}) {
      subject {
        name
      }
      predicate
      object {
        name
      }
      confidence
    }
  }
}
```

**Search with filters**:

```graphql
query SearchTrials {
  search(
    query: "pembrolizumab melanoma"
    filters: {
      documentType: CLINICAL_TRIAL
      status: RECRUITING
      phase: PHASE3
    }
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

**Mutation**:

```graphql
mutation IngestTrial {
  startIngestion(input: {
    source: CLINICALTRIALS
    identifiers: ["NCT04267848"]
  }) {
    jobId
    status
    documentsQueued
  }
}
```

### 6.4 gRPC Service Design

#### 6.4.1 Service Definitions

**Ingestion Service**:

```protobuf
service IngestionService {
  // Start ingestion job
  rpc StartIngest(StartIngestRequest) returns (IngestJobResponse);

  // Get job status
  rpc GetJobStatus(JobStatusRequest) returns (JobStatusResponse);

  // Stream job events
  rpc StreamJobEvents(JobEventsRequest) returns (stream JobEvent);
}

message StartIngestRequest {
  string source = 1;  // "clinicaltrials", "openalex", etc.
  repeated string identifiers = 2;
  string tenant_id = 3;
  map<string, string> options = 4;
}

message IngestJobResponse {
  string job_id = 1;
  JobStatus status = 2;
  int32 documents_queued = 3;
}

enum JobStatus {
  QUEUED = 0;
  PROCESSING = 1;
  COMPLETED = 2;
  FAILED = 3;
}
```

**Bidirectional Streaming** (for large batch uploads):

```protobuf
service BulkIngestionService {
  rpc BulkIngest(stream IngestChunk) returns (stream IngestResult);
}

message IngestChunk {
  string job_id = 1;
  int32 chunk_index = 2;
  bytes data = 3;  // Part of large file
}

message IngestResult {
  string document_id = 1;
  Status status = 2;
}
```

#### 6.4.2 Error Handling

```python
# Server-side
class IngestionServicer(IngestionServiceServicer):
    async def StartIngest(self, request, context):
        try:
            # Validate
            if not request.identifiers:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "No identifiers provided"
                )

            # Process
            job_id = await orchestrator.start_ingest(
                source=request.source,
                identifiers=request.identifiers,
                tenant_id=request.tenant_id
            )

            return IngestJobResponse(
                job_id=job_id,
                status=JobStatus.QUEUED,
                documents_queued=len(request.identifiers)
            )

        except PermissionError:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "Insufficient permissions")
        except Exception as e:
            logger.exception("Ingestion failed")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

# Client-side
async with grpc.aio.insecure_channel("localhost:50051") as channel:
    stub = IngestionServiceStub(channel)
    try:
        response = await stub.StartIngest(
            StartIngestRequest(
                source="clinicaltrials",
                identifiers=["NCT04267848"],
                tenant_id="tenant-123"
            )
        )
        print(f"Job ID: {response.job_id}")
    except grpc.RpcError as e:
        print(f"Error {e.code()}: {e.details()}")
```

### 6.5 AsyncAPI & Server-Sent Events

#### 6.5.1 AsyncAPI Specification

```yaml
asyncapi: 3.0.0
info:
  title: Medical_KG_rev Event API
  version: 1.0.0

channels:
  jobs/{jobId}/events:
    address: /jobs/{jobId}/events
    messages:
      jobStarted:
        $ref: '#/components/messages/JobStarted'
      jobProgress:
        $ref: '#/components/messages/JobProgress'
      jobCompleted:
        $ref: '#/components/messages/JobCompleted'
      jobFailed:
        $ref: '#/components/messages/JobFailed'
    parameters:
      jobId:
        description: Unique job identifier
        schema:
          type: string

components:
  messages:
    JobStarted:
      payload:
        type: object
        properties:
          jobId:
            type: string
          source:
            type: string
          identifiers:
            type: array
            items:
              type: string
          timestamp:
            type: string
            format: date-time

    JobProgress:
      payload:
        type: object
        properties:
          jobId:
            type: string
          stage:
            type: string
            enum: [fetching, parsing, chunking, embedding, indexing]
          progress:
            type: number
            minimum: 0
            maximum: 100
          message:
            type: string

    JobCompleted:
      payload:
        type: object
        properties:
          jobId:
            type: string
          documentsCreated:
            type: integer
          duration:
            type: number
          summary:
            type: object
```

#### 6.5.2 SSE Implementation

**Server**:

```python
from fastapi.responses import StreamingResponse

@app.get("/jobs/{job_id}/events")
async def stream_job_events(job_id: str):
    async def event_generator():
        async with redis.pubsub() as pubsub:
            await pubsub.subscribe(f"job:{job_id}")

            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    yield f"event: {data['type']}\n"
                    yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

**Client** (JavaScript):

```javascript
const eventSource = new EventSource(`/jobs/${jobId}/events`);

eventSource.addEventListener('jobs.started', (e) => {
  const data = JSON.parse(e.data);
  console.log('Job started:', data);
});

eventSource.addEventListener('jobs.progress', (e) => {
  const data = JSON.parse(e.data);
  updateProgressBar(data.progress);
});

eventSource.addEventListener('jobs.completed', (e) => {
  const data = JSON.parse(e.data);
  console.log('Job completed:', data);
  eventSource.close();
});

eventSource.addEventListener('error', (e) => {
  console.error('SSE error:', e);
});
```

---

*[Document continues with sections 7-14: Adapter SDK, Knowledge Graph, Retrieval, Security, Observability, Deployment, Design Decisions, and Future Considerations. Due to length, providing a comprehensive outline for the remaining sections.]*

## 7. Adapter SDK & Extensibility

### 7.1 Adapter Lifecycle

The Adapter SDK provides a plug-in architecture for integrating external data sources. Each adapter follows a standardized lifecycle:

```
fetch() → parse() → validate() → write()
```

#### 7.1.1 BaseAdapter Interface

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

    def validate(self, documents: list[Document]) -> list[Document]:
        """Validate documents against schema."""
        return [Document.model_validate(doc) for doc in documents]

    async def write(self, documents: list[Document]) -> list[str]:
        """Write documents to storage."""
        doc_ids = []
        for doc in documents:
            doc_id = await storage.write_document(doc)
            await ledger.mark_complete(doc_id)
            doc_ids.append(doc_id)
        return doc_ids
```

#### 7.1.2 YAML-Based Adapters

For simple REST APIs, adapters can be defined declaratively in YAML:

```yaml
# src/Medical_KG_rev/adapters/config/clinicaltrials.yaml
name: clinicaltrials
base_url: https://clinicaltrials.gov/api/v2
rate_limit:
  requests_per_second: 10
  burst: 20
auth:
  type: none
endpoints:
  get_study:
    method: GET
    path: /studies/{nct_id}
    params:
      format: json
    response_mapping:
      doc_id: "clinicaltrials:{nct_id}"
      title: "$.protocolSection.identificationModule.briefTitle"
      content: "$.protocolSection"
      metadata:
        nct_id: "$.protocolSection.identificationModule.nctId"
        phase: "$.protocolSection.designModule.phases[0]"
        status: "$.protocolSection.statusModule.overallStatus"
```

The YAML config is automatically converted to a functional adapter via `RESTAdapter` base class.

#### 7.1.3 Python-Based Adapters

For complex sources (SOAP APIs, special authentication, custom parsing), implement Python adapter:

```python
class OpenAlexAdapter(BaseAdapter):
    """Adapter for OpenAlex scholarly works API."""

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.base_url = "https://api.openalex.org"
        self.polite_pool = config.email  # Polite pool requires email

    async def fetch(self, **params: Any) -> list[dict[str, Any]]:
        """Fetch works from OpenAlex."""
        query = params.get("query")
        per_page = params.get("per_page", 25)

        await self.rate_limiter.acquire()

        headers = {
            "User-Agent": f"Medical_KG_rev/0.1.0 (mailto:{self.polite_pool})"
        }

        async with self.http_client.get(
            f"{self.base_url}/works",
            params={"search": query, "per_page": per_page},
            headers=headers,
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data["results"]

    async def parse(self, raw_data: list[dict[str, Any]]) -> list[Document]:
        """Parse OpenAlex JSON to Document IR."""
        documents = []
        for work in raw_data:
            doc = Document(
                doc_id=f"openalex:{work['id'].split('/')[-1]}",
                title=work["title"],
                content=work.get("abstract"),
                source="openalex",
                metadata={
                    "doi": work.get("doi"),
                    "publication_year": work.get("publication_year"),
                    "cited_by_count": work.get("cited_by_count"),
                    "is_oa": work.get("open_access", {}).get("is_oa", False),
                    "authors": [a["author"]["display_name"] for a in work.get("authorships", [])],
                },
                domain_type="medical",
                domain_data=MedicalDomain(
                    resource_type="Evidence",
                    doi=work.get("doi"),
                    pmcid=self._extract_pmcid(work),
                    publication_date=work.get("publication_date"),
                ),
            )
            documents.append(doc)
        return documents
```

### 7.2 Rate Limiting Strategies

#### 7.2.1 Token Bucket Algorithm

The rate limiter uses a token bucket algorithm for smooth rate limiting:

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

#### 7.2.2 Per-Source Rate Limits

Each adapter has independent rate limits configured in YAML or constructor:

| Source | Requests/Second | Burst | Notes |
|--------|----------------|-------|-------|
| ClinicalTrials.gov | 10 | 20 | No API key required |
| OpenAlex | 10 | 20 | Polite pool (100k/day) |
| OpenFDA | 5 | 10 | 1000/day without key |
| Unpaywall | 5 | 10 | Requires email in User-Agent |
| Crossref | 50 | 100 | "Plus" service |
| Europe PMC | 10 | 20 | SOAP and REST |
| Semantic Scholar | 10 | 20 | API key required |
| CORE | 5 | 10 | API key for PDF access |
| ChEMBL | 5 | 10 | Rate limited |
| ICD-11 WHO | 10 | 20 | OAuth token required |
| RxNorm | 20 | 40 | Public API |

### 7.3 Authentication Patterns

#### 7.3.1 No Authentication

Most biomedical APIs are public (ClinicalTrials.gov, OpenFDA, RxNorm):

```yaml
auth:
  type: none
```

#### 7.3.2 API Key (Header)

```yaml
auth:
  type: api_key
  location: header
  name: X-API-Key
  value_from_env: SEMANTIC_SCHOLAR_API_KEY
```

```python
headers = {
    "X-API-Key": os.getenv("SEMANTIC_SCHOLAR_API_KEY")
}
```

#### 7.3.3 OAuth 2.0 (ICD-11 WHO API)

```python
class ICD11Adapter(BaseAdapter):
    """Adapter for WHO ICD-11 API with OAuth."""

    async def _get_access_token(self) -> str:
        """Get OAuth access token."""
        async with self.http_client.post(
            "https://icdaccessmanagement.who.int/connect/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "scope": "icdapi_access",
            },
        ) as response:
            data = await response.json()
            return data["access_token"]

    async def fetch(self, **params: Any) -> list[dict[str, Any]]:
        token = await self._get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "API-Version": "v2",
            "Accept-Language": "en",
        }
        # ... fetch with headers
```

#### 7.3.4 Polite Pool (Email in User-Agent)

OpenAlex, Unpaywall, and Crossref require email in User-Agent for higher rate limits:

```python
headers = {
    "User-Agent": f"Medical_KG_rev/0.1.0 (mailto:{config.email})"
}
```

### 7.4 Error Handling & Retry

#### 7.4.1 Retry Strategy

Use `tenacity` library for exponential backoff with jitter:

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, RateLimitError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def fetch_with_retry(adapter: BaseAdapter, **params: Any) -> list[dict[str, Any]]:
    """Fetch with automatic retry on transient failures."""
    return await adapter.fetch(**params)
```

#### 7.4.2 Error Classification

**Transient Errors (Retry)**:

- Network timeouts (408, 504)
- Rate limit exceeded (429)
- Service unavailable (503)
- Connection errors

**Permanent Errors (Don't Retry)**:

- Invalid ID format (400)
- Not found (404)
- Unauthorized (401, 403)
- Schema validation errors

```python
class AdapterError(Exception):
    """Base adapter error."""
    retryable: bool = False

class RateLimitError(AdapterError):
    """Rate limit exceeded."""
    retryable = True

class NotFoundError(AdapterError):
    """Resource not found."""
    retryable = False

class ValidationError(AdapterError):
    """Schema validation failed."""
    retryable = False
```

#### 7.4.3 Dead Letter Queue

Failed jobs after max retries are moved to DLQ for manual inspection:

```python
async def process_ingest_request(message: dict) -> None:
    """Process ingestion request with DLQ fallback."""
    try:
        adapter = get_adapter(message["adapter"])
        documents = await fetch_with_retry(adapter, **message["params"])
        await adapter.write(documents)
    except Exception as e:
        if message["retry_count"] >= 3:
            # Move to DLQ
            await kafka.send("ingest.dlq.v1", message)
            logger.error(f"Job {message['job_id']} moved to DLQ: {e}")
        else:
            # Retry
            message["retry_count"] += 1
            await kafka.send("ingest.requests.v1", message)
```

### 7.5 Testing Adapters

#### 7.5.1 Unit Tests with Mocked Responses

```python
from aioresponses import aioresponses

@pytest.mark.asyncio
async def test_clinicaltrials_adapter_fetch():
    """Test ClinicalTrials adapter fetches study data."""
    adapter = ClinicalTrialsAdapter(AdapterConfig(name="clinicaltrials"))

    with aioresponses() as m:
        m.get(
            "https://clinicaltrials.gov/api/v2/studies/NCT04267848",
            payload=MOCK_CLINICALTRIALS_RESPONSE,
            status=200,
        )

        results = await adapter.fetch(nct_id="NCT04267848")

        assert len(results) == 1
        assert results[0]["protocolSection"]["identificationModule"]["nctId"] == "NCT04267848"

@pytest.mark.asyncio
async def test_clinicaltrials_adapter_parse():
    """Test ClinicalTrials adapter parses to Document IR."""
    adapter = ClinicalTrialsAdapter(AdapterConfig(name="clinicaltrials"))

    documents = await adapter.parse([MOCK_CLINICALTRIALS_RESPONSE])

    assert len(documents) == 1
    assert documents[0].doc_id == "clinicaltrials:NCT04267848"
    assert documents[0].source == "clinicaltrials"
    assert documents[0].metadata["phase"] == "PHASE3"
```

#### 7.5.2 Integration Tests with Real APIs

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_openalex_adapter_real_api():
    """Test OpenAlex adapter with real API (requires network)."""
    adapter = OpenAlexAdapter(AdapterConfig(
        name="openalex",
        email="test@example.com",
    ))

    results = await adapter.fetch(query="SGLT2 inhibitors", per_page=5)

    assert len(results) <= 5
    assert all("title" in work for work in results)
    assert all("id" in work for work in results)
```

#### 7.5.3 Rate Limit Tests

```python
@pytest.mark.asyncio
async def test_rate_limiter_enforces_limit():
    """Test rate limiter enforces requests per second."""
    limiter = RateLimiter(requests_per_second=10, burst=10)

    start = time.monotonic()

    # Consume burst
    for _ in range(10):
        await limiter.acquire()

    # Next request should wait
    await limiter.acquire()

    elapsed = time.monotonic() - start
    assert elapsed >= 0.1  # Should wait at least 100ms for 11th request
```

## 8. Knowledge Graph Schema

### 8.1 Neo4j Data Model

The knowledge graph uses Neo4j 5.x with a FHIR-aligned schema for medical domain entities.

#### 8.1.1 Core Node Types

```cypher
// Document node
CREATE (d:Document {
  doc_id: "clinicaltrials:NCT04267848",
  title: "Study of Pembrolizumab in Melanoma",
  source: "clinicaltrials",
  tenant_id: "tenant-123",
  created_at: datetime(),
  metadata: {...}
})

// Entity node (normalized)
CREATE (e:Entity:Drug {
  entity_id: "RxCUI:83367",
  name: "atorvastatin",
  canonical_name: "Atorvastatin",
  ontology: "rxnorm",
  rxcui: "83367",
  tenant_id: "tenant-123"
})

// Claim node (extracted fact)
CREATE (c:Claim {
  claim_id: "claim-001",
  subject_id: "RxCUI:83367",
  predicate: "treats",
  object_id: "ICD11:BA00",
  confidence: 0.92,
  tenant_id: "tenant-123"
})

// ExtractionActivity node (provenance)
CREATE (a:ExtractionActivity {
  activity_id: "activity-001",
  method: "llm",
  model_name: "gpt-4-2024-01",
  prompt_version: "v1.2",
  timestamp: datetime(),
  tenant_id: "tenant-123"
})
```

#### 8.1.2 Relationship Types

```cypher
// Document mentions entity
(d:Document)-[:MENTIONS {span_start: 45, span_end: 62, span_text: "atorvastatin 20mg"}]->(e:Entity)

// Claim connects entities
(c:Claim)-[:HAS_SUBJECT]->(e1:Entity)
(c:Claim)-[:HAS_OBJECT]->(e2:Entity)

// Provenance links
(a:ExtractionActivity)-[:EXTRACTED]->(c:Claim)
(a:ExtractionActivity)-[:FROM_DOCUMENT]->(d:Document)

// Organization relationships
(d:Document)-[:SPONSORED_BY]->(o:Organization)
(d:Document)-[:AUTHORED_BY]->(p:Person)
```

#### 8.1.3 Indexes and Constraints

```cypher
// Unique constraints
CREATE CONSTRAINT document_id_unique FOR (d:Document) REQUIRE d.doc_id IS UNIQUE;
CREATE CONSTRAINT entity_id_unique FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;
CREATE CONSTRAINT claim_id_unique FOR (c:Claim) REQUIRE c.claim_id IS UNIQUE;

// Composite indexes for multi-tenancy
CREATE INDEX document_tenant_idx FOR (d:Document) ON (d.tenant_id, d.doc_id);
CREATE INDEX entity_tenant_idx FOR (e:Entity) ON (d.tenant_id, e.entity_id);

// Full-text search indexes
CREATE FULLTEXT INDEX document_content FOR (d:Document) ON EACH [d.title, d.content];
CREATE FULLTEXT INDEX entity_name FOR (e:Entity) ON EACH [e.name, e.canonical_name];
```

### 8.2 SHACL Validation

SHACL (Shapes Constraint Language) validates graph structure and data quality.

#### 8.2.1 Shape Definitions

```turtle
# src/Medical_KG_rev/kg/shapes.ttl

@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix medkg: <http://medical-kg.example.com/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# Document Shape
medkg:DocumentShape a sh:NodeShape ;
    sh:targetClass medkg:Document ;
    sh:property [
        sh:path medkg:doc_id ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:datatype xsd:string ;
        sh:pattern "^[a-z]+:[A-Za-z0-9_-]+" ;
    ] ;
    sh:property [
        sh:path medkg:title ;
        sh:minCount 1 ;
        sh:datatype xsd:string ;
        sh:minLength 1 ;
    ] ;
    sh:property [
        sh:path medkg:tenant_id ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:datatype xsd:string ;
    ] .

# Claim Shape
medkg:ClaimShape a sh:NodeShape ;
    sh:targetClass medkg:Claim ;
    sh:property [
        sh:path medkg:confidence ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:datatype xsd:float ;
        sh:minInclusive 0.0 ;
        sh:maxInclusive 1.0 ;
    ] ;
    sh:property [
        sh:path medkg:hasSubject ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class medkg:Entity ;
    ] ;
    sh:property [
        sh:path medkg:hasObject ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class medkg:Entity ;
    ] .
```

#### 8.2.2 Validation Implementation

```python
from pyshacl import validate
import rdflib

class ShaclValidator:
    """Validates Neo4j graph against SHACL shapes."""

    def __init__(self, shapes_file: str = "shapes.ttl"):
        self.shapes_graph = rdflib.Graph()
        self.shapes_graph.parse(shapes_file, format="turtle")

    def validate_graph(self, data_graph: rdflib.Graph) -> tuple[bool, str]:
        """Validate data graph against shapes."""
        conforms, results_graph, results_text = validate(
            data_graph,
            shacl_graph=self.shapes_graph,
            inference="rdfs",
            abort_on_first=False,
        )
        return conforms, results_text
```

### 8.3 Cypher Query Patterns

#### 8.3.1 Multi-Tenant Query Pattern

**Always filter by tenant_id**:

```cypher
// CORRECT - with tenant filter
MATCH (d:Document)
WHERE d.doc_id = $doc_id AND d.tenant_id = $tenant_id
RETURN d

// WRONG - no tenant filter (security vulnerability)
MATCH (d:Document)
WHERE d.doc_id = $doc_id
RETURN d
```

#### 8.3.2 Idempotent MERGE Pattern

```cypher
// Create or update entity (idempotent)
MERGE (e:Entity {entity_id: $entity_id, tenant_id: $tenant_id})
ON CREATE SET
    e.name = $name,
    e.canonical_name = $canonical_name,
    e.created_at = datetime()
ON MATCH SET
    e.updated_at = datetime()
RETURN e
```

#### 8.3.3 Provenance Query

```cypher
// Find all claims about a drug with provenance
MATCH (drug:Entity {entity_id: $drug_id, tenant_id: $tenant_id})
MATCH (claim:Claim)-[:HAS_SUBJECT]->(drug)
MATCH (activity:ExtractionActivity)-[:EXTRACTED]->(claim)
MATCH (activity)-[:FROM_DOCUMENT]->(doc:Document)
RETURN
    claim.predicate as relationship,
    claim.object_id as target,
    claim.confidence as confidence,
    activity.model_name as model,
    doc.title as source_document,
    doc.doc_id as source_id
ORDER BY claim.confidence DESC
```

#### 8.3.4 Graph Traversal

```cypher
// Find drugs that treat diseases similar to a given disease
MATCH (disease1:Entity {entity_id: $disease_id, tenant_id: $tenant_id})
MATCH (disease1)<-[:HAS_OBJECT]-(claim1:Claim {predicate: "treats"})
MATCH (claim1)-[:HAS_SUBJECT]->(drug:Entity)
MATCH (drug)-[:HAS_SUBJECT]-(claim2:Claim {predicate: "treats"})
MATCH (claim2)-[:HAS_OBJECT]->(disease2:Entity)
WHERE disease2.entity_id <> $disease_id
RETURN
    drug.name as drug_name,
    disease2.name as related_disease,
    avg(claim2.confidence) as avg_confidence,
    count(claim2) as evidence_count
ORDER BY avg_confidence DESC, evidence_count DESC
LIMIT 10
```

### 8.4 Provenance Tracking

Every extracted fact includes complete provenance chain:

#### 8.4.1 Provenance Model

```
Document → ExtractionActivity → Claim → Entity
    ↓              ↓                ↓
  source      model+prompt    span grounding
```

#### 8.4.2 Provenance Query Example

```python
async def get_claim_provenance(claim_id: str, tenant_id: str) -> dict:
    """Get complete provenance for a claim."""
    query = """
    MATCH (claim:Claim {claim_id: $claim_id, tenant_id: $tenant_id})
    MATCH (activity:ExtractionActivity)-[:EXTRACTED]->(claim)
    MATCH (activity)-[:FROM_DOCUMENT]->(doc:Document)
    MATCH (claim)-[:HAS_SUBJECT]->(subject:Entity)
    MATCH (claim)-[:HAS_OBJECT]->(object:Entity)
    OPTIONAL MATCH (doc)-[:MENTIONS {claim_id: $claim_id}]->(subject)
    RETURN {
        claim: {
            id: claim.claim_id,
            predicate: claim.predicate,
            confidence: claim.confidence,
            subject: subject.name,
            object: object.name
        },
        extraction: {
            method: activity.method,
            model: activity.model_name,
            prompt_version: activity.prompt_version,
            timestamp: activity.timestamp
        },
        source: {
            doc_id: doc.doc_id,
            title: doc.title,
            source: doc.source,
            url: doc.metadata.url
        },
        evidence: {
            span_start: span.span_start,
            span_end: span.span_end,
            span_text: span.span_text
        }
    } as provenance
    """

    result = await neo4j_client.execute(query, claim_id=claim_id, tenant_id=tenant_id)
    return result[0]["provenance"]
```

---

## 9. Retrieval Architecture

- 9.1 BM25 (Lexical)
- 9.2 SPLADE (Learned Sparse)
- 9.3 Dense Vectors (Qwen-3)
- 9.4 Fusion Ranking (RRF)
- 9.5 Reranking

## 10. Security & Multi-Tenancy

- 10.1 OAuth 2.0 Flow
- 10.2 JWT Validation
- 10.3 Scope Enforcement
- 10.4 Tenant Isolation
- 10.5 Rate Limiting
- 10.6 Audit Logging

## 11. Observability & Operations

- 11.1 Prometheus Metrics
- 11.2 OpenTelemetry Tracing
- 11.3 Structured Logging
- 11.4 Grafana Dashboards
- 11.5 Alerting

## 12. Deployment Architecture

- 12.1 Docker Compose (Dev)
- 12.2 Kubernetes (Prod)
- 12.3 GPU Node Management
- 12.4 Scaling Strategies

## 13. Design Decisions & Trade-offs

- 13.1 Why FastAPI vs Flask/Django
- 13.2 Why Neo4j vs PostgreSQL
- 13.3 Why Kafka vs RabbitMQ
- 13.4 Why gRPC for GPU services
- 13.5 Why Multiple Retrieval Strategies

## 14. Future Considerations

- 14.1 GraphQL Federation
- 14.2 Multi-Region Deployment
- 14.3 FHIR Server Integration
- 14.4 Real-time Collaboration
- 14.5 Federated Learning

---

## Appendix A: API Examples

### A.1 REST API Workflows

### A.2 GraphQL Queries

### A.3 gRPC Calls

### A.4 SSE Streams

## Appendix B: Performance Benchmarks

### B.1 Ingestion Throughput

### B.2 Retrieval Latency

### B.3 GPU Service Performance

### B.4 Scalability Tests

## Appendix C: Security Audit

### C.1 OWASP Top 10 Compliance

### C.2 Penetration Test Results

### C.3 Vulnerability Scanning

## Appendix D: References

- HL7 FHIR Specification
- OpenAPI 3.1 Specification
- GraphQL Specification
- gRPC Documentation
- AsyncAPI Specification
- Neo4j Best Practices
- OpenTelemetry Documentation

---

**Document History**:

- v1.0 (October 2025): Initial comprehensive architecture document
- v0.9 (September 2025): Draft for review
- v0.1 (August 2025): Initial outline

**Contributors**:

- Architecture Team
- Security Review Board
- Performance Engineering Team

**Status**: APPROVED FOR IMPLEMENTATION

**Next Review**: January 2026
