# Project Context: Medical_KG_rev

## Purpose

Medical_KG_rev is an enterprise-grade, multi-protocol API gateway and orchestration system designed for comprehensive biomedical knowledge integration. The platform ingests, processes, normalizes, and unifies heterogeneous data from diverse sources—clinical trials, peer-reviewed literature, drug safety databases, regulatory filings, and standardized medical ontologies—into a federated knowledge graph with advanced multi-strategy retrieval.

### Core Value Proposition

**Problem**: Biomedical researchers, clinicians, and analysts face fragmented data across dozens of incompatible APIs, formats, and standards. Manual integration is time-consuming, error-prone, and doesn't scale.

**Solution**: A unified platform that:

- **Ingests** from 10+ biomedical sources automatically with resilient adapters
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

1. **Multi-Source Ingestion**: Automated data collection from 11+ biomedical sources
   - Clinical trials: ClinicalTrials.gov API v2
   - Literature: OpenAlex, PubMed Central, Unpaywall, Crossref, CORE, Semantic Scholar
   - Drug databases: OpenFDA (labels, adverse events, devices)
   - Ontologies: RxNorm, ICD-11, ChEMBL
   - Adapter SDK with YAML configs and Python classes

2. **GPU-Accelerated Processing**: Fail-fast GPU services with gRPC communication
   - MinerU: PDF parsing with layout analysis and OCR
   - Embedding: SPLADE (sparse) + Qwen-3 (dense) dual embeddings
   - Extraction: LLM-based span-grounded extraction with templates (PICO, effects, AE, dose, eligibility)
   - Batch processing for efficiency
   - GPU manager with utilization metrics

3. **Knowledge Graph Construction**: Neo4j 5.x with comprehensive validation
   - FHIR-aligned schema (Evidence, ResearchStudy, MedicationStatement)
   - SHACL shape validation (shapes.ttl)
   - Cypher query templates with MERGE operations for idempotency
   - Provenance tracking via ExtractionActivity nodes
   - Entity linking to RxNorm, ICD-11, SNOMED codes

4. **Advanced Retrieval**: Multi-strategy hybrid search with fusion ranking
   - BM25 (lexical full-text via OpenSearch)
   - SPLADE (learned sparse embeddings via OpenSearch)
   - Dense vectors (Qwen-3 embeddings via FAISS)
   - Reciprocal Rank Fusion (RRF) for combining results
   - Cross-encoder reranking for top-k results
   - Semantic chunking (paragraph, section, table-aware)
   - Span highlighting in results

5. **Multi-Protocol API**: Single GatewayService with 5 protocol facades
   - REST: FastAPI with OpenAPI 3.1, JSON:API v1.1, OData v4 query support
   - GraphQL: Strawberry with DataLoader for N+1 prevention
   - gRPC: 4 services (ingestion, embedding, extraction, mineru) with Protocol Buffers
   - SOAP: Zeep adapter for legacy integration
   - AsyncAPI/SSE: Real-time job status streaming
   - HTTP caching with ETag and Cache-Control headers

6. **Enterprise Security**: OAuth 2.0 with comprehensive protection
   - JWT validation (RS256, configurable algorithms)
   - Scope-based authorization (ingest:write, kg:read, retrieve:read, etc.)
   - Multi-tenant isolation (tenant_id in all queries)
   - API key management with rotation
   - Rate limiting with token bucket algorithm (per-client, per-endpoint)
   - Audit logging for all mutations
   - HashiCorp Vault integration for secrets
   - Security headers middleware (CSP, HSTS, X-Frame-Options)

7. **Production Observability**: Comprehensive monitoring and tracing
   - Prometheus metrics (API latency, GPU utilization, job throughput)
   - OpenTelemetry distributed tracing (Jaeger backend)
   - Structured logging with correlation IDs (structlog)
   - Sentry error tracking and alerting
   - Grafana dashboards (system health, API performance, GPU metrics)
   - Alertmanager for SLO violations (P95 latency, error rate)

8. **Automated CI/CD**: Multi-layer testing and deployment
   - Contract tests: Schemathesis (REST), GraphQL Inspector, Buf (gRPC)
   - Performance tests: k6 with P95 < 500ms threshold
   - Integration tests: Docker Compose test environment
   - Unit tests: pytest with 80%+ coverage target
   - Docker images with multi-stage builds
   - Kubernetes manifests with HPA and ingress
   - GitHub Actions CI/CD pipeline

9. **Domain Validation & Caching**: Standards-compliant validation
   - UCUM unit validation (pint library)
   - FHIR R5 resource validation (jsonschema)
   - HTTP caching (ETag, Cache-Control, Last-Modified)
   - Extraction template schemas with span validation
   - SHACL shape definitions for graph constraints

## Tech Stack

### Core Technologies

- **Language**: Python 3.12 with strict type hints (mypy)
- **API Framework**: FastAPI (REST/SSE), Strawberry GraphQL, gRPC (Protocol Buffers)
- **Data Processing**: Apache Kafka, Pydantic v2 (validation), MinerU (PDF)
- **ML/AI**: PyTorch, Transformers, SPLADE, Qwen-3, Sentence Transformers
- **Storage**: Neo4j 5.x (graph), OpenSearch (search), FAISS (vectors), MinIO/S3 (objects), Redis (cache)
- **Auth**: OAuth 2.0 with JWT (python-jose)
- **Monitoring**: Prometheus, OpenTelemetry, Grafana, Jaeger, structlog
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
- **Security**: cryptography, passlib, python-dotenv, hvac (HashiCorp Vault)
- **Validation**: pint (UCUM units), jsonschema, pyshacl, rdflib
- **Tokenization**: tiktoken
- **SOAP**: zeep
- **gRPC**: grpcio, grpcio-tools, grpcio-health-checking, grpc-stubs (type hints)

## Project Conventions

### Code Style

- Line length: 100 characters (Black/Ruff)
- Type hints: Required (strict mypy enforcement)
- Imports: Sorted with isort via Ruff (relative imports allowed within package)
- Naming: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- Docstrings: Google style with type annotations
- Async/await: Preferred for I/O operations
- Linting: Ruff with TID252 (relative imports) and N999 (package naming) ignored

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

---

## Implementation Examples

### Example 1: Adding a New Biomedical Adapter

**Scenario**: Add support for PubChem API

**Step 1**: Create YAML configuration (for simple REST APIs)

```yaml
# src/Medical_KG_rev/adapters/config/pubchem.yaml
name: pubchem
base_url: https://pubchem.ncbi.nlm.nih.gov/rest/pug
rate_limit:
  requests_per_second: 5
  burst: 10
auth:
  type: none
endpoints:
  compound_by_cid:
    method: GET
    path: /compound/cid/{cid}/JSON
    params:
      cid: required
    response_mapping:
      doc_id: "PC_{cid}"
      title: "$.PC_Compounds[0].props[?(@.urn.label=='IUPAC Name')].value.sval"
      content: "$.PC_Compounds[0]"
      metadata:
        source: pubchem
        cid: "{cid}"
```

**Step 2**: Or create Python adapter (for complex sources)

```python
# src/Medical_KG_rev/adapters/biomedical.py

class PubChemAdapter(BaseAdapter):
    """Adapter for PubChem compound database."""

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.rate_limiter = RateLimiter(requests_per_second=5)

    async def fetch(self, **params: Any) -> list[dict[str, Any]]:
        """Fetch compound data from PubChem."""
        cid = params.get("cid")
        if not cid:
            raise ValueError("cid parameter required")

        await self.rate_limiter.acquire()

        async with self.http_client.get(
            f"{self.base_url}/compound/cid/{cid}/JSON"
        ) as response:
            response.raise_for_status()
            return [await response.json()]

    async def parse(self, raw_data: list[dict[str, Any]]) -> list[Document]:
        """Parse PubChem JSON to Document IR."""
        documents = []
        for item in raw_data:
            compound = item["PC_Compounds"][0]

            # Extract IUPAC name
            iupac_name = next(
                (p["value"]["sval"] for p in compound["props"]
                 if p["urn"]["label"] == "IUPAC Name"),
                "Unknown"
            )

            doc = Document(
                doc_id=f"PC_{compound['id']['id']['cid']}",
                title=iupac_name,
                content=json.dumps(compound),
                source="pubchem",
                metadata={
                    "cid": compound["id"]["id"]["cid"],
                    "molecular_formula": self._extract_formula(compound),
                    "molecular_weight": self._extract_weight(compound),
                },
                domain_type="medical",
                domain_data=MedicalDomain(
                    resource_type="Substance",
                    code=CodeableConcept(
                        coding=[Coding(
                            system="http://pubchem.ncbi.nlm.nih.gov",
                            code=str(compound["id"]["id"]["cid"]),
                            display=iupac_name
                        )]
                    )
                )
            )
            documents.append(doc)

        return documents

    async def validate(self, documents: list[Document]) -> list[Document]:
        """Validate parsed documents."""
        for doc in documents:
            # Validate CID format
            if not doc.metadata.get("cid"):
                raise ValidationError(f"Missing CID in {doc.doc_id}")

            # Validate molecular weight
            mw = doc.metadata.get("molecular_weight")
            if mw and (mw < 0 or mw > 10000):
                raise ValidationError(f"Invalid molecular weight: {mw}")

        return documents
```

**Step 3**: Register adapter

```python
# src/Medical_KG_rev/adapters/plugins/domains/biomedical/pubchem.py

from Medical_KG_rev.adapters.plugins.base import BaseAdapterPlugin
from Medical_KG_rev.adapters.plugins.domains.metadata import BiomedicalAdapterMetadata

class PubChemAdapterPlugin(BaseAdapterPlugin):
    metadata = BiomedicalAdapterMetadata(
        name="pubchem",
        version="1.0.0",
        summary="PubChem compound ingestion",
        capabilities=["compound"],
        maintainer="Data Platform",
        dataset="pubchem",
    )

    # implement fetch/parse/validate hooks...


manager.register(PubChemAdapterPlugin())
```

**Step 4**: Add REST endpoint

```python
# src/Medical_KG_rev/gateway/rest/ingest.py

@router.post("/ingest/pubchem", response_model=JSONAPIResponse)
async def ingest_pubchem(
    request: PubChemIngestRequest,
    auth: AuthContext = Depends(require_scope("ingest:write")),
    kafka: KafkaProducer = Depends(get_kafka_producer),
) -> JSONAPIResponse:
    """Ingest compound data from PubChem."""
    job_id = generate_job_id()

    await kafka.send(
        "ingest.requests.v1",
        {
            "job_id": job_id,
            "tenant_id": auth.tenant_id,
            "adapter": "pubchem",
            "params": {"cid": request.data.attributes.cid},
        }
    )

    return JSONAPIResponse(
        data=JSONAPIResource(
            type="ingestion",
            id=job_id,
            attributes={"status": "queued", "adapter": "pubchem"}
        )
    )
```

**Step 5**: Add tests

```python
# tests/adapters/test_pubchem.py

@pytest.mark.asyncio
async def test_pubchem_adapter_fetch():
    """Test PubChem adapter fetches compound data."""
    adapter = PubChemAdapter(AdapterConfig(name="pubchem"))

    with aioresponses() as m:
        m.get(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/JSON",
            payload=MOCK_PUBCHEM_RESPONSE
        )

        results = await adapter.fetch(cid="2244")
        assert len(results) == 1
        assert results[0]["PC_Compounds"][0]["id"]["id"]["cid"] == 2244

@pytest.mark.asyncio
async def test_pubchem_adapter_parse():
    """Test PubChem adapter parses to Document IR."""
    adapter = PubChemAdapter(AdapterConfig(name="pubchem"))

    documents = await adapter.parse([MOCK_PUBCHEM_RESPONSE])
    assert len(documents) == 1
    assert documents[0].doc_id == "PC_2244"
    assert documents[0].source == "pubchem"
    assert documents[0].metadata["cid"] == 2244
```

---

### Example 2: Implementing a Custom Extraction Template

**Scenario**: Add extraction template for pharmacokinetics (PK) parameters

**Step 1**: Define template schema

```python
# src/Medical_KG_rev/services/extraction/templates.py

class PKParameterTemplate(BaseModel):
    """Pharmacokinetic parameter extraction template."""

    drug_name: str = Field(description="Name of the drug")
    parameter_type: Literal["Cmax", "Tmax", "AUC", "t1/2", "Vd", "CL"] = Field(
        description="Type of PK parameter"
    )
    value: float = Field(description="Numeric value of the parameter")
    unit: str = Field(description="Unit of measurement (UCUM)")
    population: str = Field(description="Patient population (e.g., 'healthy adults')")
    dose: Optional[str] = Field(None, description="Administered dose")
    route: Optional[str] = Field(None, description="Route of administration")

    # Span grounding
    span_start: int = Field(description="Character offset where evidence starts")
    span_end: int = Field(description="Character offset where evidence ends")
    span_text: str = Field(description="Exact text supporting this extraction")

    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")

class PKExtractionResult(BaseModel):
    """Result of PK parameter extraction."""
    doc_id: str
    parameters: list[PKParameterTemplate]
    extraction_method: str
    model_name: str
    timestamp: datetime
```

**Step 2**: Create extraction prompt

```python
# src/Medical_KG_rev/services/extraction/prompts.py

PK_EXTRACTION_PROMPT = """
Extract pharmacokinetic (PK) parameters from the following medical text.

For each PK parameter found, provide:
1. Drug name
2. Parameter type (Cmax, Tmax, AUC, t1/2, Vd, CL)
3. Numeric value
4. Unit (use UCUM standard)
5. Patient population
6. Dose (if mentioned)
7. Route of administration (if mentioned)
8. Exact text span (start, end, text)
9. Confidence score (0-1)

Text:
{text}

Return JSON array of PK parameters. Ensure units are UCUM-compliant.
"""
```

**Step 3**: Implement extraction service

```python
# src/Medical_KG_rev/services/extraction/pk_extractor.py

class PKExtractor:
    """Extracts pharmacokinetic parameters from text."""

    def __init__(
        self,
        llm_client: LLMClient,
        ucum_validator: UCUMValidator,
    ):
        self.llm = llm_client
        self.ucum = ucum_validator

    async def extract(
        self,
        doc_id: str,
        text: str,
    ) -> PKExtractionResult:
        """Extract PK parameters from document text."""

        # Call LLM with extraction prompt
        prompt = PK_EXTRACTION_PROMPT.format(text=text)
        response = await self.llm.complete(
            prompt=prompt,
            response_format="json",
            temperature=0.1,  # Low temperature for factual extraction
        )

        # Parse and validate
        raw_params = json.loads(response.content)
        validated_params = []

        for param in raw_params:
            # Validate UCUM unit
            if not self.ucum.validate(param["unit"]):
                logger.warning(
                    f"Invalid UCUM unit: {param['unit']} in {doc_id}"
                )
                continue

            # Validate span grounding
            span_text = text[param["span_start"]:param["span_end"]]
            if span_text != param["span_text"]:
                logger.warning(
                    f"Span mismatch in {doc_id}: "
                    f"expected '{param['span_text']}', got '{span_text}'"
                )
                continue

            validated_params.append(PKParameterTemplate(**param))

        return PKExtractionResult(
            doc_id=doc_id,
            parameters=validated_params,
            extraction_method="llm",
            model_name=self.llm.model_name,
            timestamp=datetime.utcnow(),
        )

    async def write_to_graph(
        self,
        result: PKExtractionResult,
        neo4j_client: Neo4jClient,
    ) -> None:
        """Write extracted PK parameters to knowledge graph."""

        for param in result.parameters:
            query = """
            MATCH (d:Document {doc_id: $doc_id})
            MERGE (drug:Drug {name: $drug_name})
            MERGE (pk:PKParameter {
                drug_name: $drug_name,
                parameter_type: $parameter_type,
                value: $value,
                unit: $unit,
                population: $population
            })
            MERGE (d)-[:MENTIONS]->(pk)
            MERGE (pk)-[:DESCRIBES]->(drug)
            CREATE (activity:ExtractionActivity {
                id: randomUUID(),
                method: $method,
                model_name: $model_name,
                timestamp: datetime($timestamp),
                confidence: $confidence
            })
            MERGE (activity)-[:EXTRACTED]->(pk)
            MERGE (activity)-[:FROM_DOCUMENT]->(d)
            SET pk.span_start = $span_start,
                pk.span_end = $span_end,
                pk.span_text = $span_text
            """

            await neo4j_client.execute(
                query,
                doc_id=result.doc_id,
                drug_name=param.drug_name,
                parameter_type=param.parameter_type,
                value=param.value,
                unit=param.unit,
                population=param.population,
                method=result.extraction_method,
                model_name=result.model_name,
                timestamp=result.timestamp.isoformat(),
                confidence=param.confidence,
                span_start=param.span_start,
                span_end=param.span_end,
                span_text=param.span_text,
            )
```

**Step 4**: Add REST endpoint

```python
# src/Medical_KG_rev/gateway/rest/extract.py

@router.post("/extract/pk", response_model=JSONAPIResponse)
async def extract_pk_parameters(
    request: PKExtractionRequest,
    auth: AuthContext = Depends(require_scope("extract:write")),
    extractor: PKExtractor = Depends(get_pk_extractor),
) -> JSONAPIResponse:
    """Extract pharmacokinetic parameters from document."""

    # Fetch document
    doc = await get_document(request.data.attributes.doc_id, auth.tenant_id)

    # Extract PK parameters
    result = await extractor.extract(doc.doc_id, doc.content)

    # Write to graph
    await extractor.write_to_graph(result, get_neo4j_client())

    return JSONAPIResponse(
        data=JSONAPIResource(
            type="extraction",
            id=result.doc_id,
            attributes={
                "parameter_count": len(result.parameters),
                "model_name": result.model_name,
                "timestamp": result.timestamp.isoformat(),
            }
        )
    )
```

---

### Example 3: Configuring Multi-Strategy Retrieval

**Scenario**: Query for "SGLT2 inhibitors heart failure outcomes" with hybrid search

**Step 1**: Client sends retrieval request

```bash
curl -X POST http://localhost:8000/v1/retrieve \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/vnd.api+json" \
  -d '{
    "data": {
      "type": "retrieval",
      "attributes": {
        "query": "SGLT2 inhibitors heart failure outcomes",
        "strategies": ["bm25", "splade", "dense"],
        "fusion_method": "rrf",
        "rerank": true,
        "top_k": 10,
        "filters": {
          "source": ["openalex", "clinicaltrials"],
          "date_range": {"start": "2020-01-01", "end": "2024-12-31"}
        }
      }
    }
  }'
```

**Step 2**: Retrieval service orchestrates multi-strategy search

```python
# src/Medical_KG_rev/services/retrieval/multi_strategy.py

class MultiStrategyRetriever:
    """Orchestrates multi-strategy hybrid retrieval."""

    def __init__(
        self,
        opensearch_client: OpenSearchClient,
        faiss_client: FAISSClient,
        embedding_service: EmbeddingService,
        reranker: CrossEncoderReranker,
    ):
        self.opensearch = opensearch_client
        self.faiss = faiss_client
        self.embedding = embedding_service
        self.reranker = reranker

    async def retrieve(
        self,
        query: str,
        strategies: list[str],
        fusion_method: str = "rrf",
        rerank: bool = True,
        top_k: int = 10,
        filters: Optional[dict] = None,
        tenant_id: str,
    ) -> RetrievalResult:
        """Execute multi-strategy retrieval with fusion and reranking."""

        # Execute strategies in parallel
        tasks = []
        if "bm25" in strategies:
            tasks.append(self._bm25_search(query, filters, tenant_id))
        if "splade" in strategies:
            tasks.append(self._splade_search(query, filters, tenant_id))
        if "dense" in strategies:
            tasks.append(self._dense_search(query, filters, tenant_id))

        results = await asyncio.gather(*tasks)

        # Fusion ranking
        if fusion_method == "rrf":
            fused = self._reciprocal_rank_fusion(results, k=60)
        elif fusion_method == "weighted":
            fused = self._weighted_fusion(results, weights=[0.3, 0.3, 0.4])
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Rerank top candidates
        if rerank:
            candidates = fused[:top_k * 3]  # Get 3x candidates for reranking
            reranked = await self.reranker.rerank(query, candidates)
            final = reranked[:top_k]
        else:
            final = fused[:top_k]

        return RetrievalResult(
            query=query,
            results=final,
            strategies_used=strategies,
            fusion_method=fusion_method,
            reranked=rerank,
            total_results=len(fused),
        )

    async def _bm25_search(
        self,
        query: str,
        filters: Optional[dict],
        tenant_id: str,
    ) -> list[ScoredDocument]:
        """BM25 lexical search via OpenSearch."""

        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"content": {"query": query, "boost": 1.0}}},
                        {"term": {"tenant_id": tenant_id}},  # Multi-tenant isolation
                    ],
                    "filter": self._build_filters(filters),
                }
            },
            "size": 100,
        }

        response = await self.opensearch.search(
            index="documents",
            body=search_body,
        )

        return [
            ScoredDocument(
                doc_id=hit["_id"],
                score=hit["_score"],
                content=hit["_source"]["content"],
                metadata=hit["_source"]["metadata"],
                strategy="bm25",
            )
            for hit in response["hits"]["hits"]
        ]

    async def _splade_search(
        self,
        query: str,
        filters: Optional[dict],
        tenant_id: str,
    ) -> list[ScoredDocument]:
        """SPLADE learned sparse search via OpenSearch."""

        # Generate SPLADE embedding for query
        query_embedding = await self.embedding.embed_splade(query)

        # Convert to sparse vector query
        sparse_query = {
            field: weight
            for field, weight in query_embedding.items()
            if weight > 0.01  # Threshold for sparsity
        }

        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {"rank_feature": {"field": f"splade.{field}", "boost": weight}}
                        for field, weight in sparse_query.items()
                    ] + [
                        {"term": {"tenant_id": tenant_id}},
                    ],
                    "filter": self._build_filters(filters),
                }
            },
            "size": 100,
        }

        response = await self.opensearch.search(
            index="documents",
            body=search_body,
        )

        return [
            ScoredDocument(
                doc_id=hit["_id"],
                score=hit["_score"],
                content=hit["_source"]["content"],
                metadata=hit["_source"]["metadata"],
                strategy="splade",
            )
            for hit in response["hits"]["hits"]
        ]

    async def _dense_search(
        self,
        query: str,
        filters: Optional[dict],
        tenant_id: str,
    ) -> list[ScoredDocument]:
        """Dense vector search via FAISS."""

        # Generate dense embedding for query
        query_vector = await self.embedding.embed_dense(query)

        # FAISS similarity search
        distances, indices = await self.faiss.search(
            query_vector=query_vector,
            k=100,
            tenant_id=tenant_id,  # Tenant-specific FAISS index
        )

        # Fetch document metadata
        doc_ids = [self.faiss.index_to_doc_id(idx) for idx in indices[0]]
        documents = await self._fetch_documents(doc_ids, filters)

        return [
            ScoredDocument(
                doc_id=doc.doc_id,
                score=1.0 / (1.0 + distances[0][i]),  # Convert distance to similarity
                content=doc.content,
                metadata=doc.metadata,
                strategy="dense",
            )
            for i, doc in enumerate(documents)
        ]

    def _reciprocal_rank_fusion(
        self,
        results: list[list[ScoredDocument]],
        k: int = 60,
    ) -> list[ScoredDocument]:
        """Fuse results using Reciprocal Rank Fusion (RRF)."""

        # Aggregate scores by doc_id
        doc_scores: dict[str, float] = {}
        doc_data: dict[str, ScoredDocument] = {}

        for result_list in results:
            for rank, doc in enumerate(result_list, start=1):
                rrf_score = 1.0 / (k + rank)
                doc_scores[doc.doc_id] = doc_scores.get(doc.doc_id, 0.0) + rrf_score
                doc_data[doc.doc_id] = doc  # Keep latest doc data

        # Sort by fused score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            ScoredDocument(
                **doc_data[doc_id].__dict__,
                score=score,
                strategy="rrf_fused",
            )
            for doc_id, score in sorted_docs
        ]
```

---

### Example 4: Implementing OAuth 2.0 Multi-Tenant Security

**Scenario**: Secure API with JWT validation and tenant isolation

**Step 1**: Configure OAuth settings

```python
# src/Medical_KG_rev/config.py

class OAuthSettings(BaseSettings):
    """OAuth 2.0 configuration."""

    issuer: str = Field(..., description="JWT issuer URL")
    audience: str = Field(..., description="JWT audience")
    algorithms: list[str] = Field(default=["RS256"], description="Allowed algorithms")
    jwks_url: str = Field(..., description="JWKS endpoint for public keys")
    scope_claim: str = Field(default="scope", description="JWT claim for scopes")
    tenant_claim: str = Field(default="tenant_id", description="JWT claim for tenant ID")

    class Config:
        env_prefix = "OAUTH_"
```

**Step 2**: Implement JWT validation middleware

```python
# src/Medical_KG_rev/security/auth.py

class JWTValidator:
    """Validates JWT tokens and extracts claims."""

    def __init__(self, settings: OAuthSettings):
        self.settings = settings
        self.jwks_client = PyJWKClient(settings.jwks_url)

    async def validate_token(self, token: str) -> dict[str, Any]:
        """Validate JWT and return decoded claims."""

        try:
            # Get signing key from JWKS
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)

            # Decode and validate
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=self.settings.algorithms,
                audience=self.settings.audience,
                issuer=self.settings.issuer,
            )

            return claims

        except JWTError as e:
            raise AuthenticationError(f"Invalid JWT: {e}")

class AuthContext(BaseModel):
    """Authentication context extracted from JWT."""

    user_id: str
    tenant_id: str
    scopes: list[str]
    email: Optional[str] = None
    name: Optional[str] = None

async def get_auth_context(
    authorization: str = Header(...),
    validator: JWTValidator = Depends(get_jwt_validator),
) -> AuthContext:
    """Extract and validate authentication context from request."""

    # Extract Bearer token
    if not authorization.startswith("Bearer "):
        raise AuthenticationError("Invalid authorization header")

    token = authorization[7:]  # Remove "Bearer " prefix

    # Validate JWT
    claims = await validator.validate_token(token)

    # Extract context
    return AuthContext(
        user_id=claims["sub"],
        tenant_id=claims.get(validator.settings.tenant_claim),
        scopes=claims.get(validator.settings.scope_claim, "").split(),
        email=claims.get("email"),
        name=claims.get("name"),
    )

def require_scope(required_scope: str):
    """Dependency that enforces scope authorization."""

    async def check_scope(auth: AuthContext = Depends(get_auth_context)) -> AuthContext:
        if required_scope not in auth.scopes:
            raise AuthorizationError(
                f"Missing required scope: {required_scope}. "
                f"Available scopes: {', '.join(auth.scopes)}"
            )
        return auth

    return check_scope
```

**Step 3**: Apply security to endpoints

```python
# src/Medical_KG_rev/gateway/rest/ingest.py

@router.post("/ingest/clinicaltrials", response_model=JSONAPIResponse)
async def ingest_clinical_trials(
    request: ClinicalTrialsIngestRequest,
    auth: AuthContext = Depends(require_scope("ingest:write")),  # Enforce scope
    kafka: KafkaProducer = Depends(get_kafka_producer),
) -> JSONAPIResponse:
    """Ingest clinical trials data (requires ingest:write scope)."""

    job_id = generate_job_id()

    # Include tenant_id for multi-tenant isolation
    await kafka.send(
        "ingest.requests.v1",
        {
            "job_id": job_id,
            "tenant_id": auth.tenant_id,  # From JWT
            "user_id": auth.user_id,
            "adapter": "clinicaltrials",
            "params": request.data.attributes.dict(),
        }
    )

    # Audit log
    await audit_log(
        action="ingest.clinicaltrials",
        user_id=auth.user_id,
        tenant_id=auth.tenant_id,
        resource_id=job_id,
        details={"nct_ids": request.data.attributes.nct_ids},
    )

    return JSONAPIResponse(
        data=JSONAPIResource(
            type="ingestion",
            id=job_id,
            attributes={"status": "queued"}
        )
    )

@router.get("/retrieve", response_model=JSONAPIResponse)
async def retrieve_documents(
    query: str,
    auth: AuthContext = Depends(require_scope("retrieve:read")),  # Different scope
    retriever: MultiStrategyRetriever = Depends(get_retriever),
) -> JSONAPIResponse:
    """Retrieve documents (requires retrieve:read scope)."""

    # Tenant isolation enforced in retrieval
    result = await retriever.retrieve(
        query=query,
        strategies=["bm25", "splade", "dense"],
        tenant_id=auth.tenant_id,  # Only return tenant's documents
    )

    return JSONAPIResponse(
        data=[
            JSONAPIResource(
                type="document",
                id=doc.doc_id,
                attributes={
                    "score": doc.score,
                    "content": doc.content[:500],  # Truncate for preview
                    "metadata": doc.metadata,
                }
            )
            for doc in result.results
        ]
    )
```

**Step 4**: Implement rate limiting per tenant

```python
# src/Medical_KG_rev/security/rate_limit.py

class TenantRateLimiter:
    """Token bucket rate limiter per tenant."""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    async def check_limit(
        self,
        tenant_id: str,
        endpoint: str,
        limit: int = 100,  # requests per minute
        window: int = 60,  # seconds
    ) -> bool:
        """Check if tenant is within rate limit."""

        key = f"rate_limit:{tenant_id}:{endpoint}"

        # Increment counter
        count = await self.redis.incr(key)

        # Set expiry on first request
        if count == 1:
            await self.redis.expire(key, window)

        # Check limit
        if count > limit:
            raise RateLimitError(
                f"Rate limit exceeded for tenant {tenant_id} on {endpoint}. "
                f"Limit: {limit} requests per {window} seconds."
            )

        return True

async def enforce_rate_limit(
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
    limiter: TenantRateLimiter = Depends(get_rate_limiter),
) -> None:
    """Middleware to enforce rate limits."""

    endpoint = request.url.path

    # Get rate limit config for endpoint
    limit_config = get_rate_limit_config(endpoint)

    await limiter.check_limit(
        tenant_id=auth.tenant_id,
        endpoint=endpoint,
        limit=limit_config.requests_per_minute,
        window=60,
    )
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: GPU Service Fails to Start

**Symptoms**:

- Error: "CUDA not available"
- Service exits immediately

**Solutions**:

1. Check GPU availability:

   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. Verify CUDA drivers:

   ```bash
   nvcc --version
   ```

3. Check Docker GPU runtime (if containerized):

   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

4. Set CUDA_VISIBLE_DEVICES:

   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

#### Issue 2: Rate Limit Errors from External APIs

**Symptoms**:

- 429 Too Many Requests errors
- Adapter failures

**Solutions**:

1. Check adapter rate limit config:

   ```python
   # src/Medical_KG_rev/adapters/config/openalex.yaml
   rate_limit:
     requests_per_second: 10  # Reduce if hitting limits
     burst: 20
   ```

2. Add polite headers:

   ```python
   headers = {
       "User-Agent": "Medical_KG_rev/0.1.0 (mailto:your@email.com)"
   }
   ```

3. Implement exponential backoff:

   ```python
   @retry(
       stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=4, max=60),
       retry=retry_if_exception_type(RateLimitError),
   )
   async def fetch_with_retry(...):
       ...
   ```

#### Issue 3: Multi-Tenant Data Leakage

**Symptoms**:

- Users seeing other tenants' data
- Security audit failures

**Solutions**:

1. Always filter by tenant_id:

   ```python
   # WRONG - no tenant filter
   query = "MATCH (d:Document) WHERE d.doc_id = $doc_id RETURN d"

   # CORRECT - with tenant filter
   query = """
   MATCH (d:Document)
   WHERE d.doc_id = $doc_id AND d.tenant_id = $tenant_id
   RETURN d
   """
   ```

2. Add tenant_id index:

   ```cypher
   CREATE INDEX document_tenant_idx FOR (d:Document) ON (d.tenant_id);
   ```

3. Test tenant isolation:

   ```python
   # tests/security/test_tenant_isolation.py
   async def test_cannot_access_other_tenant_data():
       # User from tenant A
       auth_a = AuthContext(user_id="user1", tenant_id="tenant_a", scopes=["retrieve:read"])

       # Document from tenant B
       doc_b = await create_document(tenant_id="tenant_b")

       # Should not be able to retrieve
       with pytest.raises(NotFoundError):
           await retrieve_document(doc_b.doc_id, auth_a.tenant_id)
   ```

#### Issue 4: UCUM Validation Failures

**Symptoms**:

- "Invalid UCUM unit" errors
- Extraction failures

**Solutions**:

1. Use standard UCUM units:

   ```python
   # WRONG
   "10 milligrams per deciliter"  # Full words not supported

   # CORRECT
   "10 mg/dL"  # UCUM standard abbreviations
   ```

2. Check unit registry:

   ```python
   from Medical_KG_rev.validation import UCUMValidator

   validator = UCUMValidator()
   print(validator.list_units())  # See all supported units
   ```

3. Convert units if needed:

   ```python
   validator.convert("10 mg/dL", "g/L")  # Returns converted value
   ```

#### Issue 5: OpenSearch SPLADE Indexing Slow

**Symptoms**:

- Slow ingestion pipeline
- High CPU usage on OpenSearch

**Solutions**:

1. Batch SPLADE embeddings:

   ```python
   # Batch size 32 for GPU efficiency
   embeddings = await embedding_service.embed_splade_batch(texts, batch_size=32)
   ```

2. Use bulk indexing:

   ```python
   await opensearch_client.bulk(
       index="documents",
       body=[
           {"index": {"_id": doc.doc_id}},
           {"content": doc.content, "splade": embedding, ...}
           for doc, embedding in zip(documents, embeddings)
       ]
   )
   ```

3. Tune OpenSearch settings:

   ```json
   {
     "index": {
       "refresh_interval": "30s",
       "number_of_replicas": 0
     }
   }
   ```

## Project Structure

```
Medical_KG_rev/
├── src/Medical_KG_rev/       # Main package
│   ├── models/               # Pydantic data models
│   │   ├── ir.py            # Intermediate Representation
│   │   ├── entities.py      # Entity, Claim, Evidence
│   │   ├── organization.py  # Organization, Tenant
│   │   └── overlays/        # Domain-specific extensions
│   ├── adapters/            # Data source adapters
│   │   ├── base.py          # BaseAdapter abstract class
│   │   ├── clinicaltrials.py
│   │   ├── openfda.py
│   │   ├── openalex.py
│   │   └── ...
│   ├── gateway/             # API Gateway
│   │   ├── rest/            # FastAPI REST endpoints
│   │   ├── graphql/         # Strawberry GraphQL schema
│   │   ├── sse/             # Server-Sent Events
│   │   └── soap/            # SOAP adapter
│   ├── services/            # gRPC microservices
│   │   ├── mineru/          # PDF parsing service
│   │   ├── embedding/       # Embedding generation
│   │   └── extraction/      # LLM extraction
│   ├── orchestration/       # Job orchestration
│   │   ├── kafka_client.py  # Kafka producer/consumer
│   │   ├── ledger.py        # State tracking
│   │   ├── pipeline.py      # Pipeline definitions
│   │   └── workers.py       # Background workers
│   ├── kg/                  # Knowledge Graph
│   │   ├── neo4j_client.py  # Neo4j driver wrapper
│   │   ├── cypher.py        # Query templates
│   │   └── shapes/          # SHACL shapes
│   ├── retrieval/           # Multi-strategy retrieval
│   │   ├── bm25.py
│   │   ├── splade.py
│   │   ├── dense.py
│   │   └── fusion.py
│   ├── chunking/            # Semantic chunking
│   ├── indexing/            # OpenSearch & FAISS
│   ├── auth/                # OAuth & JWT
│   ├── middleware/          # Rate limiting, tenant isolation
│   ├── utils/               # Shared utilities
│   ├── config/              # Configuration management
│   └── storage/             # Storage abstractions
├── proto/                   # gRPC Protocol Buffers
├── tests/                   # Test suites
│   ├── unit/
│   ├── integration/
│   ├── contract/
│   └── performance/
├── docs/                    # Documentation
├── openspec/                # OpenSpec proposals
├── ops/                     # Deployment
│   ├── docker-compose.yml
│   ├── k8s/
│   └── monitoring/
├── scripts/                 # Utility scripts
├── pyproject.toml           # Project metadata & dependencies
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

# Start background workers (in separate terminal)
python -m Medical_KG_rev.orchestration.workers

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
