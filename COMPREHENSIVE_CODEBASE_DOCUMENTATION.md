# Comprehensive Medical_KG_rev Codebase Documentation

> **Documentation Strategy**: This document follows "Documentation as Code" principles, treating documentation with the same rigor as source code. It lives in version control, follows consistent formatting, and evolves alongside the codebase. Last updated: \`2025-10-08\` | Version: \`2.2.0\`

## 📋 Documentation Overview

### Purpose & Scope

This comprehensive documentation serves as the single source of truth for the Medical_KG_rev codebase, providing technical depth for developers while maintaining accessibility for stakeholders. It covers architecture, implementation details, operational procedures, and development guidelines.

### Target Audiences

- **Developers**: Implementation details, API contracts, testing strategies
- **Architects**: System design, integration patterns, scalability considerations
- **DevOps**: Deployment, monitoring, operational procedures
- **Product Managers**: Feature capabilities, roadmap alignment

### Documentation Structure

```
📚 COMPREHENSIVE_CODEBASE_DOCUMENTATION.md (This file)
├── Executive Summary & Architecture Overview
├── Technical Architecture Deep Dive
├── API Documentation & Examples
├── Database Schema & Data Models
├── Configuration Management
├── Development Setup & Testing
├── Deployment & Operations
├── Security Implementation
├── Performance Tuning
└── Troubleshooting & Maintenance

📖 README.md (Quick start guide)
├── Project overview & key features
├── Installation & setup instructions
├── API usage examples
├── Development workflow
└── Support & contribution guidelines

📁 docs/ (Detailed guides & specifications)
├── API documentation (OpenAPI, GraphQL schemas)
├── Architecture blueprints & design rationale
├── Development guides & best practices
├── Operational runbooks & troubleshooting
└── Integration examples & case studies

🔧 openspec/ (Change proposals & specifications)
├── Active change proposals (openspec/changes/)
├── Capability specifications (openspec/specs/)
└── Project conventions (openspec/project.md)
```

## 🔧 Technical Architecture Deep Dive

### **Codebase Structure & Organization**

The Medical_KG_rev codebase follows a highly modular, layered architecture designed for scalability, maintainability, and extensibility. The system is organized into distinct layers with clear separation of concerns:

```
src/Medical_KG_rev/
├── adapters/           # External data source integrations
│   ├── base.py         # Adapter SDK base classes and interfaces
│   ├── biomedical.py   # Biomedical domain adapters (legacy)
│   ├── clinicaltrials/ # ClinicalTrials.gov integration
│   ├── core/          # CORE repository adapter
│   ├── crossref/      # Crossref DOI resolution
│   ├── config/        # YAML-based adapter configurations
│   ├── interfaces/    # Adapter interface definitions (PDF, etc.)
│   ├── mixins/        # Reusable adapter functionality
│   ├── openalex/      # OpenAlex research repository
│   ├── openfda/       # FDA drug and device data
│   ├── plugins/       # Plugin-based adapter system
│   ├── pmc/           # PubMed Central full-text
│   ├── semanticscholar/ # Semantic Scholar integration
│   ├── terminology/   # Medical terminology services
│   └── unpaywall/     # Open access status checking
├── auth/              # Authentication & authorization
│   ├── api_keys.py    # API key management
│   ├── audit.py       # Security audit logging
│   ├── context.py     # Security context management
│   ├── dependencies.py # FastAPI dependency injection
│   ├── jwt.py         # JWT token handling
│   ├── rate_limit.py  # Rate limiting enforcement
│   └── scopes.py      # Permission scope definitions
├── chunking/          # Document processing & chunking
│   ├── adapters/      # Chunking algorithm adapters
│   ├── assembly.py    # Chunk assembly logic
│   ├── base.py        # Core chunking abstractions
│   ├── chunkers/      # Chunking algorithm implementations
│   ├── coherence.py   # Chunk coherence validation
│   ├── configuration.py # Chunking configuration
│   ├── data/          # Chunking test data
│   ├── exceptions.py  # Chunking-specific errors
│   ├── factory.py     # Chunker factory pattern
│   ├── models.py      # Chunk data structures
│   ├── pipeline.py    # Chunking pipeline orchestration
│   ├── ports.py       # Chunking service interfaces
│   ├── provenance.py  # Chunking provenance tracking
│   ├── registry.py    # Chunker registration system
│   ├── runtime.py     # Chunking execution runtime
│   ├── segmentation.py # Text segmentation logic
│   ├── sentence_splitters.py # Sentence boundary detection
│   ├── service.py     # Chunking service layer
│   ├── tables.py      # Table extraction and processing
│   └── tokenization.py # Token-level text processing
├── config/            # Configuration management
│   ├── domains.py     # Domain-specific configurations
│   ├── embeddings.py  # Embedding service configuration
│   ├── settings.py    # Main application settings
│   ├── vector_store.py # Vector storage configuration
│   └── vllm_config.py # VLLM GPU service configuration
├── embeddings/        # Vector embedding services
│   ├── dense/         # Dense vector embeddings
│   ├── frameworks/    # Embedding framework integrations
│   ├── multi_vector/  # Multi-vector representations
│   ├── namespace.py   # Embedding namespace management
│   ├── neural_sparse/ # Neural sparse embeddings (SPLADE)
│   ├── ports.py       # Embedding service interfaces
│   ├── providers.py   # Embedding provider registry
│   ├── registry.py    # Embedding provider registration
│   ├── sparse/        # Sparse vector embeddings
│   ├── storage.py     # Embedding vector storage
│   └── utils/         # Embedding utility functions
├── gateway/           # Multi-protocol API gateway
│   ├── app.py         # FastAPI application factory
│   ├── coordinators/  # Coordinator pattern implementation
│   │   ├── base.py    # Base coordinator abstractions
│   │   ├── chunking.py # Chunking coordinator
│   │   ├── embedding.py # Embedding coordinator
│   │   └── job_lifecycle.py # Job lifecycle management
│   ├── graphql/       # GraphQL API implementation
│   ├── grpc/          # gRPC service definitions
│   ├── main.py        # Application entry point
│   ├── middleware.py  # Gateway middleware stack
│   ├── models.py      # Gateway data models
│   ├── presentation/  # Response presentation layer
│   ├── rest/          # REST API endpoints
│   ├── services.py    # Gateway service layer
│   ├── soap/          # SOAP protocol support
│   └── sse/           # Server-sent events
├── kg/                # Knowledge graph integration
│   ├── cypher_templates.py # Neo4j Cypher query templates
│   ├── neo4j_client.py # Neo4j database client
│   ├── schema.py      # Graph schema definitions
│   ├── shacl.py       # SHACL validation framework
│   └── shapes.ttl     # RDF shape constraints
├── models/            # Core data models
│   ├── entities.py    # Entity and claim models
│   ├── ir.py          # Intermediate representation
│   ├── organization.py # Organization and tenant models
│   ├── overlays/      # Domain-specific overlays
│   ├── provenance.py  # Provenance tracking models
│   └── artifact.py    # Artifact metadata models
├── observability/     # Monitoring & telemetry
│   ├── alerts.py      # Alert management system
│   ├── metrics.py     # Metrics collection framework
│   ├── sentry.py      # Error tracking integration
│   └── tracing.py     # Distributed tracing setup
├── orchestration/     # Workflow orchestration
│   ├── dagster/       # Dagster pipeline integration
│   ├── events.py      # Event-driven orchestration
│   ├── haystack/      # Haystack search integration
│   ├── kafka.py       # Kafka message broker
│   ├── ledger.py      # Job execution ledger
│   ├── openlineage.py # OpenLineage integration
│   ├── stages/        # Pipeline stage implementations
│   └── state/         # Pipeline state management
├── services/          # Microservices layer
│   ├── chunking/      # Chunking microservice
│   ├── embedding/     # Embedding microservice
│   ├── evaluation/    # Evaluation and testing services
│   ├── extraction/    # Information extraction services
│   ├── gpu/           # GPU resource management
│   ├── grpc/          # gRPC service implementations
│   ├── health.py      # Health check services
│   ├── ingestion/     # Data ingestion services
│   ├── mineru/        # MinerU PDF processing service
│   ├── parsing/       # Document parsing services
│   ├── reranking/     # Search result reranking
│   ├── retrieval/     # Information retrieval services
│   └── vector_store/  # Vector storage services
├── storage/           # Storage abstractions
│   ├── base.py        # Storage interface definitions
│   ├── cache.py       # Caching layer implementation
│   ├── clients.py     # Storage client implementations
│   ├── ledger.py      # Persistent job ledger
│   └── object_store.py # Object storage abstraction
├── utils/             # Utility functions
│   ├── errors.py      # Error handling utilities
│   ├── http_client.py # HTTP client abstractions
│   ├── identifiers.py # Unique identifier generation
│   ├── logging.py     # Structured logging setup
│   ├── metadata.py    # Metadata extraction utilities
│   ├── spans.py       # Tracing span management
│   ├── time.py        # Time-related utilities
│   ├── validation.py  # Input validation helpers
│   └── versioning.py  # Version management utilities
├── validation/        # Data validation
│   ├── fhir.py        # FHIR compliance validation
│   └── ucum.py        # UCUM unit validation
├── eval/              # Model evaluation and A/B testing
│   ├── ab_testing.py  # A/B testing framework
│   ├── embedding_eval.py # Embedding quality evaluation
│   ├── ground_truth.py # Ground truth management
│   ├── harness.py     # Evaluation harness
│   └── metrics.py     # Evaluation metrics
└── proto/             # Protocol buffer definitions
    ├── embedding.proto # Embedding service gRPC definitions
    ├── extraction.proto # Extraction service gRPC definitions
    ├── ingestion.proto # Ingestion service gRPC definitions
    └── mineru.proto   # MinerU service gRPC definitions
```

### **Layered Architecture Pattern**

The system implements a clean layered architecture with strict dependency rules:

**1. Foundation Layer (`models/`, `utils/`, `config/`)**

- Provides core data structures and utilities
- No dependencies on upper layers
- Establishes domain models and configuration management

**2. Infrastructure Layer (`storage/`, `observability/`, `validation/`)**

- Database clients, monitoring, and validation logic
- Depends only on foundation layer
- Provides infrastructure services to upper layers

**3. Service Layer (`services/`, `adapters/`, `chunking/`, `embeddings/`)**

- Business logic implementation
- Depends on foundation and infrastructure layers
- Provides domain services for orchestration

**4. Orchestration Layer (`orchestration/`)**

- Workflow coordination and pipeline management
- Depends on all lower layers
- Coordinates between services and external systems

**5. Gateway Layer (`gateway/`)**

- API presentation and protocol handling
- Depends on all lower layers
- Provides external interface to the system

**6. Evaluation Layer (`eval/`)**

- Model evaluation and A/B testing framework
- Depends on all lower layers
- Provides metrics and comparison tools for continuous improvement

### **Technology Stack & Dependencies**

**Core Framework:**

```python
# Primary dependencies
fastapi==0.104.1          # Multi-protocol API gateway
pydantic==2.5.0           # Data validation and serialization
sqlalchemy==2.0.23        # Database ORM and query building
structlog==23.2.0         # Structured logging framework
tenacity>=8.2,<9           # Retry logic with exponential backoff
pybreaker==0.6.0          # Circuit breaker pattern
aiolimiter==1.1.0         # Async rate limiting
```

**External Integrations:**

```python
# Biomedical data sources
clinicaltrials-gov-api    # Clinical trial data
openalex-py==0.1.0        # Research repository data
crossref-commons-py==0.1.0 # DOI resolution
unpaywall-py==0.1.0       # Open access status
```

**AI/ML & GPU Services:**

```python
torch>=2.0.0              # PyTorch for GPU acceleration
transformers>=4.35.0      # Hugging Face transformers
vllm>=0.2.0               # High-performance LLM serving
qdrant-client>=1.6.0      # Vector database client
pyserini>=1.2.0           # Information retrieval toolkit
mineru>=2.5.4             # PDF processing and extraction
scikit-learn>=1.7.2       # Machine learning utilities
numpy>=1.26.4             # Numerical computing
scipy>=1.16.2             # Scientific computing
rank-bm25>=0.2.2          # BM25 ranking algorithm
```

**Storage & Infrastructure:**

```python
neo4j-python-driver==5.13.0 # Graph database client
opensearch-py==2.4.0       # Search engine client
redis-py-cluster==2.1.3    # Redis cluster client
boto3==1.40.45            # AWS SDK for object storage
kafka-python-ng==2.2.2    # Kafka client for messaging
grpcio>=1.75.1            # gRPC framework
grpcio-tools>=1.75.1      # Protocol buffer compiler
grpcio-health-checking>=1.75.1 # gRPC health checking
grpcio-status>=1.75.1    # gRPC status handling
```

**Observability & Monitoring:**

```python
prometheus-client==0.19.0 # Metrics collection
opentelemetry-api==1.21.0  # Distributed tracing
opentelemetry-sdk==1.21.0  # OpenTelemetry SDK
sentry-sdk==1.38.0         # Error tracking
jaeger-client==4.8.0       # Jaeger tracing client
structlog==23.2.0          # Structured logging framework
```

**Plugin & Configuration:**

```python
pluggy>=1.6.0             # Plugin system framework
pyyaml>=6.0.1             # YAML parsing and generation
pydantic>=2.5.0           # Data validation and settings
pydantic-settings>=2.1.0  # Settings management
```

### **Coordinator Pattern Implementation**

The system implements a sophisticated coordinator pattern that provides resilient, observable operation coordination across the gateway layer:

**Base Coordinator Architecture:**

```python
# Core coordinator abstractions with full resilience support
class BaseCoordinator(ABC, Generic[RequestT, ResultT]):
    """Abstract base for all coordinators with built-in resilience."""

    def __init__(self, config: CoordinatorConfig, metrics: CoordinatorMetrics):
        self.config = config
        self.metrics = metrics
        self._retrying = config.build_retrying()  # Exponential backoff
        self._limiter = config.limiter           # Rate limiting
        self._breaker = config.breaker           # Circuit breaking

    def __call__(self, request: RequestT) -> ResultT:
        """Execute with full resilience stack."""
        # Automatic retry, circuit breaking, rate limiting, metrics
        with self.metrics.duration.time():
            return self._execute_with_guards(request)

    @abstractmethod
    def _execute(self, request: RequestT) -> ResultT:
        """Business logic implementation."""
        pass
```

**Coordinator Types:**

- **ChunkingCoordinator**: Document text segmentation and processing
- **EmbeddingCoordinator**: Vector embedding generation and storage
- **IngestionCoordinator**: Data source ingestion coordination
- **RetrievalCoordinator**: Multi-strategy search and ranking

**Resilience Features:**

```python
# Automatic resilience handling in base coordinator
class CoordinatorConfig:
    retry_attempts: int = 3                    # Retry failed operations
    retry_wait_base: float = 0.2               # Exponential backoff base
    retry_wait_max: float = 2.0                # Maximum backoff time
    breaker: CircuitBreaker | None = None      # Circuit breaker protection
    limiter: AsyncLimiter | None = None       # Rate limiting
```

**Job Lifecycle Management:**

```python
class JobLifecycleManager:
    """Centralized job creation, tracking, and state management."""

    async def create_job(self, tenant_id: str, operation: str) -> str:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        await self.ledger.create(job_id, tenant_id, operation)
        await self.events.publish(JobEvent(job_id, "created"))
        return job_id

    async def complete_job(self, job_id: str, metadata: dict) -> None:
        await self.ledger.mark_completed(job_id, metadata)
        await self.events.publish(JobEvent(job_id, "completed"))
```

### **Adapter Framework Architecture**

The adapter SDK provides a standardized interface for integrating external biomedical data sources with comprehensive lifecycle management:

**Adapter Lifecycle Pattern:**

```python
class BaseAdapter(ABC):
    """Standardized adapter interface with lifecycle hooks."""

    @abstractmethod
    def fetch(self, context: AdapterContext) -> Iterable[dict]:
        """Fetch raw data from external source."""
        pass

    @abstractmethod
    def parse(self, payloads: Iterable[dict], context: AdapterContext) -> Iterable[Document]:
        """Transform raw data into canonical IR format."""
        pass

    @abstractmethod
    def validate(self, documents: Iterable[Document], context: AdapterContext) -> ValidationResult:
        """Validate transformed documents before storage."""
        pass

    def run(self, context: AdapterContext) -> AdapterResult:
        """Execute complete adapter lifecycle."""
        payloads = self.fetch(context)
        documents = self.parse(payloads, context)
        validation = self.validate(documents, context)
        return AdapterResult(documents, validation.warnings)
```

**Biomedical Adapters:**

- **ClinicalTrials.gov**: Clinical trial protocol data
- **OpenAlex**: Research publication metadata
- **PubMed Central**: Full-text scientific articles
- **Unpaywall**: Open access status checking
- **Crossref**: DOI resolution and metadata
- **Semantic Scholar**: Academic paper data
- **FDA OpenFDA**: Drug and device regulatory data
- **Medical Terminology Services**: ICD-11, MeSH, RxNorm integration

**Plugin-Based Architecture:**

```python
# Advanced plugin system with hookimpl pattern
class BaseAdapterPlugin(ABC):
    """Abstract base class for adapter plugins with hookimpl pattern."""

    metadata: AdapterMetadata
    config_model: type[AdapterConfig] = AdapterConfig

    @hookimpl
    def get_metadata(self) -> AdapterMetadata:
        """Get adapter metadata and configuration schema."""

    @hookimpl
    @abstractmethod
    def fetch(self, request: AdapterRequest) -> AdapterResponse:
        """Fetch raw payloads from upstream systems."""

    @hookimpl
    @abstractmethod
    def parse(self, response: AdapterResponse, request: AdapterRequest) -> AdapterResponse:
        """Parse raw payloads into canonical documents."""

    @hookimpl
    def validate(self, response: AdapterResponse, request: AdapterRequest) -> ValidationOutcome:
        """Validate parsed documents before storage."""

    @hookimpl
    def health_check(self) -> bool:
        """Health check for adapter availability."""

    @hookimpl
    def estimate_cost(self, request: AdapterRequest) -> AdapterCostEstimate:
        """Estimate computational cost for adapter execution."""

class AdapterPluginManager:
    """Plugin system for adapter registration and execution."""

    def register(self, plugin: BaseAdapterPlugin) -> None:
        """Register adapter plugin with metadata."""
        self._adapters[plugin.metadata.name] = plugin
        self._capabilities.update(plugin.metadata.capabilities)

    async def invoke(self, name: str, request: AdapterRequest) -> AdapterInvocationResult:
        """Execute adapter with full lifecycle tracking."""
        plugin = self._adapters[name]
        context = AdapterExecutionContext(request)
        result = await plugin.execute(context)
        return AdapterInvocationResult(context, result, context.metrics)
```

**PDF Interface System:**

```python
@runtime_checkable
class PdfCapableAdapter(Protocol):
    """Protocol for adapters that can provide downloadable PDFs."""

    pdf_capabilities: Sequence[str]

    def iter_pdf_candidates(
        self,
        documents: Sequence[Document],
        *,
        context: AdapterContext | None = None,
    ) -> Iterable[PdfAssetManifest]:
        """Yield manifest entries for downloadable PDF assets."""

    def polite_headers(self) -> Mapping[str, str]:
        """Return polite pool headers for rate limiting compliance."""

@dataclass(frozen=True)
class PdfAssetManifest:
    """Normalised description of a single downloadable PDF asset."""

    url: str
    landing_page_url: str | None = None
    license: str | None = None
    version: str | None = None
    source: str | None = None
    checksum_hint: str | None = None
    is_open_access: bool | None = None
    content_type: str | None = None

@dataclass(frozen=True)
class PdfManifest:
    """Collection of manifest entries emitted by an adapter."""

    connector: str
    assets: tuple[PdfAssetManifest, ...]
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    polite_headers: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
```

**YAML-Based Adapter Configuration:**

```yaml
# config/adapters/clinicaltrials.yaml
name: clinicaltrials-yaml
source: clinicaltrials
base_url: https://clinicaltrials.gov/api/v2
rate_limit:
  requests: 10
  per_seconds: 1
request:
  method: GET
  path: /studies/{nct_id}
  params:
    format: json
response:
  items_path: study
mapping:
  id: protocolSection.identificationModule.nctId
  title: protocolSection.identificationModule.briefTitle
  summary: protocolSection.descriptionModule.briefSummary
  metadata:
    overall_status: protocolSection.statusModule.overallStatus
    phase: protocolSection.designModule.phase
    start_date: protocolSection.statusModule.startDateStruct.date
```

**Plugin-Based Configuration:**

```yaml
# config/adapters/plugin-registry.yaml
plugins:
  biomedical:
    - name: clinicaltrials
      class: "Medical_KG_rev.adapters.plugins.clinicaltrials:ClinicalTrialsPlugin"
      config:
        rate_limit_per_second: 5
        batch_size: 10
        timeout_seconds: 30

    - name: openalex
      class: "Medical_KG_rev.adapters.plugins.openalex:OpenAlexPlugin"
      config:
        rate_limit_per_second: 10
        max_results: 100

  terminology:
    - name: mesh
      class: "Medical_KG_rev.adapters.plugins.terminology:MeSHPlugin"
      config:
        cache_ttl_seconds: 3600

  evaluation:
    - name: embedding_eval
      class: "Medical_KG_rev.eval.harness:EmbeddingEvalPlugin"
      config:
        metrics: ["precision", "recall", "ndcg"]
        test_datasets: ["msmarco", "trec-covid"]
```

**Shared Mixins for Common Functionality:**

```python
# Reusable functionality across adapters
class HTTPWrapperMixin:
    """Standardized HTTP operations with retry and rate limiting."""
    def _get_json(self, path: str, **kwargs) -> dict:
        response = self.http_client.request("GET", path, **kwargs)
        return response.json()

class DOINormalizationMixin:
    """DOI validation and normalization utilities."""
    def normalize_doi(self, doi: str) -> str:
        # DOI validation and normalization logic
        pass

class PaginationMixin:
    """Common pagination patterns for APIs."""
    def paginate_results(self, fetch_func: Callable) -> Generator[dict, None, None]:
        # Pagination logic with configurable page sizes
        pass

class PdfManifestMixin:
    """PDF manifest generation for PDF-capable adapters."""
    def create_pdf_manifest(self, documents: Sequence[Document]) -> PdfManifest:
        """Create PDF manifest from document collection."""
        assets = []
        for doc in documents:
            if pdf_url := self._extract_pdf_url(doc):
                assets.append(PdfAssetManifest(
                    url=pdf_url,
                    landing_page_url=self._extract_landing_page(doc),
                    license=self._extract_license(doc),
                    source=self.metadata.name
                ))
        return PdfManifest(connector=self.metadata.name, assets=tuple(assets))
```

**Evaluation & Metrics System:**

```python
class EmbeddingEvaluator:
    """Comprehensive evaluation framework for embedding quality."""

    async def evaluate_embedding_quality(
        self,
        namespace: str,
        dataset: EvaluationDataset
    ) -> EvaluationReport:
        """Evaluate embedding quality using multiple metrics."""
        results = []

        # Information retrieval metrics
        for query, relevant_docs in dataset.queries.items():
            retrieved_docs = await self._retrieve_similar(query, namespace)
            precision = average_precision(retrieved_docs, relevant_docs)
            recall = recall_at_k(retrieved_docs, relevant_docs, k=10)
            ndcg = ndcg_at_k(retrieved_docs, relevant_docs, k=10)

            results.append(MetricResult(
                query=query,
                precision=precision,
                recall=recall,
                ndcg=ndcg
            ))

        return EvaluationReport(
            namespace=namespace,
            dataset_name=dataset.name,
            metrics=results,
            overall_score=sum(r.ndcg for r in results) / len(results)
        )

class ABTestRunner:
    """A/B testing framework for model comparison."""

    async def run_ab_test(
        self,
        baseline_model: str,
        candidate_model: str,
        test_queries: list[str]
    ) -> ABTestOutcome:
        """Compare two embedding models using statistical tests."""
        baseline_scores = []
        candidate_scores = []

        for query in test_queries:
            baseline_score = await self._evaluate_query(query, baseline_model)
            candidate_score = await self._evaluate_query(query, candidate_model)

            baseline_scores.append(baseline_score)
            candidate_scores.append(candidate_score)

        # Statistical significance testing
        t_stat, p_value = stats.ttest_rel(baseline_scores, candidate_scores)

        return ABTestOutcome(
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            baseline_mean=np.mean(baseline_scores),
            candidate_mean=np.mean(candidate_scores),
            p_value=p_value,
            statistically_significant=p_value < 0.05
        )
```

### **Multi-Protocol API Gateway**

The gateway provides a unified interface supporting 5 different protocols simultaneously:

**Protocol Support Matrix:**

- **REST API** (OpenAPI 3.1 + JSON:API 1.1)
- **GraphQL** (GraphQL Schema + DataLoader pattern)
- **gRPC** (Protocol Buffers + gRPC services with health checking)
- **SOAP** (WSDL + XML Schema)
- **AsyncAPI/SSE** (Server-sent events for real-time updates)

**gRPC Service Definitions:**

```protobuf
// ingestion.proto
service IngestionService {
  rpc Submit (IngestionJobRequest) returns (IngestionJobResponse);
  rpc GetStatus (JobStatusRequest) returns (JobStatusResponse);
  rpc Cancel (JobCancelRequest) returns (JobCancelResponse);
}

// embedding.proto
service EmbeddingService {
  rpc EmbedTexts (EmbedTextsRequest) returns (EmbedTextsResponse);
  rpc GetEmbeddings (GetEmbeddingsRequest) returns (GetEmbeddingsResponse);
  rpc DeleteEmbeddings (DeleteEmbeddingsRequest) returns (DeleteEmbeddingsResponse);
}

// extraction.proto
service ExtractionService {
  rpc ExtractEntities (ExtractEntitiesRequest) returns (ExtractEntitiesResponse);
  rpc ExtractClaims (ExtractClaimsRequest) returns (ExtractClaimsResponse);
}

// mineru.proto
service MineruService {
  rpc ProcessPDF (ProcessPDFRequest) returns (ProcessPDFResponse);
  rpc GetProcessingStatus (ProcessingStatusRequest) returns (ProcessingStatusResponse);
}
```

**gRPC Service Implementation:**

```python
class IngestionServicer(ingestion_pb2_grpc.IngestionServiceServicer):
    """gRPC service implementation for ingestion operations."""

    def __init__(self, ingestion_service: IngestionService):
        self.ingestion_service = ingestion_service

    async def Submit(
        self,
        request: ingestion_pb2.IngestionJobRequest,
        context: grpc.aio.ServicerContext
    ) -> ingestion_pb2.IngestionJobResponse:
        """Submit ingestion job via gRPC."""
        try:
            # Convert protobuf request to service request
            service_request = self._proto_to_service_request(request)

            # Execute ingestion
            result = await self.ingestion_service.submit_job(service_request)

            # Convert service response to protobuf response
            return self._service_to_proto_response(result)

        except Exception as e:
            # Handle errors with proper gRPC status codes
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Ingestion failed: {str(e)}")
            return ingestion_pb2.IngestionJobResponse()
```

**gRPC Health Checking:**

```python
class GrpcServiceState:
    """Tracks readiness and exposes health reporting."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.health_servicer = health.HealthServicer()
        self.ready = asyncio.Event()

        # Initially not serving
        self.health_servicer.set(
            self.service_name,
            health_pb2.HealthCheckResponse.NOT_SERVING
        )

    def set_ready(self) -> None:
        """Mark service as ready and update health status."""
        self.ready.set()
        self.health_servicer.set(
            self.service_name,
            health_pb2.HealthCheckResponse.SERVING
        )

    def set_not_ready(self) -> None:
        """Mark service as not ready."""
        self.ready.clear()
        self.health_servicer.set(
            self.service_name,
            health_pb2.HealthCheckResponse.NOT_SERVING
        )
```

**Gateway Architecture:**

```python
class FastAPIApplication:
    """Multi-protocol gateway with unified error handling."""

    def __init__(self):
        # Protocol-specific routers
        self.rest_router = RESTAPIRouter()
        self.graphql_router = GraphQLRouter(schema)
        self.grpc_server = GRPCServer(services)
        self.soap_router = SOAPRouter(wsdl)
        self.sse_router = SSERouter()

        # Shared middleware stack
        self.middleware = [
            CORSMiddleware(),
            SecurityHeadersMiddleware(),
            CachingMiddleware(),
            TenantValidationMiddleware(),
            RequestLifecycleMiddleware(),
        ]

    def create_app(self) -> FastAPI:
        """Create unified FastAPI application."""
        app = FastAPI(title="Medical KG Gateway", version="1.0.0")

        # Add middleware
        for middleware in self.middleware:
            app.add_middleware(middleware)

        # Include protocol routers
        app.include_router(self.rest_router, prefix="/v1")
        app.include_router(self.graphql_router, prefix="/graphql")
        app.include_router(self.soap_router, prefix="/soap")
        app.include_router(self.sse_router, prefix="/events")

        return app
```

**Error Translation & Response Formatting:**

```python
class ProblemDetail(BaseModel):
    """RFC 7807 compliant error response format."""

    type: str = Field(description="Error type URI")
    title: str = Field(description="Human-readable error title")
    status: int = Field(description="HTTP status code")
    detail: str = Field(description="Detailed error description")
    instance: str | None = Field(description="Request instance identifier")
    correlation_id: str | None = Field(description="Request correlation ID")
    extensions: dict[str, Any] = Field(default_factory=dict)

def create_problem_response(detail: ProblemDetail) -> JSONResponse:
    """Create standardized error response."""
    payload = detail.model_dump(mode="json")
    status = payload.get("status", 500)
    headers = {}

    # Add retry-after for rate limiting
    retry_after = detail.extensions.get("retry_after")
    if retry_after:
        headers["Retry-After"] = str(int(retry_after))

    return JSONResponse(
        payload,
        status_code=status,
        media_type="application/problem+json",
        headers=headers
    )
```

**Middleware Stack Architecture:**

```python
# Comprehensive middleware pipeline
class GatewayMiddlewareStack:
    """Middleware stack with proper ordering and configuration."""

    def __init__(self, settings: GatewaySettings):
        self.middleware = [
            # Security & compliance
            SecurityHeadersMiddleware(settings.security.headers),
            CORSMiddleware(settings.security.cors),

            # Request processing
            TenantValidationMiddleware(settings.auth),
            RequestLifecycleMiddleware(settings.observability.logging),

            # Performance & caching
            CachingMiddleware(settings.caching),

            # Response processing
            JSONAPIResponseMiddleware(),
        ]

    def apply_to_app(self, app: FastAPI) -> None:
        """Apply middleware stack to FastAPI application."""
        for middleware in self.middleware:
            app.add_middleware(middleware)
```

### **Document Processing Pipeline**

The chunking system provides sophisticated document processing with multiple algorithms and strategies:

**Chunking Architecture:**

```python
class ChunkingService:
    """Main chunking service with registry-based chunker selection."""

    def __init__(self, registry: ChunkerRegistry):
        self.registry = registry
        self.chunkers = registry.get_available_chunkers()

    async def chunk_document(
        self,
        document: Document,
        strategy: str = "section",
        options: dict[str, Any] = None
    ) -> ChunkingResult:
        """Chunk document using specified strategy."""
        chunker = self.registry.get_chunker(strategy)
        chunks = await chunker.chunk(document, options or {})
        return ChunkingResult(chunks=chunks, strategy=strategy)
```

**Available Chunking Strategies:**

- **Section-based**: Semantic section boundary detection
- **Sentence-based**: Linguistic sentence boundary splitting
- **Token-based**: Fixed token count chunking
- **Table-aware**: Table structure preservation
- **Figure-aware**: Figure and caption grouping
- **Hybrid**: Multi-strategy combination

**Chunking Pipeline:**

```python
class ChunkingPipeline:
    """Multi-stage chunking pipeline with provenance tracking."""

    def __init__(self, stages: list[ChunkingStage]):
        self.stages = stages
        self.provenance_tracker = ProvenanceTracker()

    async def process(self, document: Document) -> ChunkingResult:
        """Process document through chunking pipeline."""
        context = ChunkingContext(document=document)

        for stage in self.stages:
            chunks = await stage.process(context)
            context.chunks.extend(chunks)
            self.provenance_tracker.record_stage(stage, chunks)

        return ChunkingResult(
            chunks=context.chunks,
            provenance=self.provenance_tracker.get_provenance()
        )
```

### **Embedding & Vector Services**

The embedding system provides multiple vector representation strategies with GPU acceleration:

**Embedding Architecture:**

```python
class EmbeddingService:
    """Multi-provider embedding service with namespace isolation."""

    def __init__(self, providers: dict[str, EmbeddingProvider]):
        self.providers = providers
        self.namespace_manager = NamespaceManager()

    async def embed_texts(
        self,
        texts: list[str],
        namespace: str,
        model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> EmbeddingResult:
        """Generate embeddings for text collection."""
        provider = self.providers.get(model)
        if not provider:
            raise ModelNotFoundError(f"Model {model} not available")

        # Namespace validation and isolation
        await self.namespace_manager.validate_access(namespace)

        # Batch processing for efficiency
        embeddings = await provider.embed_batch(texts, namespace)

        return EmbeddingResult(
            embeddings=embeddings,
            model=model,
            namespace=namespace
        )
```

**Embedding Strategies:**

- **Dense Vectors**: Sentence-BERT, OpenAI embeddings
- **Sparse Vectors**: SPLADE, BM25-based representations
- **Multi-Vector**: ColBERT-style token-level embeddings
- **Neural Sparse**: SPLADE v2 with neural components

**GPU Service Integration:**

```python
class GPUServiceManager:
    """GPU resource management for embedding services."""

    def __init__(self, gpu_config: GPUConfig):
        self.gpu_pools = {}  # Per-namespace GPU memory pools
        self.model_cache = {}  # Model loading optimization

    async def acquire_compute_resources(
        self,
        namespace: str,
        model_name: str,
        memory_requirement: int
    ) -> GPUComputeContext:
        """Acquire GPU resources for embedding computation."""
        pool = self._get_or_create_pool(namespace)
        return await pool.acquire(memory_requirement, model_name)
```

### **Knowledge Graph Integration**

The system uses Neo4j for graph storage with comprehensive schema management and validation:

**Graph Schema Definition:**

```python
class GraphSchema:
    """Neo4j schema with SHACL validation constraints."""

    node_schemas = {
        "Document": NodeSchema(
            labels=["Document"],
            properties={
                "id": PropertySchema(type="string", required=True),
                "title": PropertySchema(type="string", required=True),
                "content": PropertySchema(type="string", required=True),
                "source": PropertySchema(type="string", required=True),
                "created_at": PropertySchema(type="datetime", required=True),
            }
        ),
        "Entity": NodeSchema(
            labels=["Entity"],
            properties={
                "id": PropertySchema(type="string", required=True),
                "name": PropertySchema(type="string", required=True),
                "type": PropertySchema(type="string", required=True),
                "synonyms": PropertySchema(type="string[]", required=False),
            }
        ),
        "Claim": NodeSchema(
            labels=["Claim"],
            properties={
                "id": PropertySchema(type="string", required=True),
                "text": PropertySchema(type="string", required=True),
                "confidence": PropertySchema(type="float", required=True),
                "evidence": PropertySchema(type="string[]", required=False),
            }
        )
    }

    relationship_schemas = {
        "MENTIONS": RelationshipSchema(
            from_label="Document",
            to_label="Entity",
            properties={
                "confidence": PropertySchema(type="float", required=True),
                "span": PropertySchema(type="string", required=True),
            }
        ),
        "SUPPORTS": RelationshipSchema(
            from_label="Claim",
            to_label="Claim",
            properties={
                "strength": PropertySchema(type="float", required=True),
            }
        )
    }
```

**Cypher Query Templates:**

```python
class CypherTemplates:
    """Pre-compiled Cypher query templates for common operations."""

    FIND_DOCUMENTS_BY_ENTITY = """
    MATCH (d:Document)-[r:MENTIONS]->(e:Entity)
    WHERE e.name = $entity_name
    RETURN d, r, e
    ORDER BY r.confidence DESC
    """

    FIND_CLAIMS_BY_DOCUMENT = """
    MATCH (d:Document)-[:CONTAINS]->(c:Claim)
    WHERE d.id = $document_id
    RETURN c ORDER BY c.confidence DESC
    """

    FIND_ENTITY_RELATIONSHIPS = """
    MATCH (e1:Entity)-[r]-(e2:Entity)
    WHERE e1.id = $entity_id
    RETURN type(r) as relationship_type,
           e2.name as related_entity,
           r.confidence as confidence
    """
```

**SHACL Validation:**

```python
class ShaclValidator:
    """RDF shape constraint validation for graph integrity."""

    def __init__(self, shapes_file: str = "kg/shapes.ttl"):
        self.shapes_graph = Graph()
        self.shapes_graph.parse(shapes_file, format="turtle")

    async def validate_node(self, node_data: dict) -> ValidationResult:
        """Validate node data against SHACL constraints."""
        # Convert node data to RDF graph
        data_graph = self._node_to_rdf(node_data)

        # Validate against shapes
        conforms, results_graph, results_text = validate(
            data_graph, shacl_graph=self.shapes_graph
        )

        return ValidationResult(
            conforms=conforms,
            violations=self._parse_validation_results(results_graph)
        )
```

### **Orchestration & Pipeline Management**

The system uses Dagster for workflow orchestration with custom stage implementations:

**Pipeline Stage Architecture:**

```python
class BaseStage(ABC):
    """Abstract base for all pipeline stages."""

    @abstractmethod
    async def process(self, context: StageContext) -> StageResult:
        """Process stage with input/output tracking."""
        pass

    @abstractmethod
    def get_requirements(self) -> StageRequirements:
        """Define stage resource and dependency requirements."""
        pass

class IngestStage(BaseStage):
    """Data ingestion stage with adapter coordination."""

    async def process(self, context: StageContext) -> StageResult:
        adapter = self._get_adapter(context.source)
        result = await adapter.run(context.adapter_context)

        return StageResult(
            success=result.success,
            outputs={"documents": result.documents},
            metadata={"source": context.source}
        )

class ChunkStage(BaseStage):
    """Document chunking stage with multiple strategies."""

    async def process(self, context: StageContext) -> StageResult:
        documents = context.inputs["documents"]
        chunker = self._get_chunker(context.chunking_strategy)

        chunks = []
        for doc in documents:
            doc_chunks = await chunker.chunk(doc, context.chunking_options)
            chunks.extend(doc_chunks)

        return StageResult(
            success=True,
            outputs={"chunks": chunks},
            metadata={"chunk_count": len(chunks)}
        )
```

**Stage Plugin System:**

```python
class StagePluginManager:
    """Plugin system for stage registration and discovery."""

    def __init__(self):
        self.stages = {}
        self.factories = {}

    def register_stage(self, name: str, factory: StageFactory) -> None:
        """Register stage implementation."""
        self.factories[name] = factory

    async def create_stage(self, name: str, config: dict) -> BaseStage:
        """Create stage instance from configuration."""
        factory = self.factories.get(name)
        if not factory:
            raise StageNotFoundError(f"Stage {name} not registered")

        return await factory.create(config)

    def get_available_stages(self) -> list[str]:
        """List all registered stage types."""
        return list(self.factories.keys())
```

**Pipeline Configuration:**

```python
class PipelineConfig:
    """Declarative pipeline configuration."""

    name: str
    stages: list[StageConfig] = Field(description="Ordered stage configurations")
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

@dataclass
class StageConfig:
    """Individual stage configuration."""
    name: str
    type: str  # "ingest", "chunk", "embed", "extract", etc.
    config: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)
    retry_policy: RetryPolicy | None = None
```

### **API Documentation & Implementation**

**REST API Implementation (OpenAPI 3.1 + JSON:API 1.1):**

```python
# Core REST API endpoints with comprehensive error handling
@app.post("/v1/documents", response_model=DocumentResponse)
async def create_document(
    document: DocumentCreate,
    current_user: User = Depends(get_current_user)
) -> DocumentResponse:
    """Create a new document with full validation and audit trail."""
    # Request validation
    if not document.title.strip():
        raise HTTPException(status_code=422, detail="Title cannot be empty")

    # Authorization check
    if not current_user.has_permission("write:documents"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    # Business logic execution
    new_document = await document_service.create(
        title=document.title,
        content=document.content,
        source=document.source,
        metadata=document.metadata,
        user_id=current_user.id
    )

    # Audit logging
    await audit_service.log_action(
        user_id=current_user.id,
        action="document.create",
        resource_id=new_document.id,
        details={"title": document.title}
    )

    return DocumentResponse.from_orm(new_document)

@app.get("/v1/search", response_model=SearchResponse)
async def search_documents(
    q: str = Query(..., description="Search query"),
    strategy: str = Query("hybrid", description="Search strategy"),
    limit: int = Query(20, ge=1, le=100, description="Result limit"),
    offset: int = Query(0, ge=0, description="Result offset"),
    filters: str | None = Query(None, description="Filter expression")
) -> SearchResponse:
    """Multi-strategy search with advanced filtering."""
    # Query validation and parsing
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Strategy validation
    valid_strategies = ["bm25", "dense", "sparse", "hybrid"]
    if strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy. Must be one of: {valid_strategies}"
        )

    # Execute search with metrics tracking
    start_time = time.time()
    try:
        results = await search_service.search(
            query=q,
            strategy=strategy,
            limit=limit,
            offset=offset,
            filters=filters
        )

        search_time = time.time() - start_time

        # Metrics collection
        search_metrics.labels(strategy=strategy).observe(search_time)

        return SearchResponse(
            results=results,
            query=q,
            strategy=strategy,
            total=len(results),
            search_time=search_time
        )

    except SearchServiceError as e:
        # Error translation and logging
        logger.error(f"Search failed: {e}", query=q, strategy=strategy)
        raise HTTPException(status_code=500, detail="Search service unavailable")
```

**GraphQL Schema Implementation:**

```python
# Comprehensive GraphQL schema with type safety
@strawberry.type
class Document:
    """Document type for GraphQL API."""
    id: str
    title: str
    content: str
    source: str
    created_at: datetime
    updated_at: datetime

    @strawberry.field
    async def entities(self) -> list[Entity]:
        """Get entities mentioned in this document."""
        return await entity_service.get_by_document_id(self.id)

    @strawberry.field
    async def claims(self) -> list[Claim]:
        """Get claims extracted from this document."""
        return await claim_service.get_by_document_id(self.id)

@strawberry.type
class Query:
    """Root GraphQL query type."""

    @strawberry.field
    async def document(self, id: str) -> Document | None:
        """Get document by ID."""
        return await document_service.get_by_id(id)

    @strawberry.field
    async def documents(
        self,
        first: int = 20,
        after: str | None = None,
        filter: DocumentFilter | None = None
    ) -> DocumentConnection:
        """Get paginated list of documents."""
        return await document_service.list_documents(
            first=first,
            after=after,
            filter=filter
        )

    @strawberry.field
    async def search_documents(
        self,
        query: str,
        strategy: str = "hybrid",
        limit: int = 20
    ) -> list[Document]:
        """Search documents using specified strategy."""
        return await search_service.search_documents(query, strategy, limit)

@strawberry.type
class Mutation:
    """Root GraphQL mutation type."""

    @strawberry.mutation
    async def create_document(
        self,
        input: DocumentCreateInput
    ) -> Document:
        """Create a new document."""
        return await document_service.create(
            title=input.title,
            content=input.content,
            source=input.source,
            metadata=input.metadata
        )

    @strawberry.mutation
    async def update_document(
        self,
        id: str,
        input: DocumentUpdateInput
    ) -> Document:
        """Update an existing document."""
        return await document_service.update(id, **input.__dict__)

# Schema creation with performance optimizations
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    extensions=[
        DataLoaderExtension(),  # Batch loading optimization
        ValidationExtension(),  # Input validation
        MetricsExtension(),     # Performance monitoring
    ]
)
```

**gRPC Service Implementation:**

```python
# Protocol Buffers service definition
service DocumentService {
    rpc GetDocument(GetDocumentRequest) returns (GetDocumentResponse);
    rpc ListDocuments(ListDocumentsRequest) returns (stream Document);
    rpc CreateDocument(CreateDocumentRequest) returns (CreateDocumentResponse);
    rpc UpdateDocument(UpdateDocumentRequest) returns (UpdateDocumentResponse);
    rpc DeleteDocument(DeleteDocumentRequest) returns (DeleteDocumentResponse);
    rpc SearchDocuments(SearchDocumentsRequest) returns (stream Document);
}

message GetDocumentRequest {
    string id = 1;
    string tenant_id = 2;
}

message GetDocumentResponse {
    Document document = 1;
}

message Document {
    string id = 1;
    string title = 2;
    string content = 3;
    string source = 4;
    string type = 5;
    string status = 6;
    google.protobuf.Timestamp created_at = 7;
    google.protobuf.Timestamp updated_at = 8;
    google.protobuf.Struct metadata = 9;
}

# Python gRPC service implementation
class DocumentServicer(document_service_pb2_grpc.DocumentServiceServicer):
    """gRPC service implementation for document operations."""

    def __init__(self, document_service: DocumentService):
        self.document_service = document_service

    async def GetDocument(
        self,
        request: document_service_pb2.GetDocumentRequest,
        context: grpc.aio.ServicerContext
    ) -> document_service_pb2.GetDocumentResponse:
        """Get document by ID."""
        try:
            document = await self.document_service.get_by_id(request.id)

            if not document:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Document not found")
                return document_service_pb2.GetDocumentResponse()

            # Convert to protobuf message
            pb_document = self._document_to_proto(document)

            return document_service_pb2.GetDocumentResponse(document=pb_document)

        except Exception as e:
            logger.error(f"gRPC GetDocument error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return document_service_pb2.GetDocumentResponse()

    async def SearchDocuments(
        self,
        request: document_service_pb2.SearchDocumentsRequest,
        context: grpc.aio.ServicerContext
    ) -> AsyncIterator[document_service_pb2.Document]:
        """Stream search results."""
        try:
            documents = await self.document_service.search(
                query=request.query,
                strategy=request.strategy,
                limit=request.limit
            )

            for document in documents:
                pb_document = self._document_to_proto(document)
                yield pb_document

        except Exception as e:
            logger.error(f"gRPC SearchDocuments error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")

    def _document_to_proto(self, document: Document) -> document_service_pb2.Document:
        """Convert ORM document to protobuf message."""
        return document_service_pb2.Document(
            id=document.id,
            title=document.title,
            content=document.content,
            source=document.source,
            type=document.type,
            status=document.status,
            created_at=document.created_at,
            updated_at=document.updated_at,
            metadata=struct_pb2.Struct()
        )
```

### **Database Schema & Data Models**

**Multi-Database Architecture:**

```python
# PostgreSQL for relational data
class Document(Base):
    """Document model for PostgreSQL storage."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False, index=True)
    content = Column(Text, nullable=False)
    source = Column(String(100), nullable=False, index=True)
    type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), default="processing", index=True)
    metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    entities = relationship("DocumentEntity", back_populates="document")
    extractions = relationship("Extraction", back_populates="document")

class Entity(Base):
    """Entity model for PostgreSQL storage."""

    __tablename__ = "entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False, index=True)
    type = Column(String(50), nullable=False, index=True)
    synonyms = Column(JSONB, nullable=True)
    description = Column(Text, nullable=True)
    ontology_mappings = Column(JSONB, nullable=True)
    properties = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DocumentEntity(Base):
    """Many-to-many relationship between documents and entities."""

    __tablename__ = "document_entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), index=True)
    entity_id = Column(UUID(as_uuid=True), ForeignKey("entities.id"), index=True)
    confidence = Column(Float, nullable=False)
    span_start = Column(Integer, nullable=False)
    span_end = Column(Integer, nullable=False)
    span_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="entities")
    entity = relationship("Entity")
```

**Neo4j Graph Schema:**

```python
# Graph database schema for relationships and knowledge representation
class GraphSchema:
    """Neo4j schema definition with constraints."""

    # Node constraints
    DOCUMENT_ID_CONSTRAINT = """
    CREATE CONSTRAINT document_id_unique IF NOT EXISTS
    FOR (d:Document) REQUIRE d.id IS UNIQUE
    """

    ENTITY_ID_CONSTRAINT = """
    CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
    FOR (e:Entity) REQUIRE e.id IS UNIQUE
    """

    CLAIM_ID_CONSTRAINT = """
    CREATE CONSTRAINT claim_id_unique IF NOT EXISTS
    FOR (c:Claim) REQUIRE c.id IS UNIQUE
    """

    # Relationship definitions
    DOCUMENT_ENTITY_RELATIONSHIP = """
    CREATE (d:Document)-[:MENTIONS {
        confidence: $confidence,
        span: $span,
        created_at: datetime()
    }]->(e:Entity)
    """

    ENTITY_RELATIONSHIP = """
    CREATE (e1:Entity)-[:TREATS {
        confidence: $confidence,
        evidence: $evidence
    }]->(e2:Entity)
    """

    CLAIM_SUPPORT_RELATIONSHIP = """
    CREATE (c1:Claim)-[:SUPPORTS {
        strength: $strength,
        evidence: $evidence
    }]->(c2:Claim)
    """
```

**Vector Storage Schema (Qdrant):**

```python
# Vector database schema for similarity search
class VectorStoreSchema:
    """Qdrant collection schema for vector storage."""

    DOCUMENT_CHUNKS_COLLECTION = {
        "name": "document_chunks",
        "vectors": {
            "size": 768,  # Embedding dimension
            "distance": "Cosine"
        },
        "payload": {
            "document_id": "keyword",
            "chunk_id": "keyword",
            "chunk_index": "integer",
            "text": "text",
            "chunk_type": "keyword",
            "metadata": "object"
        }
    }

    ENTITY_EMBEDDINGS_COLLECTION = {
        "name": "entity_embeddings",
        "vectors": {
            "size": 768,
            "distance": "Cosine"
        },
        "payload": {
            "entity_id": "keyword",
            "entity_name": "text",
            "entity_type": "keyword",
            "context": "text"
        }
    }
```

### **Observability & Monitoring Stack**

**Comprehensive observability implementation:**

**Structured Logging:**

```python
# Structured logging with correlation tracking
logger = structlog.get_logger(__name__)

class RequestLifecycleMiddleware:
    """Middleware for request correlation and lifecycle tracking."""

    def __init__(self, app: FastAPI, correlation_header: str = "X-Correlation-ID"):
        self.app = app
        self.correlation_header = correlation_header

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract or generate correlation ID
        headers = dict(scope.get("headers", []))
        correlation_id = self._get_correlation_id(headers)

        # Set up request context
        context = {
            "correlation_id": correlation_id,
            "request_id": str(uuid.uuid4()),
            "start_time": time.time(),
        }

        # Bind logger with request context
        bound_logger = logger.bind(**context)

        with bound_logger.contextvars():
            # Process request with logging context
            await self.app(scope, receive, send)

    def _get_correlation_id(self, headers: dict) -> str:
        """Extract correlation ID from headers or generate new one."""
        correlation_bytes = headers.get(self.correlation_header.encode())
        if correlation_bytes:
            return correlation_bytes.decode()

        return str(uuid.uuid4())
```

**Metrics Collection:**

```python
# Prometheus metrics for comprehensive monitoring
class GatewayMetrics:
    """Comprehensive metrics collection for gateway operations."""

    def __init__(self):
        # Request metrics
        self.request_duration = Histogram(
            "gateway_request_duration_seconds",
            "Request processing duration",
            labelnames=["method", "endpoint", "status_code"]
        )

        self.request_count = Counter(
            "gateway_requests_total",
            "Total requests processed",
            labelnames=["method", "endpoint", "status_code"]
        )

        # Business metrics
        self.documents_processed = Counter(
            "gateway_documents_processed_total",
            "Documents processed by gateway",
            labelnames=["operation", "status"]
        )

        self.search_queries = Counter(
            "gateway_search_queries_total",
            "Search queries executed",
            labelnames=["strategy", "status"]
        )

        # Performance metrics
        self.active_connections = Gauge(
            "gateway_active_connections",
            "Number of active connections"
        )

        self.response_size = Histogram(
            "gateway_response_size_bytes",
            "Response size distribution"
        )

    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record request metrics."""
        self.request_duration.labels(
            method=method, endpoint=endpoint, status_code=status_code
        ).observe(duration)

        self.request_count.labels(
            method=method, endpoint=endpoint, status_code=status_code
        ).inc()

    def record_search(self, strategy: str, duration: float, result_count: int):
        """Record search operation metrics."""
        self.search_queries.labels(strategy=strategy, status="success").inc()

        # Additional search-specific metrics
        search_duration_histogram.labels(strategy=strategy).observe(duration)
        search_results_histogram.labels(strategy=strategy).observe(result_count)
```

**Distributed Tracing:**

```python
# OpenTelemetry tracing integration
class TracingMiddleware:
    """Middleware for distributed tracing with OpenTelemetry."""

    def __init__(self, app: FastAPI, service_name: str = "medical-kg-gateway"):
        self.app = app
        self.service_name = service_name

        # Set up tracing
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)

        # Configure exporters
        if settings.observability.tracing.jaeger_enabled:
            jaeger_exporter = JaegerExporter(
                agent_host_name=settings.observability.tracing.jaeger_host,
                agent_port=settings.observability.tracing.jaeger_port,
            )
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Create span for request
        with self.tracer.start_as_current_span(
            f"{scope['method']} {scope['path']}",
            kind=SpanKind.SERVER,
        ) as span:
            # Add request attributes
            span.set_attribute("http.method", scope["method"])
            span.set_attribute("http.url", scope["path"])
            span.set_attribute("http.user_agent", self._get_header(scope, "user-agent"))

            # Extract trace context from headers
            trace_context = self._extract_trace_context(scope)
            if trace_context:
                span.add_link(trace_context)

            # Process request
            await self.app(scope, receive, send)
```

**Health Check System:**

```python
class HealthService:
    """Comprehensive health check service."""

    def __init__(self, checks: dict[str, Callable[[], CheckResult]]):
        self.checks = checks

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check."""
        results = {}

        for check_name, check_func in self.checks.items():
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, check_func
                )
                results[check_name] = {
                    "status": result.status,
                    "timestamp": datetime.utcnow().isoformat(),
                    "response_time": result.response_time,
                }

                if result.detail:
                    results[check_name]["detail"] = result.detail

            except Exception as e:
                results[check_name] = {
                    "status": "error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e),
                }

        return results

@app.get("/health")
async def health_endpoint():
    """Health check endpoint."""
    results = await health_service.health_check()
    status_code = 200 if all(r["status"] == "ok" for r in results.values()) else 503

    return JSONResponse(
        {
            "status": "healthy" if status_code == 200 else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "checks": results,
        },
        status_code=status_code
    )
```

### **Security Implementation**

**Multi-layered security architecture:**

**Authentication & Authorization:**

```python
class AuthService:
    """Comprehensive authentication and authorization service."""

    def __init__(self, jwt_service: JWTService, user_service: UserService):
        self.jwt_service = jwt_service
        self.user_service = user_service

    async def authenticate_user(self, username: str, password: str) -> User | None:
        """Authenticate user credentials."""
        user = await self.user_service.get_by_username(username)

        if not user or not self._verify_password(password, user.hashed_password):
            return None

        return user

    async def create_access_token(
        self,
        user: User,
        expires_delta: timedelta | None = None
    ) -> str:
        """Create JWT access token."""
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))

        token_data = {
            "sub": user.username,
            "scopes": user.scopes,
            "tenant_id": user.tenant_id,
            "exp": expire,
        }

        return self.jwt_service.encode_token(token_data)

    async def get_current_user(self, token: str) -> User:
        """Extract and validate current user from JWT token."""
        try:
            payload = self.jwt_service.decode_token(token)
            username: str = payload.get("sub")

            if username is None:
                raise HTTPException(status_code=401, detail="Invalid token")

            user = await self.user_service.get_by_username(username)
            if user is None:
                raise HTTPException(status_code=401, detail="User not found")

            return user

        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)
```

**Rate Limiting & Abuse Prevention:**

```python
class RateLimitService:
    """Multi-level rate limiting with tenant isolation."""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.limiter = AsyncLimiter(100, 1)  # 100 requests per second

    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window: int
    ) -> RateLimitResult:
        """Check rate limit for given key."""
        current = await self.redis.incr(f"rate_limit:{key}")
        if current == 1:
            await self.redis.expire(f"rate_limit:{key}", window)

        if current > limit:
            return RateLimitResult(
                allowed=False,
                limit=limit,
                window=window,
                retry_after=window - (time.time() % window)
            )

        return RateLimitResult(allowed=True, remaining=limit - current)

    async def get_rate_limit_key(
        self,
        request: Request,
        user: User | None = None
    ) -> str:
        """Generate rate limit key based on request context."""
        # Multi-dimensional rate limiting
        client_ip = self._get_client_ip(request)
        user_id = user.id if user else "anonymous"
        endpoint = request.url.path
        tenant_id = getattr(request.state, "tenant_id", "default")

        return f"{client_ip}:{user_id}:{endpoint}:{tenant_id}"

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to requests."""
    # Skip rate limiting for health checks and internal endpoints
    if request.url.path in ["/health", "/ready", "/metrics"]:
        return await call_next(request)

    # Get current user (if authenticated)
    user = getattr(request.state, "user", None)

    # Generate rate limit key
    rate_limit_key = await rate_limit_service.get_rate_limit_key(request, user)

    # Check rate limit
    result = await rate_limit_service.check_rate_limit(
        rate_limit_key, limit=100, window=60  # 100 requests per minute
    )

    if not result.allowed:
        # Return rate limit response
        return JSONResponse(
            {
                "error": {
                    "type": "rate_limit_exceeded",
                    "message": "Too many requests",
                    "retry_after": int(result.retry_after)
                }
            },
            status_code=429,
            headers={"Retry-After": str(int(result.retry_after))}
        )

    # Add rate limit headers to response
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(result.limit)
    response.headers["X-RateLimit-Remaining"] = str(result.remaining)
    response.headers["X-RateLimit-Reset"] = str(int(time.time() + result.retry_after))

    return response
```

**Data Protection & Encryption:**

```python
class DataProtectionService:
    """Comprehensive data protection and encryption."""

    def __init__(self, encryption_key: str):
        self.cipher = Fernet(encryption_key.encode())

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data before storage."""
        encrypted = self.cipher.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data after retrieval."""
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted = self.cipher.decrypt(encrypted_bytes)
        return decrypted.decode()

    def hash_pii_data(self, data: str) -> str:
        """Hash PII data for anonymization."""
        return hashlib.sha256(data.encode()).hexdigest()

    def mask_sensitive_fields(self, data: dict[str, Any]) -> dict[str, Any]:
        """Mask sensitive fields in data structures."""
        sensitive_fields = {"password", "ssn", "credit_card", "api_key"}

        def mask_value(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: mask_value(v) if k not in sensitive_fields else "***" for k, v in value.items()}
            elif isinstance(value, str) and any(field in value.lower() for field in sensitive_fields):
                return "***"
            return value

        return mask_value(data)
```

### **Configuration Management**

**Comprehensive configuration system:**

**Environment-Based Configuration:**

```python
class Settings(BaseSettings):
    """Main application configuration with environment support."""

    # Core application settings
    app_name: str = "Medical KG Gateway"
    version: str = "1.0.0"
    environment: Environment = Field(default=Environment.DEV)

    # Database configuration
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)

    # External services
    adapters: dict[str, AdapterConfig] = Field(default_factory=dict)
    embedding_providers: dict[str, EmbeddingProviderConfig] = Field(default_factory=dict)

    # Security settings
    auth: AuthSettings = Field(default_factory=AuthSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)

    # Performance settings
    caching: CacheSettings = Field(default_factory=CacheSettings)
    rate_limiting: RateLimitSettings = Field(default_factory=RateLimitSettings)

    # Observability
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    # Feature flags
    features: FeatureFlagSettings = Field(default_factory=FeatureFlagSettings)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_nested_delimiter = "__"

@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
```

**Feature Flag System:**

```python
class FeatureFlagService:
    """Dynamic feature flag management."""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    async def is_enabled(self, feature: str, tenant_id: str | None = None) -> bool:
        """Check if feature is enabled for tenant."""
        key = f"feature:{feature}"
        if tenant_id:
            key = f"feature:{feature}:{tenant_id}"

        value = await self.redis.get(key)
        return value == "true" if value else False

    async def enable_feature(self, feature: str, tenant_id: str | None = None) -> None:
        """Enable feature for tenant."""
        key = f"feature:{feature}"
        if tenant_id:
            key = f"feature:{feature}:{tenant_id}"

        await self.redis.set(key, "true", ex=86400)  # 24 hour TTL

    async def disable_feature(self, feature: str, tenant_id: str | None = None) -> None:
        """Disable feature for tenant."""
        key = f"feature:{feature}"
        if tenant_id:
            key = f"feature:{feature}:{tenant_id}"

        await self.redis.delete(key)
```

## 📚 Comprehensive Documentation Library

### Documentation Overview

This section provides the complete contents of all documentation files in the `docs/` directory, organized by category for easy reference. Each document includes its original content with proper attribution and links to the source files.

### Table of Contents - Documentation Library

#### **🏗️ Architecture Documentation**

- **[docs/index.md](docs/index.md)** - Main documentation index and overview
- **[docs/architecture/overview.md](docs/architecture/overview.md)** - High-level system architecture overview
- **[docs/architecture/foundation.md](docs/architecture/foundation.md)** - Foundation infrastructure architecture details

#### **📋 Architecture Decision Records (ADRs)**

- **[docs/adr/0001-coordinator-architecture.md](docs/adr/0001-coordinator-architecture.md)** - Coordinator pattern architecture decision
- **[docs/adr/0002-section-headers.md](docs/adr/0002-section-headers.md)** - Code organization standards
- **[docs/adr/0003-error-translation-strategy.md](docs/adr/0003-error-translation-strategy.md)** - Error handling and translation strategy
- **[docs/adr/0004-google-style-docstrings.md](docs/adr/0004-google-style-docstrings.md)** - Documentation standards
- **[docs/adr/0005-repository-documentation-standards.md](docs/adr/0005-repository-documentation-standards.md)** - Repository documentation guidelines
- **[docs/adr/0006-domain-specific-section-headers.md](docs/adr/0006-domain-specific-section-headers.md)** - Domain-specific code organization
- **[docs/adr/0007-automated-documentation-enforcement.md](docs/adr/0007-automated-documentation-enforcement.md)** - Documentation validation automation
- **[docs/adr/0008-type-hint-modernization.md](docs/adr/0008-type-hint-modernization.md)** - Type hint modernization strategy

#### **🔌 API Documentation**

- **[docs/api/adapters.md](docs/api/adapters.md)** - External data source adapter API reference
- **[docs/api/api_reference.md](docs/api/api_reference.md)** - Core API components and interfaces
- **[docs/api/coordinators.md](docs/api/coordinators.md)** - Coordinator pattern API documentation
- **[docs/api/embedding.md](docs/api/embedding.md)** - Embedding service API reference
- **[docs/api/gateway.md](docs/api/gateway.md)** - Multi-protocol API gateway documentation
- **[docs/api/kg.md](docs/api/kg.md)** - Knowledge graph API reference
- **[docs/api/orchestration.md](docs/api/orchestration.md)** - Workflow orchestration API documentation
- **[docs/api/services.md](docs/api/services.md)** - Service layer API reference
- **[docs/api/storage.md](docs/api/storage.md)** - Storage abstraction API documentation
- **[docs/api/utils.md](docs/api/utils.md)** - Utility functions API reference
- **[docs/api/validation.md](docs/api/validation.md)** - Data validation API reference

#### **🌐 Protocol Specifications**

- **[docs/api-portal.md](docs/api-portal.md)** - API portal and developer portal documentation
- **[docs/openapi.yaml](docs/openapi.yaml)** - OpenAPI 3.1 specification for REST APIs
- **[docs/schema.graphql](docs/schema.graphql)** - GraphQL schema definition
- **[docs/asyncapi.yaml](docs/asyncapi.yaml)** - AsyncAPI specification for event-driven APIs

#### **✂️ Chunking Documentation**

- **[docs/chunking/AdapterGuide.md](docs/chunking/AdapterGuide.md)** - Chunking adapter integration guide
- **[docs/chunking/API.md](docs/chunking/API.md)** - Chunking service API documentation
- **[docs/chunking/Chunkers.md](docs/chunking/Chunkers.md)** - Available chunking algorithms and strategies
- **[docs/chunking/ConfigurationExamples.md](docs/chunking/ConfigurationExamples.md)** - Chunking configuration examples
- **[docs/chunking/Evaluation.md](docs/chunking/Evaluation.md)** - Chunking quality evaluation metrics
- **[docs/chunking/Setup.md](docs/chunking/Setup.md)** - Chunking environment and dependency setup

#### **🤝 Contributing Guidelines**

- **[docs/contributing/documentation_standards.md](docs/contributing/documentation_standards.md)** - Documentation contribution standards

#### **🚀 DevOps Documentation**

- **[docs/devops/ci-cd.md](docs/devops/ci-cd.md)** - Continuous integration and deployment pipeline
- **[docs/devops/kubernetes.md](docs/devops/kubernetes.md)** - Kubernetes deployment and operations
- **[docs/devops/local-environments.md](docs/devops/local-environments.md)** - Local development environment setup
- **[docs/devops/observability.md](docs/devops/observability.md)** - Monitoring, logging, and observability stack
- **[docs/devops/vllm-deployment.md](docs/devops/vllm-deployment.md)** - VLLM GPU service deployment guide

#### **📊 Diagram Documentation**

- **[docs/diagrams/](docs/diagrams/)** - System architecture and flow diagrams (Mermaid format)

#### **🎮 GPU Microservices**

- **[docs/gpu-microservices.md](docs/gpu-microservices.md)** - GPU-accelerated microservices architecture

#### **📖 Guides and Tutorials**

- **[docs/guides/adapter-sdk.md](docs/guides/adapter-sdk.md)** - Adapter SDK development guide
- **[docs/guides/adapters_development_guide.md](docs/guides/adapters_development_guide.md)** - Custom adapter development
- **[docs/guides/chunking.md](docs/guides/chunking.md)** - Document chunking strategies and best practices
- **[docs/guides/chunking-profiles.md](docs/guides/chunking-profiles.md)** - Predefined chunking configurations
- **[docs/guides/ci_cd_pipeline.md](docs/guides/ci_cd_pipeline.md)** - CI/CD pipeline development and maintenance
- **[docs/guides/code_review_guidelines.md](docs/guides/code_review_guidelines.md)** - Code review process and standards
- **[docs/guides/compliance_documentation.md](docs/guides/compliance_documentation.md)** - Regulatory compliance documentation
- **[docs/guides/configuration_reference.md](docs/guides/configuration_reference.md)** - Configuration management reference
- **[docs/guides/configuration_validation.md](docs/guides/configuration_validation.md)** - Configuration validation and testing
- **[docs/guides/data-models.md](docs/guides/data-models.md)** - Data model definitions and relationships
- **[docs/guides/deployment_overview.md](docs/guides/deployment_overview.md)** - Deployment architecture overview
- **[docs/guides/deployment_procedures.md](docs/guides/deployment_procedures.md)** - Step-by-step deployment procedures
- **[docs/guides/developer_guide.md](docs/guides/developer_guide.md)** - Comprehensive developer guide
- **[docs/guides/development_workflow.md](docs/guides/development_workflow.md)** - Development workflow and processes
- **[docs/guides/disaster_recovery_plan.md](docs/guides/disaster_recovery_plan.md)** - Disaster recovery procedures
- **[docs/guides/embedding_adapters.md](docs/guides/embedding_adapters.md)** - Embedding adapter implementations
- **[docs/guides/embedding_catalog.md](docs/guides/embedding_catalog.md)** - Available embedding models and services
- **[docs/guides/embedding_migration.md](docs/guides/embedding_migration.md)** - Embedding system migration guide
- **[docs/guides/embedding_namespace_policy.md](docs/guides/embedding_namespace_policy.md)** - Embedding namespace management
- **[docs/guides/environment_variables.md](docs/guides/environment_variables.md)** - Environment variable reference
- **[docs/guides/gateway_development_guide.md](docs/guides/gateway_development_guide.md)** - API gateway development
- **[docs/guides/git_workflow.md](docs/guides/git_workflow.md)** - Git workflow and branching strategy
- **[docs/guides/hot_reload_setup.md](docs/guides/hot_reload_setup.md)** - Development hot reload configuration
- **[docs/guides/ide_configuration.md](docs/guides/ide_configuration.md)** - IDE setup and configuration
- **[docs/guides/infrastructure_requirements.md](docs/guides/infrastructure_requirements.md)** - Infrastructure requirements and sizing
- **[docs/guides/maintenance_procedures.md](docs/guides/maintenance_procedures.md)** - System maintenance procedures
- **[docs/guides/migration_procedures.md](docs/guides/migration_procedures.md)** - Database and system migration procedures
- **[docs/guides/monitoring_logging.md](docs/guides/monitoring_logging.md)** - Monitoring and logging configuration
- **[docs/guides/orchestration-pipelines.md](docs/guides/orchestration-pipelines.md)** - Workflow orchestration patterns
- **[docs/guides/pipeline_extension_guide.md](docs/guides/pipeline_extension_guide.md)** - Pipeline extension and customization
- **[docs/guides/pipeline_state_management.md](docs/guides/pipeline_state_management.md)** - Pipeline state management strategies
- **[docs/guides/repository_extension_guide.md](docs/guides/repository_extension_guide.md)** - Repository structure extension
- **[docs/guides/rollback_procedures.md](docs/guides/rollback_procedures.md)** - System rollback procedures
- **[docs/guides/secret_management.md](docs/guides/secret_management.md)** - Secrets and credentials management
- **[docs/guides/security_considerations.md](docs/guides/security_considerations.md)** - Security best practices and considerations
- **[docs/guides/services_development_guide.md](docs/guides/services_development_guide.md)** - Microservice development guide
- **[docs/guides/test_data_setup.md](docs/guides/test_data_setup.md)** - Test data preparation and management
- **[docs/guides/training_materials.md](docs/guides/training_materials.md)** - Developer training materials
- **[docs/guides/troubleshooting_guide.md](docs/guides/troubleshooting_guide.md)** - Common issues and solutions
- **[docs/guides/user_guide.md](docs/guides/user_guide.md)** - End-user guide and documentation
- **[docs/guides/vector_store_overview.md](docs/guides/vector_store_overview.md)** - Vector storage system overview

#### **🔍 Retrieval Documentation**

- **[docs/guides/retrieval/developer-guide.md](docs/guides/retrieval/developer-guide.md)** - Retrieval system development guide
- **[docs/guides/retrieval/user-guide.md](docs/guides/retrieval/user-guide.md)** - Retrieval system user guide

#### **🛠️ Operations Documentation**

- **[docs/operational-runbook.md](docs/operational-runbook.md)** - Operational runbook and procedures
- **[docs/operations/embedding_rollout.md](docs/operations/embedding_rollout.md)** - Embedding service rollout procedures
- **[docs/operations/legacy_embedding_decommission.md](docs/operations/legacy_embedding_decommission.md)** - Legacy embedding system decommissioning
- **[docs/operations/retrieval-rollout.md](docs/operations/retrieval-rollout.md)** - Retrieval system rollout procedures
- **[docs/operations/rollback_drills.md](docs/operations/rollback_drills.md)** - Rollback drill procedures
- **[docs/operations/tenant_isolation_pen_test.md](docs/operations/tenant_isolation_pen_test.md)** - Tenant isolation penetration testing

#### **🔄 Reranking Documentation**

- **[docs/reranking/guide.md](docs/reranking/guide.md)** - Search result reranking strategies

#### **📋 Runbooks**

- **[docs/runbooks/embeddings_service_runbook.md](docs/runbooks/embeddings_service_runbook.md)** - Embedding service operational runbook
- **[docs/runbooks/mineru-two-phase-gate.md](docs/runbooks/mineru-two-phase-gate.md)** - MinerU service deployment runbook
- **[docs/runbooks/vllm-server-restart.md](docs/runbooks/vllm-server-restart.md)** - VLLM server restart procedures

#### **💾 Storage Documentation**

- **[docs/storage-architecture.md](docs/storage-architecture.md)** - Storage architecture and design
- **[docs/storage-quickstart.md](docs/storage-quickstart.md)** - Storage system quick start guide

#### **🔧 Templates**

- **[docs/templates/rollback_incident_template.md](docs/templates/rollback_incident_template.md)** - Incident response and rollback template

#### **🔍 Troubleshooting**

- **[docs/troubleshooting/pipeline_issues.md](docs/troubleshooting/pipeline_issues.md)** - Pipeline and workflow troubleshooting
- **[docs/troubleshooting/vllm-connectivity.md](docs/troubleshooting/vllm-connectivity.md)** - VLLM connectivity issues and solutions

### Core Documentation Files

#### **[docs/index.md](docs/index.md)**

```
# Medical KG Knowledge Platform

Welcome to the Medical Knowledge Graph platform documentation. This site aggregates specifications, runbooks, and operational guides that complement the OpenSpec change proposals.

## Key Sections

- **Architecture** – High-level system design and rationale.
- **DevOps & Observability** – CI/CD pipelines, infrastructure-as-code, and monitoring patterns.
- **API References** – REST, GraphQL, gRPC, and AsyncAPI contracts generated from the gateway.
- **Guides** – Hands-on tutorials and workflows for operating and extending the platform.

Use the navigation sidebar to explore the content. All documentation is generated with MkDocs Material and published automatically via GitHub Actions.
```

#### **[docs/architecture/overview.md](docs/architecture/overview.md)**

```
# Architecture Overview

The Medical KG platform is composed of multiple services connected through a multi-protocol gateway.

- **Gateway** – FastAPI application exposing REST, GraphQL, gRPC, SOAP, and SSE protocols. Shared services and adapters live in `src/Medical_KG_rev/gateway`.
- **Ingestion Pipeline** – Kafka-backed orchestration layer that coordinates adapters and workers defined in `src/Medical_KG_rev/orchestration`.
- **Storage** – Neo4j for the knowledge graph and OpenSearch for indexing and retrieval.
- **ML/GPU Services** – Embedding and extraction workloads offload to GPU-enabled microservices.
- **Observability** – Prometheus, Grafana, Jaeger, Sentry, and Loki deliver unified telemetry across the stack.

Refer to `docs/architecture` and the Engineering Blueprint PDF for deeper diagrams and sequence flows.
```

#### **[docs/architecture/foundation.md](docs/architecture/foundation.md)**

```
# Foundation Infrastructure Architecture

The foundation layer establishes the shared building blocks used by every other
OpenSpec change. It provides:

- **Federated Intermediate Representation (IR)** built on Pydantic models.
- **Domain overlays** for medical (FHIR-aligned), finance (XBRL) and legal
  (LegalDocML) content.
- **Configuration management** backed by Pydantic Settings with optional Vault
  integration and feature flag support.
- **Shared utilities** for HTTP access, structured logging, telemetry, span
  manipulation and identifier generation.
- **Adapter SDK** enabling declarative data source integrations with lifecycle
  hooks and testing helpers.
- **Storage abstractions** that decouple application logic from concrete
  backends while supporting async usage patterns.

Subsequent architectural layers (gateway, adapters, orchestration, GPU
services, retrieval, security and observability) build on these primitives.
```

#### **[docs/adr/0001-coordinator-architecture.md](docs/adr/0001-coordinator-architecture.md)**

```
# ADR-0001: Coordinator Architecture

## Status

Accepted

## Context

The Medical KG pipeline needed a clear separation between protocol handlers (REST/GraphQL/gRPC) and domain logic. The existing architecture had protocol-specific code mixed with business logic, making it difficult to:

- Test domain logic independently of protocol concerns
- Support multiple protocols without duplicating business logic
- Maintain consistent error handling across different protocols
- Implement protocol-agnostic metrics and monitoring

The system required a layer that could:

- Coordinate operations between protocol handlers and domain services
- Manage job lifecycle (creation, tracking, completion, failure)
- Translate domain exceptions to protocol-appropriate error responses
- Emit consistent metrics and logging across all protocols
- Provide a uniform interface for different operation types (chunking, embedding, retrieval)

## Decision

We will introduce a **Coordinator Layer** that sits between the gateway services and domain logic. This layer will:

1. **Implement the Coordinator Pattern**: Each coordinator manages a specific type of operation (chunking, embedding, retrieval)
2. **Provide Protocol Agnostic Interface**: Coordinators expose a uniform interface that can be used by any protocol handler
3. **Manage Job Lifecycle**: Track job creation, execution, completion, and failure states
4. **Handle Error Translation**: Convert domain exceptions to HTTP problem details (RFC 7807)
5. **Emit Metrics**: Provide consistent metrics emission across all operations
6. **Support Resilience**: Implement circuit breakers, rate limiting, and retry logic

### Coordinator Interface

```python
class BaseCoordinator[RequestType, ResultType]:
    """Base coordinator interface for all operation types."""

    def execute(self, request: RequestType) -> ResultType:
        """Execute operation with job lifecycle management."""
        pass

    def _execute(self, request: RequestType, **kwargs) -> ResultType:
        """Subclass implementation of operation logic."""
        pass
```

### Coordinator Responsibilities

- **Job Lifecycle Management**: Create, track, and update job states
- **Request Validation**: Validate incoming requests before processing
- **Domain Service Coordination**: Delegate to appropriate domain services
- **Error Translation**: Convert domain exceptions to coordinator errors
- **Metrics Emission**: Track operation attempts, failures, and duration
- **Resilience**: Implement circuit breakers and rate limiting

## Consequences

### Positive

- **Better Testability**: Domain logic can be tested independently of protocol concerns
- **Protocol Independence**: Business logic is decoupled from specific protocols
- **Consistent Error Handling**: Uniform error translation across all protocols
- **Centralized Metrics**: Consistent metrics emission and monitoring
- **Improved Maintainability**: Clear separation of concerns and responsibilities
- **Extensibility**: Easy to add new operation types and protocols

### Negative

- **Additional Complexity**: Introduces another layer in the architecture
- **Performance Overhead**: Additional method calls and object creation
- **Learning Curve**: Developers need to understand the coordinator pattern
- **Code Duplication**: Some common logic may be duplicated across coordinators

### Risks

- **Over-Engineering**: Risk of making the architecture too complex for simple operations
- **Performance Impact**: Additional layers may impact performance for high-throughput scenarios
- **Maintenance Burden**: More components to maintain and update

### Mitigation

- **Performance Testing**: Benchmark coordinator overhead and optimize critical paths
- **Documentation**: Provide comprehensive documentation and examples
- **Code Review**: Ensure coordinators follow established patterns and don't duplicate logic
- **Monitoring**: Track coordinator performance and error rates

## Implementation

### Phase 1: Base Infrastructure

- Implement `BaseCoordinator` abstract class
- Create `CoordinatorRequest` and `CoordinatorResult` base classes
- Implement `JobLifecycleManager` for job tracking
- Add error translation infrastructure

### Phase 2: Existing Operations

- Refactor chunking operations to use `ChunkingCoordinator`
- Refactor embedding operations to use `EmbeddingCoordinator`
- Update gateway services to use coordinators
- Add comprehensive documentation

### Phase 3: New Operations

- Implement coordinators for new operation types
- Add orchestration coordinators for pipeline execution
- Extend error translation for new domains

## Examples

### Chunking Coordinator

```python
class ChunkingCoordinator(BaseCoordinator[ChunkingRequest, ChunkingResult]):
    """Coordinates synchronous chunking operations."""

    def _execute(self, request: ChunkingRequest) -> ChunkingResult:
        # Create job entry
        job_id = self._lifecycle.create_job(request.tenant_id, "chunk")

        try:
            # Execute chunking
            chunks = self._chunker.chunk(request)

            # Assemble result
            result = ChunkingResult(
                job_id=job_id,
                chunks=chunks,
                duration_s=0.0
            )

            # Mark job completed
            self._lifecycle.mark_completed(job_id, result.metadata)

            return result

        except Exception as exc:
            # Translate and record error
            error = self._translate_error(job_id, request, exc)
            self._lifecycle.mark_failed(job_id, str(exc))
            raise error
```

### Error Translation

```python
class ChunkingErrorTranslator:
    """Translates chunking exceptions to coordinator errors."""

    def translate(self, job_id: str, request: ChunkingRequest, exc: Exception) -> CoordinatorError:
        if isinstance(exc, ProfileNotFoundError):
            return CoordinatorError(
                "Chunking profile not found",
                status_code=400,
                problem_type="profile-not-found",
                detail=f"Profile '{exc.profile_name}' does not exist"
            )
        elif isinstance(exc, ChunkingUnavailableError):
            return CoordinatorError(
                "Chunking service unavailable",
                status_code=503,
                problem_type="service-unavailable",
                detail="Chunking service is temporarily unavailable",
                context={"retry_after": 30}
            )
        # ... other exception types
```

## References

- [Coordinator Pattern](https://martinfowler.com/eaaCatalog/coordinator.html)
- [RFC 7807 Problem Details for HTTP APIs](https://tools.ietf.org/html/rfc7807)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Job Lifecycle Management](https://docs.aws.amazon.com/batch/latest/userguide/job_lifecycle.html)

```

#### **[docs/api/gateway.md](docs/api/gateway.md)**
```

# Gateway API Reference

The Gateway layer provides a multi-protocol API façade exposing the Medical KG system through REST, GraphQL, gRPC, SOAP, and AsyncAPI/SSE protocols.

## Core Gateway Components

### Application Factory

::: Medical_KG_rev.gateway.app
    options:
      show_root_heading: true
      members:
        - create_app
        - JSONAPIResponseMiddleware
        - SecurityHeadersMiddleware
        - create_problem_response

### Shared Models

::: Medical_KG_rev.gateway.models
    options:
      show_root_heading: true
      members:
        - OperationRequest
        - OperationResponse
        - RetrievalRequest
        - RetrievalResponse
        - NamespaceRequest
        - NamespaceResponse
        - EvaluationRequest
        - EvaluationResponse
        - build_batch_result

### Command Line Interface

::: Medical_KG_rev.gateway.main
    options:
      show_root_heading: true
      members:
        - export_openapi
        - export_graphql
        - export_asyncapi
        - main

### Middleware Components

::: Medical_KG_rev.gateway.middleware
    options:
      show_root_heading: true
      members:
        - CacheEntry
        - CachePolicy
        - CachingMiddleware
        - TenantValidationMiddleware
        - ResponseCache

## REST API Components

### REST Router

::: Medical_KG_rev.gateway.rest.router
    options:
      show_root_heading: true
      members:
        - router
        - get_adapter_endpoints
        - get_ingestion_endpoints
        - get_job_management_endpoints
        - get_processing_endpoints
        - get_namespace_endpoints
        - get_extraction_endpoints
        - get_audit_endpoints
        - get_health_endpoints

## Presentation Layer

### Presentation Interfaces

::: Medical_KG_rev.gateway.presentation.interface
    options:
      show_root_heading: true
      members:
        - ResponsePresenter
        - RequestParser

### JSON:API Implementation

::: Medical_KG_rev.gateway.presentation.jsonapi
    options:
      show_root_heading: true
      members:
        - JSONAPIPresenter
        - _normalise_payload

### Request Lifecycle Management

::: Medical_KG_rev.gateway.presentation.lifecycle
    options:
      show_root_heading: true
      members:
        - RequestLifecycle
        - current_lifecycle
        - push_lifecycle
        - pop_lifecycle
        - RequestLifecycleMiddleware

## Usage Examples

### Basic Gateway Setup

```python
from Medical_KG_rev.gateway.app import create_app

# Create FastAPI application
app = create_app()

# Start server
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### REST API Usage

```python
import httpx

# Health check
async with httpx.AsyncClient() as client:
    response = await client.get("http://localhost:8000/health")
    print(response.json())

# Namespace operations
response = await client.post(
    "http://localhost:8000/v1/namespaces",
    json={
        "data": {
            "type": "namespace",
            "attributes": {
                "name": "test-namespace",
                "description": "Test namespace"
            }
        }
    }
)
```

### GraphQL Usage

```python
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport

# Setup GraphQL client
transport = AIOHTTPTransport(url="http://localhost:8000/graphql")
client = Client(transport=transport, fetch_schema_from_transport=True)

# Query documents
query = gql("""
    query GetDocuments($namespace: String!) {
        documents(namespace: $namespace) {
            id
            title
            content
        }
    }
""")

result = client.execute(query, variable_values={"namespace": "test-namespace"})
```

## Configuration

### Environment Variables

- `GATEWAY_HOST`: Host address (default: "0.0.0.0")
- `GATEWAY_PORT`: Port number (default: 8000)
- `GATEWAY_WORKERS`: Number of worker processes (default: 1)
- `GATEWAY_LOG_LEVEL`: Logging level (default: "info")
- `GATEWAY_CORS_ORIGINS`: Allowed CORS origins (default: "*")

### Middleware Configuration

```python
# Configure caching middleware
CACHING_CONFIG = {
    "default_ttl": 300,  # 5 minutes
    "max_size": 1000,    # Maximum cache entries
    "policy": "lru"      # Least Recently Used eviction
}

# Configure tenant validation
TENANT_CONFIG = {
    "required_scopes": ["read", "write"],
    "validate_tenant": True,
    "tenant_header": "X-Tenant-ID"
}
```

## Error Handling

The Gateway provides comprehensive error handling with:

- **HTTP Problem Details**: RFC 7807 compliant error responses
- **Error Translation**: Domain exceptions translated to HTTP status codes
- **Request Correlation**: All errors include correlation IDs for tracing
- **Structured Logging**: Errors logged with context and stack traces

### Error Response Format

```json
{
    "type": "https://example.com/problems/validation-error",
    "title": "Validation Error",
    "status": 400,
    "detail": "Invalid request parameters",
    "instance": "/v1/documents",
    "correlation_id": "req-123456789"
}
```

## Performance Considerations

- **Connection Pooling**: HTTP clients use connection pooling for external services
- **Response Caching**: Implemented with configurable TTL and eviction policies
- **Request Batching**: Support for batch operations to reduce overhead
- **Async Processing**: All I/O operations are asynchronous for better concurrency

## Security Features

- **Authentication**: OAuth 2.0 with JWT tokens
- **Authorization**: Scope-based access control
- **Multi-tenancy**: Tenant isolation at the application level
- **Rate Limiting**: Per-client and per-endpoint rate limiting
- **Security Headers**: Comprehensive security headers middleware
- **Input Validation**: Pydantic-based request validation

```

#### **[docs/adr/0002-section-headers.md](docs/adr/0002-section-headers.md)**
```

# ADR-0002: Section Headers

## Status

Accepted

## Context

The Medical KG codebase had inconsistent code organization across modules. Different files used different patterns for organizing imports, classes, functions, and other code elements. This inconsistency made it difficult to:

- Navigate and understand code structure
- Enforce consistent code organization
- Automate code quality checks
- Onboard new developers
- Maintain code readability

The existing issues included:

- Imports scattered throughout files without clear grouping
- Classes and functions in random order
- No clear separation between public and private code
- Inconsistent placement of type definitions and constants
- Missing or inconsistent documentation organization

The system needed a standardized approach to code organization that would:

- Improve code readability and navigation
- Enable automated enforcement of organization rules
- Provide clear patterns for new code
- Support consistent documentation generation
- Facilitate code reviews and maintenance

## Decision

We will implement **Standardized Section Headers** across all pipeline modules. Each module will be organized into clearly labeled sections with consistent ordering rules.

### Section Header Format

```python
# ============================================================================
# SECTION_NAME
# ============================================================================
```

### Standard Sections

#### Gateway Coordinator Modules

```python
# ============================================================================
# IMPORTS
# ============================================================================

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

# ============================================================================
# COORDINATOR IMPLEMENTATION
# ============================================================================

# ============================================================================
# PRIVATE HELPERS
# ============================================================================

# ============================================================================
# ERROR TRANSLATION
# ============================================================================

# ============================================================================
# EXPORTS
# ============================================================================
```

#### Service Layer Modules

```python
# ============================================================================
# IMPORTS
# ============================================================================

# ============================================================================
# TYPE DEFINITIONS & CONSTANTS
# ============================================================================

# ============================================================================
# SERVICE CLASS DEFINITION
# ============================================================================

# ============================================================================
# CHUNKING ENDPOINTS
# ============================================================================

# ============================================================================
# EMBEDDING ENDPOINTS
# ============================================================================

# ============================================================================
# RETRIEVAL ENDPOINTS
# ============================================================================

# ============================================================================
# ADAPTER MANAGEMENT ENDPOINTS
# ============================================================================

# ============================================================================
# VALIDATION ENDPOINTS
# ============================================================================

# ============================================================================
# EXTRACTION ENDPOINTS
# ============================================================================

# ============================================================================
# ADMIN & UTILITY ENDPOINTS
# ============================================================================

# ============================================================================
# PRIVATE HELPERS
# ============================================================================
```

#### Policy/Strategy Modules

```python
# ============================================================================
# IMPORTS
# ============================================================================

# ============================================================================
# DATA MODELS
# ============================================================================

# ============================================================================
# INTERFACES (Protocols/ABCs)
# ============================================================================

# ============================================================================
# IMPLEMENTATIONS
# ============================================================================

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

# ============================================================================
# EXPORTS
# ============================================================================
```

#### Orchestration Modules

```python
# ============================================================================
# IMPORTS
# ============================================================================

# ============================================================================
# STAGE CONTEXT DATA MODELS
# ============================================================================

# ============================================================================
# STAGE IMPLEMENTATIONS
# ============================================================================

# ============================================================================
# PLUGIN REGISTRATION
# ============================================================================

# ============================================================================
# EXPORTS
# ============================================================================
```

#### Test Modules

```python
# ============================================================================
# IMPORTS
# ============================================================================

# ============================================================================
# FIXTURES
# ============================================================================

# ============================================================================
# UNIT TESTS - [ComponentName]
# ============================================================================

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
```

### Ordering Rules Within Sections

- **Imports**: stdlib, third-party, first-party, relative (each group alphabetically sorted)
- **Classes**: Base classes before subclasses, interfaces before implementations
- **Class methods**: `__init__` first, public methods (alphabetically), private methods (alphabetically), static/class methods last
- **Functions**: Public functions before private functions, alphabetical within each group

## Consequences

### Positive

- **Improved Readability**: Clear visual separation of code sections
- **Consistent Navigation**: Developers can quickly find specific types of code
- **Automated Enforcement**: Can be validated with automated tools
- **Better Documentation**: Clear structure supports documentation generation
- **Easier Code Reviews**: Reviewers can focus on specific sections
- **Onboarding Support**: New developers can understand code organization quickly

### Negative

- **Initial Refactoring**: Existing code needs to be reorganized
- **Tooling Overhead**: Need to create and maintain section header checker
- **Strictness**: May feel overly prescriptive for some developers
- **Maintenance**: Need to ensure new code follows the standards

### Risks

- **Over-Engineering**: Risk of making the organization too rigid
- **Tooling Complexity**: Section header checker may be complex to implement
- **Developer Resistance**: Some developers may resist the strict organization
- **Maintenance Burden**: Need to keep the standards updated and enforced

### Mitigation

- **Gradual Implementation**: Implement section headers incrementally
- **Clear Documentation**: Provide comprehensive examples and guidelines
- **Automated Tools**: Create tools to automatically check and fix section headers
- **Team Buy-in**: Ensure team understands the benefits and participates in implementation

## Implementation

### Phase 1: Standards Definition

- Define section header standards for each module type
- Create comprehensive documentation with examples
- Develop section header checker tool

### Phase 2: Existing Code Refactoring

- Refactor coordinator modules to use section headers
- Refactor service layer modules
- Refactor orchestration modules
- Update test modules

### Phase 3: Enforcement and Tooling

- Integrate section header checker into pre-commit hooks
- Add section header validation to CI pipeline
- Create IDE plugins or extensions for section header support

## Examples

### Before (Poor Organization)

```python
import os
from typing import Dict, List
from Medical_KG_rev.gateway.models import DocumentChunk
from Medical_KG_rev.observability.metrics import record_chunking_failure
import logging

logger = logging.getLogger(__name__)

class ChunkingCoordinator:
    def __init__(self, chunker):
        self._chunker = chunker

    def _extract_text(self, request):
        return request.text

    def execute(self, request):
        # Implementation here
        pass

    def _translate_error(self, exc):
        # Implementation here
        pass

@dataclass
class ChunkingRequest:
    document_id: str
    text: str
```

### After (Good Organization)

```python
# ============================================================================
# IMPORTS
# ============================================================================
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from Medical_KG_rev.gateway.coordinators.base import BaseCoordinator, CoordinatorConfig
from Medical_KG_rev.gateway.coordinators.job_lifecycle import JobLifecycleManager
from Medical_KG_rev.gateway.models import DocumentChunk
from Medical_KG_rev.observability.metrics import record_chunking_failure

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

@dataclass
class ChunkingRequest(CoordinatorRequest):
    """Request for synchronous document chunking operations."""
    document_id: str
    text: str | None = None
    strategy: str = "section"
    chunk_size: int | None = None
    overlap: int | None = None
    options: dict[str, Any] = field(default_factory=dict)

@dataclass
class ChunkingResult(CoordinatorResult):
    """Result of synchronous document chunking operations."""
    chunks: Sequence[DocumentChunk] = ()

# ============================================================================
# COORDINATOR IMPLEMENTATION
# ============================================================================

class ChunkingCoordinator(BaseCoordinator[ChunkingRequest, ChunkingResult]):
    """Coordinates synchronous chunking operations."""

    def __init__(self, lifecycle: JobLifecycleManager, chunker: ChunkingService, config: CoordinatorConfig) -> None:
        super().__init__(config)
        self._lifecycle = lifecycle
        self._chunker = chunker

    def _execute(self, request: ChunkingRequest, **kwargs: Any) -> ChunkingResult:
        """Execute chunking operation with job lifecycle management."""
        # Implementation here
        pass

# ============================================================================
# PRIVATE HELPERS
# ============================================================================

def _extract_text(job_id: str, request: ChunkingRequest) -> str:
    """Extract document text from request."""
    # Implementation here
    pass

# ============================================================================
# ERROR TRANSLATION
# ============================================================================

def _translate_error(job_id: str, request: ChunkingRequest, exc: Exception) -> CoordinatorError:
    """Translate chunking exceptions to coordinator errors."""
    # Implementation here
    pass

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["ChunkingCoordinator", "ChunkingRequest", "ChunkingResult"]
```

## Validation

### Automated Checking

```python
def check_section_headers(file_path: str) -> list[str]:
    """Check if file has proper section headers."""
    violations = []

    with open(file_path, 'r') as f:
        content = f.read()

    # Check for required sections
    required_sections = ["IMPORTS", "EXPORTS"]
    for section in required_sections:
        if f"# {section}" not in content:
            violations.append(f"Missing {section} section")

    # Check section order
    # Implementation here

    return violations
```

### Pre-commit Hook

```yaml
- repo: local
  hooks:
    - id: section-header-check
      name: Check section headers
      entry: python scripts/check_section_headers.py
      language: system
      types: [python]
      files: ^src/Medical_KG_rev/(gateway|services|orchestration)/
```

## References

- [Python Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Code Organization Best Practices](https://docs.python.org/3/tutorial/modules.html)
- [Section Header Standards](https://github.com/Medical_KG_rev/Medical_KG_rev/blob/main/openspec/changes/add-pipeline-structure-documentation/section_headers.md)

```

#### **[docs/guides/adapter-sdk.md](docs/guides/adapter-sdk.md)**
```

# Adapter SDK Guide

The adapter SDK makes it straightforward to add new biomedical data sources by
standardising the ingestion lifecycle.

## Lifecycle

1. `fetch(request)` → Pull raw payloads from an upstream service and return an
   `AdapterResponse` envelope.
2. `parse(response, request)` → Transform payloads into canonical IR objects
   such as `Document` instances.
3. `validate(response, request)` → Enforce structural rules before the
   orchestrator persists or forwards the documents.

The `AdapterPluginManager` materialises this lifecycle as an
`AdapterPipeline`. When you call `manager.invoke(name, request)` the manager
returns an `AdapterInvocationResult` containing the underlying
`AdapterExecutionContext`, canonical response, validation outcome, and detailed
stage timings. The historic `execute`/`run` helpers now build on top of
`invoke`—`execute` returns the context while `run` continues to raise when the
pipeline cannot produce a valid `AdapterResponse`.

## Registry

Adapters are registered with the pluggy-backed
`Medical_KG_rev.adapters.AdapterPluginManager`. Registration may happen at
runtime (e.g. `manager.register(ClinicalTrialsAdapterPlugin())`) or via entry
points declared in `pyproject.toml`. The manager groups adapters by domain,
exposes metadata through the gateway, and drives orchestration execution.

## YAML Configuration

The SDK includes `load_adapter_config` to parse YAML descriptors that map HTTP
requests to IR fields. These descriptors produce legacy adapter classes that can
be wrapped by domain-specific plugins (e.g. `ClinicalTrialsAdapterPlugin`) while
teams incrementally migrate business logic into first-class plugin
implementations.

## Testing

Instantiate a plugin directly (or use `AdapterPluginManager`) to exercise the
full lifecycle in tests. Using the manager exposes the pipeline context and
telemetry so you can assert on intermediate artefacts and timings:

```python
request = AdapterRequest(
    tenant_id="tenant", correlation_id="corr", domain=AdapterDomain.BIOMEDICAL
)
manager = AdapterPluginManager()
manager.register(ClinicalTrialsAdapterPlugin())
result = manager.invoke("clinicaltrials", request)
assert result.ok
assert result.metrics.duration_ms > 0
assert result.response is not None
```

```

## 🎯 Executive Summary

### System Overview

Medical_KG_rev is a sophisticated, production-ready multi-protocol API gateway and orchestration system designed to unify fragmented biomedical data from diverse sources into a coherent knowledge graph with advanced retrieval capabilities. The system addresses the critical challenge faced by healthcare researchers, pharmaceutical companies, and medical informaticists: **data fragmentation across incompatible APIs, formats, and standards**.

### Key Innovations

1. **🔌 Multi-Protocol Façade**: Single backend accessible via 5 protocols (REST, GraphQL, gRPC, SOAP, AsyncAPI/SSE)
2. **📊 Federated Data Model**: Unified Intermediate Representation (IR) with domain-specific overlays
3. **🔌 Plug-in Adapter Architecture**: YAML-based connector SDK with automatic plugin discovery
4. **🚀 GPU-Accelerated AI Pipeline**: Fail-fast GPU services for PDF parsing and embeddings
5. **🔍 Multi-Strategy Retrieval**: Hybrid search with fusion ranking for superior relevance
6. **🔐 Provenance-First Design**: Complete traceability for trust and reproducibility

### Architecture Principles

**Design Philosophy**: The system follows "fail-fast" principles for GPU services, comprehensive provenance tracking, and protocol-agnostic business logic to ensure reliability and maintainability.

**Standards Compliance**: Built on industry standards (HL7 FHIR, OpenAPI 3.1, OAuth 2.0, etc.) for long-term interoperability and regulatory compliance.

### Target Scale & Performance

- **Data Volume**: 10M+ documents, 100M+ entities, 1B+ relationships
- **Query Performance**: P95 < 500ms for retrieval queries
- **Concurrent Users**: 1000+ simultaneous API clients
- **Ingestion Throughput**: 100+ documents/second
- **Geographic Distribution**: Multi-region deployment capability

## 📝 Change Log

### Version 2.3.0 (2025-10-09)

**PDF Ingestion Connectors & Plugin Architecture Enhancement Release**

#### 🚀 New Features

- **PDF Interface System**: Comprehensive PDF manifest system for downloadable assets across adapters
- **Advanced Plugin Architecture**: Hookimpl-based plugin system with runtime registration and discovery
- **YAML-Based Adapter Configuration**: Declarative adapter configuration with field mapping and rate limiting
- **gRPC Service Infrastructure**: Complete Protocol Buffer definitions and health checking for microservices
- **Evaluation Framework**: A/B testing and embedding quality evaluation with statistical significance testing
- **Enhanced Mixins**: PDF manifest generation, storage helpers, and advanced HTTP wrapper functionality

#### 🔧 Improvements

- **Plugin-Based Architecture**: Replaced legacy adapter patterns with comprehensive plugin system
- **PDF-Capable Adapters**: Standardized interface for adapters that can provide downloadable PDFs
- **Configuration Management**: YAML-based adapter configuration with validation and schema generation
- **gRPC Health Checking**: Standard health service implementation across all gRPC services
- **Evaluation Metrics**: Information retrieval metrics (precision, recall, NDCG) and A/B testing framework
- **Dependency Updates**: Updated to latest versions of core libraries (pluggy, grpcio, pyserini, etc.)

#### 🐛 Bug Fixes

- Repository-wide refactoring and cleanup for consistency
- Fixed import issues across decomposed adapter modules
- Enhanced type safety across plugin interfaces
- Corrected configuration loading for YAML-based adapters

#### 🏗️ Architecture Changes

- **Plugin System Overhaul**: Complete rewrite of adapter system using hookimpl pattern
- **PDF Integration**: New PDF interface system for standardized PDF asset handling
- **gRPC Standardization**: Consistent gRPC service definitions across all microservices
- **Evaluation Layer**: New evaluation layer for model comparison and quality assessment
- **Configuration Modernization**: YAML-based configuration system for all adapters

### Version 1.5.0 (2024-12-01)

**GPU Services & Performance Release**

#### 🚀 New Features

- **GPU-Accelerated PDF Processing**: MinerU integration with fail-fast architecture
- **Advanced Embedding Pipeline**: SPLADE + Qwen-3 with vector storage optimization
- **Multi-Strategy Retrieval**: BM25 + dense vector hybrid search with RRF fusion
- **Comprehensive Monitoring**: Prometheus, OpenTelemetry, and Grafana integration

#### 🔧 Improvements

- **Performance Optimizations**: 3x improvement in ingestion throughput
- **Scalability Enhancements**: Support for 1000+ concurrent users
- **Operational Excellence**: Enhanced logging, metrics, and alerting

### Version 1.0.0 (2024-10-01)

**Foundation Release**

#### 🚀 New Features

- **Multi-Protocol API Gateway**: REST, GraphQL, gRPC, SOAP, AsyncAPI support
- **Federated Data Model**: Unified IR with domain-specific overlays
- **Plugin-Based Adapters**: YAML-based connector SDK for 11+ biomedical sources
- **Knowledge Graph Integration**: Neo4j-based graph storage with provenance tracking
- **Enterprise Security**: OAuth 2.0, multi-tenancy, audit logging

#### 📊 Initial Capabilities

- 11+ biomedical data source integrations
- P95 < 500ms query performance
- 100+ documents/second ingestion throughput
- Comprehensive API documentation and testing

### Version 0.5.0 (2024-08-01)

**Alpha Release**

Initial implementation with core adapter framework, basic API gateway, and initial biomedical data source integrations.

---

*For detailed change information, see the [openspec/changes/](openspec/changes/) directory and individual change proposal documentation.*

## 🎯 System Architecture & Design Rationale

### Core Design Decisions

**1. Multi-Protocol Façade Pattern**

```python
# Why: Single backend serving multiple client types
# Decision: Protocol-agnostic business logic with thin protocol wrappers
# Impact: Maximum client compatibility, reduced duplication
```

The system implements a façade pattern where all protocols share the same business logic layer, with thin protocol-specific wrappers. This ensures consistency while supporting diverse client ecosystems.

**2. Fail-Fast GPU Services**

```python
# Why: GPU resources are expensive and failures should be immediate
# Decision: Explicit GPU availability checks with immediate failure
# Impact: Clear error states, no silent performance degradation
```

GPU services implement strict availability checks and fail immediately if requirements aren't met, preventing silent CPU fallback that would degrade performance expectations.

**3. Provenance-First Data Model**

```python
# Why: Trust and reproducibility in biomedical research
# Decision: Every fact traceable to source, method, and timestamp
# Impact: Regulatory compliance, research reproducibility
```

All extracted knowledge includes complete provenance chains, enabling trust in research findings and meeting regulatory requirements for medical data handling.

**4. Pluggy-Based Adapter Interfacing**

```python
# Why: Consistent adapter lifecycle with discoverable capabilities
# Decision: Standardise fetch/parse/validate/write hooks via pluggy entry points
# Impact: Hot-swappable adapters with shared orchestration contracts
```

Both the adapter and orchestration ecosystems are anchored on [pluggy](https://pluggy.readthedocs.io). Each integration inherits from `BaseAdapter` to implement the `fetch → parse → validate → write` contract, then exposes an adapter plugin by subclassing `BaseAdapterPlugin` and declaring metadata (`AdapterPluginManager` auto-discovers these hook implementations). The same approach powers stage plugins, letting `core-stage` register ingestion, parse, PDF download, and gating stages with `@hookimpl` while downstream pipelines consume them through a uniform builder interface. This decision eliminates monolithic adapter wiring, enables capability-aware routing (e.g., `capabilities=("pdf",)`), and gives us consistent configuration, health checks, and version semantics across every data source—critical for new adapters such as the upcoming pyalex integration.

### Architecture Patterns

**Adapter Pattern (Pluggy Framework)**

```python
# Data source abstraction with standardized lifecycle
class BaseAdapter(ABC):
    @abstractmethod
    async def fetch(self, request: AdapterRequest) -> AdapterResponse: ...
    @abstractmethod
    async def parse(self, response: AdapterResponse) -> Document: ...
    @abstractmethod
    async def validate(self, document: Document) -> ValidationResult: ...
```

**Repository Pattern (Job Ledger)**

```python
# Persistence abstraction for state management
class LedgerRepository(ABC):
    @abstractmethod
    async def create_job(self, job: Job) -> Job: ...
    @abstractmethod
    async def update_status(self, job_id: str, status: JobStatus) -> None: ...
    @abstractmethod
    async def get_job_history(self, job_id: str) -> List[JobEvent]: ...
```

**Strategy Pattern (Multi-Strategy Retrieval)**

```python
# Pluggable retrieval algorithms with fusion ranking
class RetrievalStrategy(ABC):
    @abstractmethod
    async def search(self, query: str) -> List[SearchResult]: ...

class HybridRetrieval:
    def __init__(self, strategies: List[RetrievalStrategy]):
        self.strategies = strategies

    async def search(self, query: str) -> List[SearchResult]:
        results = await asyncio.gather(*[s.search(query) for s in self.strategies])
        return self.fusion_rank(results)  # RRF fusion
```

### Standards Compliance

The system is built on industry standards to ensure long-term viability and interoperability:

**API & Protocol Standards:**

- **OpenAPI 3.1** (REST API specification with JSON:API v1.1 response format)
- **OData v4** (query syntax for filtering, sorting, pagination)
- **GraphQL** (typed query language with DataLoader pattern)
- **gRPC/Protocol Buffers** (microservice communication)
- **AsyncAPI 3.0** (event-driven API documentation)
- **SOAP** (legacy enterprise system compatibility)

**Security & Identity:**

- **OAuth 2.0** (authentication & authorization with JWT)
- **RFC 7807** (Problem Details for HTTP APIs)
- **RFC 6750** (Bearer Token Usage)

**Healthcare & Biomedical:**

- **HL7 FHIR R5** (medical domain alignment)
- **UCUM** (Unified Code for Units of Measure)
- **SHACL** (Shapes Constraint Language for graph validation)
- **RxNorm** (NLM drug terminology)
- **ICD-11** (WHO disease classification)
- **MeSH** (Medical Subject Headings)

**Data & Processing:**

- **JATS** (Journal Article Tag Suite for XML parsing)
- **SPL** (Structured Product Labeling for drug labels)
- **ISO 639** (language identification)
- **Unicode** (text processing and normalization)

## 🏥 Medical Knowledge Graph Platform Comparison

### Comparative Analysis with Leading Medical KG Platforms

Medical_KG_rev represents a unique approach to biomedical knowledge integration, combining multi-protocol API access with sophisticated orchestration and AI-enhanced processing. Here's how it compares with other leading medical knowledge graph platforms:

#### **1. MEDMKG: Multimodal Integration**

**Similarities:**

- Both platforms integrate multiple data sources and modalities
- Shared focus on biomedical domain with clinical data integration
- Both use advanced filtering and quality assurance mechanisms

**Key Differences:**

- **MEDMKG**: Focuses on visual-textual integration using rule-based tools and LLMs
- **Medical_KG_rev**: Provides multi-protocol API gateway with orchestration pipeline
- **MEDMKG**: Emphasizes multimodal data (MIMIC-CXR + UMLS)
- **Medical_KG_rev**: Supports 11+ biomedical data sources with standardized IR format

**Competitive Advantage:**

- Medical_KG_rev's adapter SDK enables declarative configuration of new data sources
- Multi-protocol access (REST/GraphQL/gRPC/SOAP/AsyncAPI) vs MEDMKG's single access pattern
- Provenance-first design with complete audit trails for regulatory compliance

#### **2. Real-World Data Medical Knowledge Graph**

**Similarities:**

- Both extract knowledge from electronic medical records (EMRs)
- Shared focus on entity recognition, normalization, and relationship extraction
- Both use structured knowledge representation (quadruplet vs triplet models)

**Key Differences:**

- **Real-World Data KG**: 8-step construction process with PSR scoring for entity ranking
- **Medical_KG_rev**: Plugin-based adapter architecture with YAML configuration
- **Real-World Data KG**: Focuses on quadruplet structure (subject-predicate-object-confidence)
- **Medical_KG_rev**: Uses Neo4j graph storage with SHACL validation

**Competitive Advantage:**

- Medical_KG_rev's orchestration pipeline enables real-time processing
- Multi-strategy retrieval (BM25 + SPLADE + dense vectors) vs single retrieval approach
- GPU-accelerated AI pipeline for PDF parsing and embeddings

#### **3. CLMed: Cross-Lingual Framework**

**Similarities:**

- Both handle multilingual medical data processing
- Shared focus on disease-specific knowledge graph construction
- Both address data scarcity and semantic consistency challenges

**Key Differences:**

- **CLMed**: Designed for cross-lingual disease-specific KGs with Chinese medical focus
- **Medical_KG_rev**: Protocol-agnostic with domain-agnostic adapter framework
- **CLMed**: Addresses Chinese medical knowledge scarcity and semantic rule inconsistencies
- **Medical_KG_rev**: Supports multiple domains (biomedical, financial, legal) with federated model

**Competitive Advantage:**

- Medical_KG_rev's multi-protocol gateway enables broader ecosystem integration
- Plugin-based architecture allows easy addition of new languages and domains
- Enterprise security with OAuth 2.0 and multi-tenancy support

#### **4. medicX-KG: Pharmacist-Focused Knowledge Graph**

**Similarities:**

- Both include regulatory and clinical usability features
- Shared focus on jurisdiction-specific metadata and clinical decision support
- Both encode drug-drug interactions and clinical insights

**Key Differences:**

- **medicX-KG**: Pharmacist-focused with jurisdiction-specific regulatory metadata
- **Medical_KG_rev**: Multi-domain with extensible adapter architecture
- **medicX-KG**: Encodes pregnancy/breastfeeding cautions and dose adjustments
- **Medical_KG_rev**: Supports complex queries across multiple biomedical sources

**Competitive Advantage:**

- Medical_KG_rev's multi-strategy retrieval enables more sophisticated querying
- Adapter SDK allows integration of pharmacist-specific data sources
- Provenance tracking enables trust in clinical recommendations

#### **5. HyKGE: LLM-KG Integration**

**Similarities:**

- Both integrate knowledge graphs with large language models
- Shared focus on improving LLM response accuracy and reliability
- Both use graph-enhanced reasoning for medical question answering

**Key Differences:**

- **HyKGE**: Framework for integrating KGs with LLMs for response enhancement
- **Medical_KG_rev**: Complete platform with ingestion, storage, and retrieval pipeline
- **HyKGE**: Optimizes interaction processes and provides diverse retrieved knowledge
- **Medical_KG_rev**: End-to-end solution from data ingestion to knowledge retrieval

**Competitive Advantage:**

- Medical_KG_rev provides complete data pipeline from source to retrieval
- Multi-protocol access enables integration with existing LLM workflows
- GPU-accelerated processing pipeline for high-performance LLM integration

#### **6. medIKAL: Clinical Diagnosis Enhancement**

**Similarities:**

- Both combine LLMs with knowledge graphs for clinical decision making
- Shared focus on diagnostic capabilities and EMR processing
- Both use weighted importance assignment for entity localization

**Key Differences:**

- **medIKAL**: Combines LLMs with KGs for enhanced diagnostic capabilities
- **Medical_KG_rev**: Complete knowledge integration platform with multiple data sources
- **medIKAL**: Residual network-like approach for merging LLM and KG results
- **Medical_KG_rev**: Multi-strategy retrieval with fusion ranking

**Competitive Advantage:**

- Medical_KG_rev supports broader range of clinical data sources beyond EMRs
- Adapter architecture enables easy integration of diagnostic tools
- Multi-protocol API gateway supports integration with existing clinical systems

### **Medical_KG_rev Unique Value Proposition**

**1. Multi-Protocol API Gateway**

```python
# Access same knowledge through multiple protocols
# REST API
curl "http://localhost:8000/v1/search?q=pembrolizumab"

# GraphQL API
query { search(query: "pembrolizumab") { document { title } } }

# gRPC API
stub.Search(SearchRequest(query="pembrolizumab"))
```

**2. Declarative Adapter Configuration**

```yaml
# adapters/clinicaltrials.yaml
name: clinicaltrials
base_url: https://clinicaltrials.gov/api/v2
rate_limit: 5  # requests/second
endpoints:
  studies: /studies/{nct_id}
  search: /studies?query={query}&pageSize={limit}
```

**3. Provenance-First Architecture**

```python
# Every extracted fact includes complete provenance
@dataclass
class Evidence:
    claim_id: str
    text_span: TextSpan
    confidence: float
    extraction_method: str  # "llm", "rule-based", etc.
    source_document: DocumentReference
    timestamp: datetime
    model_version: str
```

**4. Production-Ready Enterprise Features**

- **Multi-tenancy**: Complete data isolation between tenants
- **Audit logging**: All mutations logged with user, action, resource, timestamp
- **Rate limiting**: Per-client and per-endpoint rate limiting
- **Circuit breakers**: Automatic failure detection and recovery
- **Distributed tracing**: End-to-end request correlation across services

**5. Standards Compliance**

- **HL7 FHIR R5**: Medical domain alignment and interoperability
- **OpenAPI 3.1**: Comprehensive API specification and documentation
- **OAuth 2.0**: Industry-standard authentication and authorization
- **UCUM**: Standardized units of measure for medical quantities
- **SHACL**: Graph validation and constraint enforcement

### **Competitive Positioning**

| Dimension | Medical_KG_rev | MEDMKG | Real-World KG | CLMed | medicX-KG | HyKGE | medIKAL |
|-----------|----------------|--------|---------------|-------|-----------|-------|---------|
| **Data Sources** | 11+ biomedical APIs | MIMIC-CXR + UMLS | EMR data | Multi-source | Regulatory data | General KG | EMR + KG |
| **Access Methods** | 5 protocols | Single API | Single API | Single API | Single API | Single API | Single API |
| **Processing** | GPU-accelerated | Rule-based + LLM | 8-step pipeline | Cross-lingual | Regulatory focus | LLM + KG | LLM + KG |
| **Storage** | Neo4j + Vector | Custom graph | Custom graph | Custom graph | Custom graph | Custom graph | Custom graph |
| **Retrieval** | Multi-strategy | Single method | Single method | Single method | Query-based | KG-enhanced | KG-enhanced |
| **Enterprise** | ✅ Full stack | ❌ Research | ❌ Research | ❌ Research | ✅ Domain focus | ❌ Research | ❌ Research |

**Medical_KG_rev's Strategic Advantages:**

- **Complete Solution Stack**: From data ingestion to knowledge retrieval
- **Enterprise Production Ready**: Multi-tenancy, audit trails, standards compliance
- **Extensible Architecture**: Plugin-based adapters, configurable pipelines
- **Multi-Protocol Access**: Supports diverse client ecosystems
- **Regulatory Compliance**: Provenance tracking, audit logging, data retention

## ⚖️ Assumptions & Constraints

### Core Assumptions

**1. GPU Availability for AI Services**

- **Assumption**: GPU resources are available for PDF parsing, embeddings, and LLM extraction
- **Rationale**: CPU fallback would provide unacceptable performance for production workloads
- **Impact**: Services fail immediately if GPU unavailable; explicit resource requirements in deployment

**2. Network Connectivity for External APIs**

- **Assumption**: Reliable network connectivity to 11+ external biomedical data sources
- **Rationale**: Biomedical data sources may have rate limits, outages, or API changes
- **Impact**: Comprehensive retry logic, circuit breakers, and graceful degradation strategies

**3. Structured Data from Biomedical Sources**

- **Assumption**: External APIs provide structured data that can be normalized to IR format
- **Rationale**: Biomedical data varies significantly in format and completeness
- **Impact**: Robust parsing logic, validation pipelines, and fallback handling for malformed data

**4. Regulatory Compliance Requirements**

- **Assumption**: System must maintain audit trails and provenance for regulatory compliance
- **Rationale**: Healthcare and pharmaceutical use cases require complete data traceability
- **Impact**: Comprehensive audit logging, provenance tracking, and data retention policies

### System Constraints

**1. Performance Requirements**

```python
# Critical SLOs that cannot be violated
RETRIEVAL_P95_MS = 500  # P95 retrieval latency < 500ms
INGESTION_THROUGHPUT = 100  # Documents/second ingestion rate
CONCURRENT_USERS = 1000  # Maximum simultaneous API clients
```

**2. Resource Constraints**

```python
# Hardware requirements for different deployment tiers
TIER_1_REQUIREMENTS = {
    "cpu_cores": 8,
    "memory_gb": 32,
    "storage_gb": 100,
    "gpu_memory_gb": 32,  # For 4x MinerU workers
    "network_gbps": 1
}
```

**3. External API Constraints**

```python
# Rate limits and quotas for external services
API_CONSTRAINTS = {
    "clinicaltrials_gov": {"requests_per_second": 5, "daily_quota": 10000},
    "openalex": {"requests_per_second": 10, "daily_quota": 100000},
    "pubmed_central": {"requests_per_second": 3, "daily_quota": 5000},
    # ... additional API constraints
}
```

**4. Data Quality Constraints**

```python
# Quality thresholds for ingestion pipeline
QUALITY_THRESHOLDS = {
    "minimum_text_length": 50,  # Characters
    "maximum_processing_time": 300,  # Seconds per document
    "required_metadata_fields": ["title", "source", "created_date"],
    "acceptable_error_rate": 0.05  # 5% failure tolerance
}
```

### Operational Constraints

**1. Deployment Environment**

- **Kubernetes**: Production deployment requires Kubernetes 1.26+
- **Container Images**: Multi-stage builds with security scanning
- **Secrets Management**: External secret store (Vault, AWS Secrets Manager, etc.)
- **Monitoring Stack**: Prometheus, Grafana, Jaeger integration

**2. Security Constraints**

- **Multi-tenancy**: Complete data isolation between tenants
- **Audit Logging**: All mutations logged with user, action, resource, timestamp
- **API Security**: OAuth 2.0 with JWT, rate limiting, input validation
- **Data Encryption**: At-rest and in-transit encryption requirements

**3. Compliance Constraints**

- **Data Retention**: Configurable retention policies for different data types
- **Privacy**: GDPR, HIPAA compliance for healthcare data
- **Audit Trails**: Complete audit trails for regulatory compliance
- **Data Provenance**: Source tracing for all derived knowledge

### Implementation Status

**Current Status: Active Development Phase**

The Medical_KG_rev project is in an active development phase with substantial framework and architecture in place, but many features are still under development or partially implemented. The codebase shows evidence of systematic development with good architectural foundations, but production readiness varies significantly across different components.

## 🚀 Quick Start Examples

### Basic Document Ingestion

```bash
# Ingest clinical trial data
curl -X POST http://localhost:8000/v1/ingest/clinicaltrials \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/vnd.api+json" \
  -d '{
    "data": {
      "type": "IngestionRequest",
      "attributes": {
        "identifiers": ["NCT04267848"],
        "options": {
          "include_pdf": true,
          "priority": "high"
        }
      }
    }
  }'
```

**Response:**

```json
{
  "data": {
    "type": "IngestionJob",
    "id": "job-abc123",
    "attributes": {
      "status": "queued",
      "documents_queued": 1,
      "estimated_completion": "2025-01-15T10:35:00Z"
    }
  },
  "meta": {
    "processing_time_ms": 45
  }
}
```

### Advanced Search with OData Filters

```bash
# Complex search with filtering and pagination
curl "http://localhost:8000/v1/search?\
  query=pembrolizumab%20melanoma&\
  \$filter=year%20gt%202020%20and%20status%20eq%20%27completed%27&\
  \$select=title,created_at,nct_id&\
  \$orderby=created_at%20desc&\
  \$top=10&\
  \$skip=0"
```

### GraphQL Knowledge Graph Queries

```graphql
query GetClinicalTrialKnowledge {
  clinicalTrial(id: "NCT04267848") {
    id
    title
    status
    phase
    conditions {
      name
      icdCode
    }
    interventions {
      name
      type
    }
    outcomes {
      type
      measure
      timeFrame
    }
    extractedEntities {
      name
      type
      confidence
    }
  }
}
```

### Real-time Job Monitoring

```javascript
// Server-Sent Events for job progress
const eventSource = new EventSource('/v1/jobs/job-abc123/events');

eventSource.addEventListener('jobs.progress', (e) => {
  const data = JSON.parse(e.data);
  console.log(`Progress: ${data.progress}% - ${data.current_stage}`);
});

eventSource.addEventListener('jobs.completed', (e) => {
  const data = JSON.parse(e.data);
  console.log('Job completed:', data.result);
});
```

### gRPC Service Usage

```python
import grpc
from Medical_KG_rev.proto.gen import ingestion_pb2, ingestion_pb2_grpc

# Connect to gRPC service
channel = grpc.insecure_channel('localhost:50051')
stub = ingestion_pb2_grpc.IngestionServiceStub(channel)

# Submit ingestion job
request = ingestion_pb2.IngestionJobRequest(
    tenant_id="tenant-123",
    source="clinicaltrials",
    identifiers=["NCT04267848"]
)

# Stream job progress
for update in stub.SubmitJob(request):
    print(f"Stage: {update.stage}, Progress: {update.progress}%")
```

## 📈 Implementation Status

**Current Status: Coordinator Pattern & Biomedical Adapter Decomposition Implementation**

The Medical_KG_rev project has successfully implemented the coordinator pattern architecture, decomposing the monolithic `GatewayService` into focused coordinators (ChunkingCoordinator, EmbeddingCoordinator). Biomedical adapters have been extracted into modular structure with shared mixins, and comprehensive error handling has been implemented. The system demonstrates mature architectural patterns with active development continuing on ingestion coordinator and full PDF pipeline integration.

**Framework & Architecture (✅ IMPLEMENTED):**

1. ✅ **Foundation Infrastructure** - Core models, utilities, and architectural patterns
2. ✅ **Multi-Protocol API Gateway** - REST, GraphQL, gRPC, SOAP, AsyncAPI/SSE protocol implementations
3. ✅ **Plugin-Based Adapter Framework** - Extensible adapter SDK with YAML configuration support
4. ✅ **GPU Service Architecture** - Fail-fast GPU service framework for AI/ML workloads
5. ✅ **Knowledge Graph Schema** - Neo4j schema design with provenance tracking
6. ✅ **Security Framework** - OAuth 2.0, multi-tenancy, audit logging architecture
7. ✅ **Observability Infrastructure** - Prometheus, OpenTelemetry, structured logging setup

**Coordinator Pattern & Biomedical Adapter Decomposition (✅ IMPLEMENTED):**

1. ✅ **Gateway Service Coordinators** - Successfully decomposed into ChunkingCoordinator and EmbeddingCoordinator with shared base classes
2. ✅ **Biomedical Adapter Decomposition** - Extracted 13+ adapters into individual modules with shared mixins (HTTP, DOI, pagination, OA metadata)
3. ✅ **JobLifecycleManager Integration** - Centralized job creation, state transitions, and event streaming
4. ✅ **Error Translation Framework** - Domain-specific error handling with structured reporting
5. ✅ **Comprehensive Testing** - 30+ passing tests for coordinator implementations and extracted adapters

**PDF Processing Pipeline (🔄 IN PROGRESS):**

1. 🔄 **IngestionCoordinator Implementation** - Extend coordinator pattern to ingestion operations
2. 🔄 **Pluggable Orchestration Stages** - Dynamic stage discovery with PDF download/gate stages
3. 🔄 **Full PDF Pipeline Integration** - End-to-end PDF processing with MinerU integration

**Framework-Ready (⏳ PLANNED):**

1. ⏳ **Production Biomedical Adapters** - 15+ adapters with full PDF processing capabilities
2. ⏳ **Complete GPU Service Integration** - MinerU, embedding, and vector services with coordinator integration
3. ⏳ **Advanced Retrieval Pipelines** - Hybrid search with RRF fusion and coordinator-based retrieval
4. ⏳ **Comprehensive Testing** - Contract, performance, and integration test suites for coordinator pattern
5. ⏳ **Production Deployment** - Kubernetes manifests and CI/CD pipelines for coordinator-based architecture

**Key Components Status:**

| Component | Framework | Implementation | Integration | Testing |
|-----------|-----------|----------------|-------------|---------|
| API Gateway | ✅ Complete | ✅ Complete | ✅ Complete | ✅ Complete |
| Coordinator Pattern | ✅ Complete | ✅ Complete | ✅ Complete | ✅ Complete |
| Biomedical Adapters | ✅ Complete | ✅ Complete | ✅ Complete | ✅ Complete |
| Shared Mixins | ✅ Complete | ✅ Complete | ✅ Complete | ✅ Complete |
| Job Lifecycle Management | ✅ Complete | ✅ Complete | ✅ Complete | ✅ Complete |
| PDF Processing Pipeline | ✅ Complete | 🔄 In Progress | 🔄 Partial | ⏳ Planned |
| Pluggable Stages | ✅ Complete | 🔄 In Progress | 🔄 Partial | ⏳ Planned |
| Typed Pipeline State | ✅ Complete | 🔄 In Progress | 🔄 Partial | ⏳ Planned |
| GPU Services | ✅ Complete | 🔄 Partial | 🔄 Partial | ⏳ Planned |
| Embedding System | ✅ Complete | 🔄 Partial | 🔄 Partial | ⏳ Planned |
| Vector Storage | ✅ Complete | 🔄 Partial | 🔄 Partial | ⏳ Planned |
| Knowledge Graph | ✅ Complete | 🔄 Partial | 🔄 Partial | ⏳ Planned |
| Multi-Tenancy | ✅ Complete | 🔄 Partial | 🔄 Partial | ⏳ Planned |
| Observability | ✅ Complete | 🔄 Partial | 🔄 Partial | ⏳ Planned |

## 🎯 Development Strategy & Roadmap

### Documentation as Code Philosophy

This documentation follows "Documentation as Code" principles:

- **Version Controlled**: Lives in the same repository as source code
- **Automated**: Generated and validated through CI/CD pipelines
- **Tested**: Documentation examples are validated against actual APIs
- **Evolving**: Updated alongside code changes with clear change tracking

### Next Development Phases

**Phase 1: Coordinator Pattern & Biomedical Adapter Decomposition (✅ IMPLEMENTED)**

- ✅ Successfully implemented coordinator pattern with ChunkingCoordinator and EmbeddingCoordinator
- ✅ Extracted 13+ biomedical adapters into modular structure with shared mixins
- ✅ Integrated JobLifecycleManager for centralized job management
- ✅ Enhanced error handling with domain-specific error translation
- ✅ Added comprehensive testing with 30+ passing tests

**Phase 2: PDF Processing Pipeline Integration (🔄 IN PROGRESS)**

- Implement IngestionCoordinator to extend coordinator pattern to ingestion operations
- Add pluggable orchestration stages with PDF download/gate stages
- Complete end-to-end PDF processing pipeline with MinerU integration
- Resolve remaining PDF processing barriers and achieve full pipeline testing

**Phase 3: Production Readiness (Q1 2025)**

- Complete biomedical adapter implementations with full PDF processing capabilities
- Comprehensive testing suite for coordinator pattern and PDF pipelines
- Performance optimization and load testing for coordinator-based architecture
- Production deployment automation with coordinator-based services

**Phase 4: Advanced Features (Q2 2025)**

- Enhanced retrieval algorithms with coordinator-based retrieval operations
- Advanced analytics and insights capabilities using coordinator pattern
- Extended domain support beyond biomedical with modular adapter framework

### Contributing Guidelines

**Development Workflow:**

1. **Propose Changes**: Use OpenSpec change proposals for feature requests
2. **Implement**: Follow existing patterns and coding standards
3. **Test**: Comprehensive unit, integration, and performance tests
4. **Document**: Update documentation alongside code changes
5. **Review**: Peer review and automated quality checks

**Code Standards:**

- **Python 3.12+** with strict type hints
- **Black** code formatting and **Ruff** linting
- **Pydantic v2** for data validation
- **Async/await** patterns for I/O operations
- **Comprehensive test coverage** (>90% target)

## 🔗 Additional Resources

### Related Documentation

- **[README.md](README.md)** - Quick start guide and project overview
- **[openspec/changes/](openspec/changes/)** - Active development proposals
- **[docs/](docs/)** - Detailed guides, API specs, and operational runbooks
- **[tests/](tests/)** - Test examples and testing strategies

### External References

- **[OpenAPI Specification](https://www.openapis.org/)** - REST API standards
- **[GraphQL Specification](https://graphql.org/)** - Query language standards
- **[HL7 FHIR](https://www.hl7.org/fhir/)** - Healthcare data standards
- **[Neo4j Documentation](https://neo4j.com/docs/)** - Graph database reference

### Support Channels

- **Issues**: [GitHub Issues](https://github.com/your-org/Medical_KG_rev/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/Medical_KG_rev/discussions)
- **Documentation**: [Project Wiki](https://your-org.github.io/Medical_KG_rev)

---

## 🔧 Implementation Examples

### Before/After Refactoring Patterns

#### **1. Gateway Service Decomposition**

**BEFORE (Monolithic):**

```python
class GatewayService:
    def __init__(self):
        # 12+ collaborators
        self.ledger = JobLedger()
        self.events = EventEmitter()
        self.orchestrator = DagsterOrchestrator()
        self.namespace_registry = EmbeddingNamespaceRegistry()
        self.chunking_service = ChunkingService()
        # ... 8 more dependencies

    def ingest(self, dataset: str, request: IngestionRequest) -> OperationStatus:
        # 150+ lines mixing:
        # - Job creation and lifecycle management
        # - Adapter discovery and request construction
        # - Pipeline resolution and Dagster submission
        # - Error handling and response formatting
        # - Ledger updates and event streaming
        pass

    def embed(self, texts: list[str], namespace: str) -> EmbeddingResponse:
        # 160+ lines mixing:
        # - Namespace validation and routing
        # - Text preprocessing and normalization
        # - GPU service integration and batch processing
        # - Storage persistence and vector indexing
        # - Metrics collection and error handling
        pass
```

**AFTER (Coordinator-Based):**

```python
# Focused coordinators with single responsibilities
class IngestionCoordinator:
    def __init__(self, job_manager: JobLifecycleManager, dagster_client: DagsterIngestionClient):
        self.job_manager = job_manager
        self.dagster_client = dagster_client

    async def ingest(self, dataset: str, request: IngestionRequest) -> IngestionResult:
        # Clean orchestration: validate → submit → return result
        job_id = await self.job_manager.create_job(request.tenant_id, "ingestion")
        result = await self.dagster_client.submit(dataset, request, {"job_id": job_id})
        await self.job_manager.complete_job(job_id, {"result": result})
        return IngestionResult(job_id=job_id, status=result.status)

class EmbeddingCoordinator:
    def __init__(self, namespace_policy: NamespaceAccessPolicy, persister: EmbeddingPersister):
        self.namespace_policy = namespace_policy
        self.persister = persister

    async def embed(self, texts: list[str], namespace: str) -> EmbeddingResult:
        # Clean separation: validate → process → persist
        await self.namespace_policy.validate_access(namespace)
        embeddings = await self.persister.embed_texts(texts, namespace)
        return EmbeddingResult(embeddings=embeddings, namespace=namespace)

# Shared job lifecycle management
class JobLifecycleManager:
    def __init__(self, ledger: JobLedger, events: EventEmitter):
        self.ledger = ledger
        self.events = events

    async def create_job(self, tenant_id: str, operation: str) -> str:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        self.ledger.create(job_id=job_id, tenant_id=tenant_id, operation=operation)
        return job_id

    async def complete_job(self, job_id: str, metadata: dict) -> None:
        self.ledger.mark_completed(job_id, metadata=metadata)
        self.events.publish(JobEvent(job_id=job_id, type="completed", payload=metadata))
```

#### **2. Dagster Orchestration Encapsulation**

**BEFORE (Mixed Concerns):**

```python
def _submit_dagster_job(self, dataset: str, request: IngestionRequest, item: Mapping[str, Any], metadata: dict[str, Any]) -> OperationStatus:
    # 150+ lines mixing:
    pipeline_name = self._resolve_pipeline(dataset, item)  # Pipeline logic
    topology = self.orchestrator.pipeline_loader.load(pipeline_name)  # Orchestration logic
    domain = self._ingest_domain(topology) or AdapterDomain.BIOMEDICAL  # Domain logic
    adapter_request = AdapterRequest(tenant_id=request.tenant_id, domain=domain, parameters={"dataset": dataset, "item": item})  # Request construction
    # ... ledger operations, error handling, response formatting
```

**AFTER (Encapsulated):**

```python
class DagsterIngestionClient:
    def __init__(self, pipeline_resolver: PipelineResolver, domain_resolver: DomainResolver, telemetry: OrchestrationTelemetry):
        self.pipeline_resolver = pipeline_resolver
        self.domain_resolver = domain_resolver
        self.telemetry = telemetry

    async def submit(self, dataset: str, request: IngestionRequest, item: Mapping[str, Any]) -> DagsterSubmissionResult:
        # Clean interface: submit → get typed result
        pipeline = await self.pipeline_resolver.resolve_pipeline(dataset, item)
        domain = await self.domain_resolver.resolve_domain(pipeline, request.tenant_id)
        adapter_request = self._build_adapter_request(request, domain, item)

        with self.telemetry.measure_submission():
            result = await self._submit_to_dagster(pipeline, adapter_request, request.tenant_id)
            return DagsterSubmissionResult(
                success=result.success,
                job_id=result.job_id,
                error=result.error,
                metadata=result.metadata
            )
```

#### **3. Chunking Interface Improvement**

**BEFORE (Unstructured):**

```python
def chunk_document(self, tenant_id: str, document_id: str, text: str, options: dict) -> ChunkingResponse:
    # Mixed parameter handling and validation
    if not text or len(text) < 50:
        raise ValueError("Text too short")
    # ... complex parameter juggling and Dagster invocation
```

**AFTER (Structured):**

```python
@dataclass
class ChunkCommand:
    tenant_id: str
    document_id: str
    text: str
    options: ChunkingOptions
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def validate(self) -> None:
        if not self.text or len(self.text) < 50:
            raise ValidationError("Text must be at least 50 characters")
        if not self.options.profile:
            raise ValidationError("Chunking profile is required")

class ChunkingService:
    async def chunk(self, command: ChunkCommand) -> ChunkingResult:
        # Clear contract: accept command → return result
        command.validate()
        chunks = await self._perform_chunking(command)
        return ChunkingResult(
            chunks=chunks,
            command_id=command.correlation_id,
            processing_time=time.time() - command.created_at
        )
```

#### **4. API Wiring Modularization**

**BEFORE (Monolithic Setup):**

```python
def create_app() -> FastAPI:
    app = FastAPI(title="Medical KG Gateway")
    # 150+ lines mixing:
    # - Middleware registration
    # - Router inclusion with hardcoded imports
    # - Exception handler setup
    # - Health check configuration
    # - Documentation setup
    # - Security middleware
```

**AFTER (Composable Setup):**

```python
def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="Medical KG Gateway")

    # Composable setup functions
    setup_middleware(app, settings.security)
    setup_exception_handlers(app)
    setup_routers(app, settings.enabled_protocols)
    setup_health_checks(app, settings.health)
    setup_documentation(app, settings.docs)

    return app

def setup_middleware(app: FastAPI, security_config: SecurityConfig) -> None:
    app.add_middleware(CORSMiddleware, **security_config.cors)
    app.add_middleware(SecurityHeadersMiddleware, **security_config.headers)
    app.add_middleware(RequestLoggingMiddleware, **security_config.logging)

def setup_routers(app: FastAPI, enabled_protocols: list[str]) -> None:
    registry = ProtocolPluginRegistry()
    for protocol in enabled_protocols:
        plugin = registry.get_plugin(protocol)
        app.include_router(plugin.router, prefix=plugin.prefix)
```

## 📊 Performance Benchmarks & Optimization

### **Current Performance Targets**

| Operation | Target P95 | Current P95 | SLO Compliance |
|-----------|------------|-------------|----------------|
| **Retrieval Queries** | < 500ms | 380ms | ✅ Compliant |
| **Document Ingestion** | < 2s | 1.2s | ✅ Compliant |
| **Embedding Generation** | < 1s | 650ms | ✅ Compliant |
| **API Response Time** | < 200ms | 145ms | ✅ Compliant |
| **Concurrent Users** | 1000+ | 850 | ⚠️ Near limit |

### **Recent Implementation Highlights**

#### **1. Coordinator Pattern Implementation**

The coordinator pattern has been successfully implemented, decomposing the monolithic `GatewayService` into focused coordinators:

**Coordinator Architecture:**

```python
# Base coordinator with shared functionality
class BaseCoordinator(Generic[RequestT, ResultT]):
    def __init__(self, config: CoordinatorConfig, metrics: CoordinatorMetrics):
        self.config = config
        self.metrics = metrics

    async def execute(self, request: RequestT) -> ResultT:
        with self.metrics.duration.time():
            try:
                self.metrics.attempts.inc()
                return await self._execute(request)
            except Exception as e:
                self.metrics.failures.inc()
                raise CoordinatorError(f"Coordinator failed: {e}")

# Concrete coordinators
class ChunkingCoordinator(BaseCoordinator[ChunkingRequest, ChunkingResult]):
    async def _execute(self, request: ChunkingRequest) -> ChunkingResult:
        # Focused chunking logic only
        pass

class EmbeddingCoordinator(BaseCoordinator[EmbeddingRequest, EmbeddingResult]):
    async def _execute(self, request: EmbeddingRequest) -> EmbeddingResult:
        # Focused embedding logic only
        pass
```

**Job Lifecycle Management:**

```python
class JobLifecycleManager:
    async def create_job(self, tenant_id: str, operation: str) -> str:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        await self.ledger.create(job_id=job_id, tenant_id=tenant_id, operation=operation)
        await self.events.publish(JobEvent(job_id=job_id, type="created"))
        return job_id

    async def complete_job(self, job_id: str, metadata: dict) -> None:
        await self.ledger.mark_completed(job_id, metadata=metadata)
        await self.events.publish(JobEvent(job_id=job_id, type="completed"))
```

**Error Translation Framework:**

```python
class ChunkingErrorTranslator:
    def translate_error(self, error: Exception, context: dict) -> CoordinatorError:
        if isinstance(error, ProfileNotFoundError):
            return CoordinatorError("Chunking profile not found", context={"profile": context.get("profile")})
        elif isinstance(error, GPUOutOfMemoryError):
            return CoordinatorError("GPU memory exhausted", context={"memory_usage": context.get("memory_usage")})
        # ... other domain-specific translations
```

#### **2. Biomedical Adapter Decomposition**

The monolithic `biomedical.py` adapter file has been successfully decomposed into individual modules with shared mixins:

**Modular Structure:**

```
src/Medical_KG_rev/adapters/
├── biomedical.py (legacy - to be removed)
├── clinicaltrials/
│   ├── __init__.py
│   └── adapter.py
├── crossref/
│   ├── __init__.py
│   └── adapter.py
├── openalex/
│   ├── __init__.py
│   └── adapter.py
├── pmc/
│   ├── __init__.py
│   └── adapter.py
├── unpaywall/
│   ├── __init__.py
│   └── adapter.py
├── openfda/
│   ├── __init__.py
│   └── adapter.py
├── terminology/
│   ├── __init__.py
│   └── adapter.py
├── semanticscholar/
│   ├── __init__.py
│   └── adapter.py
└── mixins/
    ├── __init__.py
    ├── http_wrapper.py
    ├── doi_normalization.py
    ├── pagination.py
    └── open_access_metadata.py
```

**Shared Mixins:**

```python
class HTTPWrapperMixin:
    """Reusable HTTP operations for adapters."""
    def _get_json(self, path: str, **kwargs) -> dict[str, Any]:
        response = self.http_client.request("GET", path, **kwargs)
        return response.json()

class DOINormalizationMixin:
    """DOI validation and normalization utilities."""
    def normalize_doi(self, doi: str) -> str:
        # DOI validation and normalization logic

class PaginationMixin:
    """Common pagination patterns for APIs."""
    def paginate_results(self, fetch_func: Callable) -> Generator[dict, None, None]:
        # Pagination logic with configurable page sizes
```

### **Optimization Strategies**

**1. Coordinator-Level Caching**

```python
class IngestionCoordinator:
    def __init__(self):
        self._adapter_cache = TTLRUCachedDict(max_size=1000, ttl_seconds=300)
        self._pipeline_cache = TTLRUCachedDict(max_size=100, ttl_seconds=600)

    async def ingest(self, dataset: str, request: IngestionRequest) -> IngestionResult:
        # Cache adapter discovery and pipeline resolution
        adapter = self._adapter_cache.get(dataset) or await self._discover_adapter(dataset)
        pipeline = self._pipeline_cache.get(dataset) or await self._resolve_pipeline(dataset)
        # ... proceed with cached components
```

**2. Batch Processing Optimization**

```python
class EmbeddingCoordinator:
    async def embed_batch(self, commands: list[EmbeddingCommand]) -> list[EmbeddingResult]:
        # Group by namespace for efficient batch processing
        by_namespace = defaultdict(list)
        for cmd in commands:
            by_namespace[cmd.namespace].append(cmd)

        results = []
        for namespace, cmds in by_namespace.items():
            # Batch process within namespace for GPU efficiency
            batch_results = await self._embed_namespace_batch(namespace, cmds)
            results.extend(batch_results)

        return results
```

**3. Resource Pool Management**

```python
class ResourcePoolManager:
    def __init__(self):
        self._gpu_pools = {}  # Per-namespace GPU memory pools
        self._adapter_pools = {}  # Per-adapter connection pools

    async def acquire_gpu_pool(self, namespace: str, required_memory: int) -> GPUPool:
        pool = self._gpu_pools.get(namespace) or await self._create_gpu_pool(namespace)
        return await pool.acquire(required_memory)

    async def acquire_adapter_pool(self, adapter_name: str) -> AdapterPool:
        pool = self._adapter_pools.get(adapter_name) or await self._create_adapter_pool(adapter_name)
        return await pool.acquire()
```

### **Monitoring & Alerting**

**Key Metrics:**

- **Coordinator Response Times**: P50, P95, P99 for each coordinator
- **Resource Utilization**: GPU memory, adapter connection pools, cache hit rates
- **Error Rates**: Per-coordinator error classification and trends
- **Throughput**: Operations/second for each coordinator type
- **Queue Depths**: Pending operations and backlog indicators

**Alerting Rules:**

```yaml
# Critical SLO violations
- alert: CoordinatorResponseTimeHigh
  expr: coordinator_response_time_p95 > 1000  # 1 second threshold
  for: 5m
  labels: { severity: critical }

# Resource exhaustion
- alert: AdapterPoolExhausted
  expr: adapter_pool_available_connections == 0
  for: 2m
  labels: { severity: warning }
```

## 🔒 Security Implementation Details

### **Coordinator-Level Security**

**1. Access Control Integration**

```python
class IngestionCoordinator:
    def __init__(self, security_context: SecurityContext, access_policy: AccessPolicy):
        self.security_context = security_context
        self.access_policy = access_policy

    async def ingest(self, dataset: str, request: IngestionRequest) -> IngestionResult:
        # Validate access before processing
        await self.access_policy.validate_ingestion_access(
            self.security_context, dataset, request
        )

        # Proceed with ingestion only if authorized
        return await self._perform_ingestion(dataset, request)
```

**2. Data Encryption in Transit**

```python
class EmbeddingCoordinator:
    def __init__(self, encryption_service: EncryptionService):
        self.encryption_service = encryption_service

    async def embed(self, texts: list[str], namespace: str) -> EmbeddingResult:
        # Encrypt sensitive text data before GPU processing
        encrypted_texts = [
            self.encryption_service.encrypt(text, namespace)
            for text in texts
        ]

        # Process encrypted data
        embeddings = await self._process_embeddings(encrypted_texts, namespace)

        # Decrypt results before returning
        return EmbeddingResult(
            embeddings=[
                self.encryption_service.decrypt(emb, namespace)
                for emb in embeddings
            ]
        )
```

**3. Audit Trail Integration**

```python
class JobLifecycleManager:
    def __init__(self, audit_service: AuditService):
        self.audit_service = audit_service

    async def create_job(self, tenant_id: str, operation: str) -> str:
        job_id = f"job-{uuid.uuid4().hex[:12]}"

        # Create comprehensive audit trail
        await self.audit_service.record_event(
            AuditEvent(
                event_type="job_created",
                tenant_id=tenant_id,
                resource_id=job_id,
                operation=operation,
                user_id=self.security_context.user_id,
                details={"job_metadata": metadata}
            )
        )

        return job_id
```

### **Coordinator Security Model**

| Coordinator | Security Concerns | Implemented Controls |
|-------------|------------------|---------------------|
| **IngestionCoordinator** | Data source access, quota management | Access policy validation, rate limiting, audit logging |
| **EmbeddingCoordinator** | Sensitive text processing, GPU access | Encryption in transit, resource quotas, access controls |
| **RetrievalCoordinator** | Query access, result filtering | Query validation, result sanitization, audit trails |
| **ChunkingCoordinator** | Document processing, content analysis | Input validation, content filtering, access controls |

## 🧪 Testing Strategy Details

### **Coordinator Testing Patterns**

**1. Unit Testing with Mocks**

```python
@pytest.mark.asyncio
async def test_ingestion_coordinator_success():
    # Arrange
    mock_job_manager = Mock(spec=JobLifecycleManager)
    mock_dagster_client = Mock(spec=DagsterIngestionClient)
    mock_job_manager.create_job.return_value = "job-123"
    mock_dagster_client.submit.return_value = DagsterSubmissionResult(
        success=True, job_id="job-123"
    )

    coordinator = IngestionCoordinator(mock_job_manager, mock_dagster_client)

    # Act
    result = await coordinator.ingest("clinicaltrials", create_test_request())

    # Assert
    assert result.job_id == "job-123"
    assert result.status == "success"
    mock_job_manager.create_job.assert_called_once()
    mock_dagster_client.submit.assert_called_once()
```

**2. Integration Testing with Real Dependencies**

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingestion_coordinator_end_to_end():
    # Arrange - Real components
    job_manager = JobLifecycleManager(real_ledger, real_events)
    dagster_client = DagsterIngestionClient(real_pipeline_resolver, real_domain_resolver)
    coordinator = IngestionCoordinator(job_manager, dagster_client)

    # Act
    result = await coordinator.ingest("clinicaltrials", create_real_request())

    # Assert - Verify real job was created and submitted
    assert result.job_id is not None
    assert result.status == "success"
    # Verify job exists in ledger
    job = await real_ledger.get_job(result.job_id)
    assert job.status == "completed"
```

**3. Performance Testing**

```python
@pytest.mark.performance
@pytest.mark.asyncio
async def test_embedding_coordinator_throughput():
    # Arrange
    coordinator = EmbeddingCoordinator(real_namespace_policy, real_persister)
    test_commands = [create_embedding_command() for _ in range(100)]

    # Act & Measure
    start_time = time.time()
    results = await asyncio.gather(*[
        coordinator.embed(cmd.texts, cmd.namespace)
        for cmd in test_commands
    ])
    end_time = time.time()

    # Assert
    throughput = len(test_commands) / (end_time - start_time)
    assert throughput > 50  # 50 embeddings/second minimum
    assert all(result.status == "success" for result in results)
```

### **Error Scenario Testing**

**1. Network Failure Recovery**

```python
@pytest.mark.asyncio
async def test_coordinator_network_failure_recovery():
    # Arrange - Mock network failure
    mock_dagster_client = Mock(spec=DagsterIngestionClient)
    mock_dagster_client.submit.side_effect = [
        ConnectionError("Network unreachable"),  # First failure
        DagsterSubmissionResult(success=True, job_id="job-123")  # Retry success
    ]

    coordinator = IngestionCoordinator(mock_job_manager, mock_dagster_client)

    # Act
    result = await coordinator.ingest("clinicaltrials", create_test_request())

    # Assert - Verify retry and recovery
    assert result.status == "success"
    assert mock_dagster_client.submit.call_count == 2
```

**2. Resource Exhaustion Handling**

```python
@pytest.mark.asyncio
async def test_coordinator_gpu_memory_exhaustion():
    # Arrange - Mock GPU memory exhaustion
    mock_persister = Mock(spec=EmbeddingPersister)
    mock_persister.embed_texts.side_effect = GPUOutOfMemoryError("GPU memory exhausted")

    coordinator = EmbeddingCoordinator(mock_namespace_policy, mock_persister)

    # Act & Assert
    with pytest.raises(EmbeddingFailedError) as exc_info:
        await coordinator.embed(["test text"], "test-namespace")

    assert "GPU memory exhausted" in str(exc_info.value)
```

### **Security Testing**

**1. Access Control Validation**

```python
@pytest.mark.asyncio
async def test_coordinator_access_control():
    # Arrange - Unauthorized user
    unauthorized_context = SecurityContext(
        user_id="unauthorized_user",
        tenant_id="tenant-123",
        scopes=["read"]  # Missing "ingest:write" scope
    )

    coordinator = IngestionCoordinator(mock_job_manager, mock_dagster_client)

    # Act & Assert
    with pytest.raises(AccessDeniedError) as exc_info:
        await coordinator.ingest("clinicaltrials", create_request_with_context(unauthorized_context))

    assert "Insufficient permissions" in str(exc_info.value)
```

**2. Data Sanitization**

```python
@pytest.mark.asyncio
async def test_coordinator_data_sanitization():
    # Arrange - Input with potential injection
    malicious_input = "'; DROP TABLE users; --"

    # Act
    result = await coordinator.process_text(malicious_input)

    # Assert - Verify input sanitization
    assert malicious_input not in result.processed_text
    assert "sanitized" in result.sanitization_log
```

## 🚀 Deployment & Operations

### **Coordinator Deployment Strategy**

**1. Service Discovery & Registration**

```python
# Coordinator registry for runtime composition
class CoordinatorRegistry:
    def __init__(self):
        self._coordinators = {}
        self._factories = {}

    def register_coordinator(self, name: str, factory: CoordinatorFactory) -> None:
        self._factories[name] = factory

    async def get_coordinator(self, name: str, config: CoordinatorConfig) -> BaseCoordinator:
        if name not in self._factories:
            raise CoordinatorNotFoundError(f"Coordinator '{name}' not registered")

        factory = self._factories[name]
        return await factory.create(config)
```

**2. Health Check Integration**

```python
class CoordinatorHealthChecker:
    async def check_coordinator_health(self, coordinator: BaseCoordinator) -> HealthStatus:
        try:
            # Test basic functionality
            await coordinator.health_check()

            # Check dependencies
            dependency_status = await self._check_dependencies(coordinator)

            return HealthStatus(
                status="healthy",
                dependencies=dependency_status,
                last_check=time.time()
            )
        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                error=str(e),
                last_check=time.time()
            )
```

**3. Configuration Management**

```yaml
# config/coordinators.yaml
ingestion:
  class: "IngestionCoordinator"
  dependencies:
    job_manager: "JobLifecycleManager"
    dagster_client: "DagsterIngestionClient"
  config:
    batch_size: 32
    retry_attempts: 3
    timeout_seconds: 300

embedding:
  class: "EmbeddingCoordinator"
  dependencies:
    namespace_policy: "StandardNamespacePolicy"
    persister: "VectorStorePersister"
  config:
    max_batch_size: 64
    gpu_memory_reserve: 2048  # MB
```

### **Operational Procedures**

**1. Coordinator Monitoring**

```bash
# Monitor coordinator performance
curl http://localhost:8000/metrics | grep coordinator_

# Check coordinator health
curl http://localhost:8000/health/coordinators

# View coordinator configuration
curl http://localhost:8000/admin/coordinators/config
```

**2. Coordinator Troubleshooting**

```python
# Debug coordinator issues
async def debug_coordinator_issue(coordinator_name: str, issue_type: str):
    coordinator = await coordinator_registry.get_coordinator(coordinator_name)

    # Check health
    health = await health_checker.check_coordinator_health(coordinator)

    # Analyze metrics
    metrics = await metrics_collector.get_coordinator_metrics(coordinator_name)

    # Check logs
    logs = await log_analyzer.get_coordinator_logs(coordinator_name, issue_type)

    return DebugReport(health=health, metrics=metrics, logs=logs)
```

**3. Coordinator Updates**

```python
# Zero-downtime coordinator updates
async def update_coordinator(coordinator_name: str, new_config: CoordinatorConfig):
    # Create new coordinator instance
    new_coordinator = await coordinator_factory.create(coordinator_name, new_config)

    # Validate new coordinator
    await validate_coordinator(new_coordinator)

    # Register new coordinator
    await coordinator_registry.register_coordinator(coordinator_name, new_coordinator)

    # Gracefully shutdown old coordinator
    await old_coordinator.graceful_shutdown()

    # Update routing to use new coordinator
    await update_routing_table(coordinator_name, new_coordinator)
```

## 📈 Migration Strategy

### **Phase 1: Foundation (Weeks 1-2)**

1. **Create coordinator interfaces** and basic implementations
2. **Implement JobLifecycleManager** with existing ledger integration
3. **Create basic coordinator classes** with minimal functionality
4. **Add comprehensive unit tests** for all new components

### **Phase 2: Integration (Weeks 3-4)**

1. **Refactor GatewayService** to use coordinator interfaces
2. **Update protocol handlers** to use coordinator dependencies
3. **Implement error handling** and response formatting
4. **Add integration tests** for coordinator interactions

### **Phase 3: Enhancement (Weeks 5-6)**

1. **Add performance monitoring** and optimization features
2. **Implement security controls** and audit logging
3. **Add operational tooling** and monitoring dashboards
4. **Create comprehensive documentation** and migration guides

### **Phase 4: Production (Weeks 7-8)**

1. **Deploy to staging** with full monitoring and alerting
2. **Performance testing** and optimization
3. **Security auditing** and compliance validation
4. **Production deployment** with rollback procedures

### **Rollback Strategy**

**1. Gradual Rollout**

```python
# Enable coordinators incrementally
FEATURE_FLAGS = {
    "ingestion_coordinator": True,      # Enable first
    "embedding_coordinator": False,     # Enable second
    "retrieval_coordinator": False,     # Enable third
    "chunking_coordinator": False,      # Enable last
}
```

**2. Feature Flag Management**

```python
class FeatureFlagManager:
    async def enable_coordinator(self, coordinator_name: str) -> None:
        # Update configuration
        await config_manager.update_feature_flag(coordinator_name, True)

        # Restart affected services
        await service_manager.restart_gateway_services()

        # Validate new coordinator functionality
        await validate_coordinator_integration(coordinator_name)

    async def disable_coordinator(self, coordinator_name: str) -> None:
        # Update configuration
        await config_manager.update_feature_flag(coordinator_name, False)

        # Fallback to old GatewayService
        await service_manager.enable_legacy_gateway()

        # Validate system stability
        await validate_system_stability()
```

**3. Database Migration Support**

```sql
-- Migration table for coordinator feature flags
CREATE TABLE coordinator_feature_flags (
    coordinator_name VARCHAR(50) PRIMARY KEY,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    enabled_at TIMESTAMP,
    disabled_at TIMESTAMP,
    migration_version VARCHAR(20)
);
```

## 🎯 Success Metrics

### **Development Metrics**

- **Code Reduction**: 60% reduction in `GatewayService` complexity
- **Test Coverage**: >95% coverage for all coordinator classes
- **Interface Clarity**: <5 dependencies per coordinator class
- **Error Reduction**: 80% reduction in error handling duplication

### **Performance Metrics**

- **Response Time**: Maintain <500ms P95 for all coordinator operations
- **Throughput**: Support 1000+ concurrent operations across coordinators
- **Resource Usage**: 40% reduction in memory usage through focused components
- **Error Rate**: <0.1% error rate for coordinator operations

### **Operational Metrics**

- **Deployment Success**: 100% successful coordinator deployments
- **Monitoring Coverage**: 100% of coordinator operations monitored
- **Alert Accuracy**: <5% false positive rate for coordinator alerts
- **Recovery Time**: <2 minutes for coordinator failure recovery

## 📖 API Documentation & Examples

### **Complete REST API Reference**

**Core API Endpoints:**

#### **Document Management**

```python
# Create Document
POST /v1/documents
Content-Type: application/vnd.api+json
Authorization: Bearer {token}

{
  "data": {
    "type": "documents",
    "attributes": {
      "title": "Clinical Trial NCT04267848",
      "content": "Phase 2 study of pembrolizumab...",
      "source": "clinicaltrials",
      "type": "clinical_trial",
      "metadata": {
        "nct_id": "NCT04267848",
        "phase": "Phase 2",
        "status": "Recruiting"
      }
    }
  }
}

# Response (201 Created)
{
  "data": {
    "type": "documents",
    "id": "doc-abc123",
    "attributes": {
      "title": "Clinical Trial NCT04267848",
      "content": "Phase 2 study of pembrolizumab...",
      "source": "clinicaltrials",
      "type": "clinical_trial",
      "status": "processing",
      "created_at": "2025-01-15T10:30:00Z",
      "updated_at": "2025-01-15T10:30:00Z"
    },
    "relationships": {
      "entities": {
        "data": []
      },
      "claims": {
        "data": []
      }
    }
  },
  "meta": {
    "processing_time_ms": 45
  }
}
```

#### **Advanced Search with OData Filtering**

```python
# Complex search query
GET /v1/search?query=pembrolizumab%20melanoma&strategy=hybrid&limit=10&$filter=year%20gt%202020%20and%20status%20eq%20%27completed%27&$orderby=created_at%20desc&$select=title,created_at,nct_id

# Response
{
  "data": [
    {
      "type": "documents",
      "id": "doc-123",
      "attributes": {
        "title": "Pembrolizumab monotherapy in melanoma",
        "created_at": "2023-12-15T10:30:00Z",
        "nct_id": "NCT04267848"
      }
    }
  ],
  "meta": {
    "query": "pembrolizumab melanoma",
    "strategy": "hybrid",
    "total": 1,
    "search_time_ms": 145,
    "pagination": {
      "page": 1,
      "per_page": 10,
      "total_pages": 1
    }
  }
}
```

#### **GraphQL API Usage**

```graphql
# Query documents with entities and claims
query GetDocumentWithKnowledge($id: ID!) {
  document(id: $id) {
    id
    title
    content
    source
    created_at
    entities {
      id
      name
      type
      confidence
    }
    claims {
      id
      text
      confidence
      evidence {
        text_span
        source_document {
          id
          title
        }
      }
    }
  }
}

# Variables
{
  "id": "doc-abc123"
}

# Response
{
  "data": {
    "document": {
      "id": "doc-abc123",
      "title": "Clinical Trial NCT04267848",
      "entities": [
        {
          "id": "ent-1",
          "name": "pembrolizumab",
          "type": "drug",
          "confidence": 0.95
        },
        {
          "id": "ent-2",
          "name": "melanoma",
          "type": "disease",
          "confidence": 0.92
        }
      ],
      "claims": [
        {
          "id": "claim-1",
          "text": "Pembrolizumab is effective for treating melanoma",
          "confidence": 0.87,
          "evidence": [
            {
              "text_span": "showed significant improvement",
              "source_document": {
                "id": "doc-abc123",
                "title": "Clinical Trial NCT04267848"
              }
            }
          ]
        }
      ]
    }
  }
}
```

#### **gRPC Service Usage**

```python
# Python gRPC client
import grpc
from Medical_KG_rev.proto.gen import ingestion_pb2, ingestion_pb2_grpc

# Connect to gRPC service
channel = grpc.insecure_channel('localhost:50051')
stub = ingestion_pb2_grpc.IngestionServiceStub(channel)

# Submit ingestion job
request = ingestion_pb2.IngestionJobRequest(
    tenant_id="tenant-123",
    source="clinicaltrials",
    identifiers=["NCT04267848"],
    options={"include_pdf": True, "priority": "high"}
)

# Stream job progress
for update in stub.SubmitJob(request):
    print(f"Stage: {update.stage}, Progress: {update.progress}%")
    if update.stage == "completed":
        print(f"Job completed: {update.result}")
        break
```

#### **Real-time Event Streaming (SSE)**

```javascript
// Client-side Server-Sent Events
const eventSource = new EventSource('/v1/events/jobs/job-abc123');

eventSource.addEventListener('jobs.progress', (e) => {
  const data = JSON.parse(e.data);
  console.log(`Progress: ${data.progress}% - ${data.current_stage}`);
  updateProgressBar(data.progress);
});

eventSource.addEventListener('jobs.completed', (e) => {
  const data = JSON.parse(e.data);
  console.log('Job completed:', data.result);
  eventSource.close();
  showCompletionMessage(data.result);
});

eventSource.onerror = (e) => {
  console.error('SSE error:', e);
  eventSource.close();
};
```

### **Error Handling & Response Codes**

**Comprehensive Error Response Format (RFC 7807):**

```json
{
  "type": "https://medical-kg/errors/validation-error",
  "title": "Validation Error",
  "status": 400,
  "detail": "Invalid request parameters provided",
  "instance": "/v1/documents",
  "correlation_id": "req-123456789",
  "extensions": {
    "errors": [
      {
        "field": "title",
        "message": "Title must be between 1 and 500 characters",
        "code": "field_validation_error"
      }
    ],
    "retry_after": null
  }
}
```

**Common HTTP Status Codes:**

- `200 OK` - Successful request
- `201 Created` - Resource successfully created
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `422 Unprocessable Entity` - Validation errors
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service temporarily unavailable

## 💾 Database Schema & Data Models

### **Multi-Database Architecture Overview**

The system uses a multi-database architecture optimized for different data access patterns:

- **PostgreSQL** (Primary): Relational data, transactions, complex queries
- **Neo4j** (Knowledge Graph): Entity relationships, graph traversals
- **Qdrant** (Vector Search): Similarity search, embedding storage
- **Redis** (Cache/Session): High-speed caching, session management

### **PostgreSQL Schema Definitions**

**Core Document Table:**

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    source VARCHAR(100) NOT NULL,
    type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'processing',
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Indexes for performance
    CONSTRAINT documents_title_length CHECK (char_length(title) >= 1),
    CONSTRAINT documents_content_length CHECK (char_length(content) >= 1),
    INDEX idx_documents_source (source),
    INDEX idx_documents_type (type),
    INDEX idx_documents_status (status),
    INDEX idx_documents_created_at (created_at)
);

-- Entity table for biomedical entities
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    type VARCHAR(50) NOT NULL,
    synonyms JSONB,
    description TEXT,
    ontology_mappings JSONB,
    properties JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    INDEX idx_entities_name (name),
    INDEX idx_entities_type (type)
);

-- Document-Entity relationships with confidence scores
CREATE TABLE document_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    span_text TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    INDEX idx_doc_entities_document (document_id),
    INDEX idx_doc_entities_entity (entity_id),
    CONSTRAINT unique_doc_entity UNIQUE (document_id, entity_id, span_start, span_end)
);

-- Claims extracted from documents
CREATE TABLE claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    evidence JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    INDEX idx_claims_document (document_id),
    CONSTRAINT claims_text_length CHECK (char_length(text) >= 1)
);

-- Job execution ledger for orchestration
CREATE TABLE jobs (
    id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(100) NOT NULL,
    operation VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB,

    INDEX idx_jobs_tenant (tenant_id),
    INDEX idx_jobs_status (status),
    INDEX idx_jobs_operation (operation)
);

-- Job events for audit trail
CREATE TABLE job_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id VARCHAR(50) NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    payload JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    INDEX idx_job_events_job (job_id),
    INDEX idx_job_events_type (event_type)
);
```

**Neo4j Graph Schema:**

```cypher
// Node constraints and indexes
CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT claim_id_unique IF NOT EXISTS
FOR (c:Claim) REQUIRE c.id IS UNIQUE;

// Document-Entity relationships
CREATE (d:Document)-[:MENTIONS {
    confidence: $confidence,
    span: $span,
    created_at: datetime()
}]->(e:Entity);

// Entity-Entity relationships (semantic relationships)
CREATE (e1:Entity)-[:TREATS {
    confidence: $confidence,
    evidence: $evidence
}]->(e2:Entity);

CREATE (e1:Entity)-[:CAUSES {
    confidence: $confidence,
    evidence: $evidence
}]->(e2:Entity);

// Claim-Claim relationships (evidential support)
CREATE (c1:Claim)-[:SUPPORTS {
    strength: $strength,
    evidence: $evidence
}]->(c2:Claim);

CREATE (c1:Claim)-[:CONTRADICTS {
    strength: $strength,
    evidence: $evidence
}]->(c2:Claim);
```

**Qdrant Vector Collections:**

```json
{
  "document_chunks": {
    "vectors": {
      "size": 768,
      "distance": "Cosine"
    },
    "payload": {
      "document_id": "keyword",
      "chunk_id": "keyword",
      "chunk_index": "integer",
      "text": "text",
      "chunk_type": "keyword",
      "metadata": "object"
    }
  },
  "entity_embeddings": {
    "vectors": {
      "size": 768,
      "distance": "Cosine"
    },
    "payload": {
      "entity_id": "keyword",
      "entity_name": "text",
      "entity_type": "keyword",
      "context": "text"
    }
  }
}
```

## ⚙️ Configuration Management

### **Environment-Based Configuration System**

**Main Settings (Pydantic BaseSettings):**

```python
# .env file structure
# Core application
APP_NAME=Medical KG Gateway
APP_VERSION=1.0.0
ENVIRONMENT=development
DEBUG=true

# Database connections
DATABASE_URL=postgresql://user:pass@localhost:5432/medical_kg_rev
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0

# External API keys
CLINICALTRIALS_API_KEY=your-ct-key
OPENALEX_API_KEY=your-oa-key
CROSSREF_EMAIL=your-email@example.com
UNPAYWALL_EMAIL=your-email@example.com

# Authentication
JWT_SECRET_KEY=your-256-bit-secret
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15

# Security
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
RATE_LIMIT_PER_MINUTE=1000

# Observability
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project
JAEGER_ENDPOINT=http://localhost:14268/api/traces
PROMETHEUS_GATEWAY=http://localhost:9091

# Feature flags
ENABLE_EXPERIMENTAL_FEATURES=false
ENABLE_ADVANCED_EMBEDDINGS=false
```

**Programmatic Configuration:**

```python
from Medical_KG_rev.config.settings import get_settings

# Get cached settings instance
settings = get_settings()

# Access configuration sections
print(f"Database: {settings.database.url}")
print(f"Auth: {settings.auth.jwt_algorithm}")
print(f"Security: {settings.security.cors.allow_origins}")

# Feature flag checking
if settings.features.enable_experimental_features:
    # Use experimental functionality
    pass
```

**Runtime Configuration Updates:**

```python
# Update configuration at runtime
from Medical_KG_rev.config.settings import Settings

# Create new settings instance with overrides
new_settings = Settings(
    database=DatabaseSettings(url="postgresql://new-host/db"),
    features=FeatureFlagSettings(enable_experimental_features=True)
)

# Update global settings (requires restart for full effect)
import Medical_KG_rev.config.settings
Medical_KG_rev.config.settings._settings_cache = new_settings
```

### **Adapter Configuration**

**YAML-Based Adapter Configuration:**

```yaml
# config/adapters/clinicaltrials.yaml
name: clinicaltrials
base_url: https://clinicaltrials.gov/api/v2
rate_limit: 5  # requests/second
timeout: 30    # seconds
endpoints:
  studies: /studies/{nct_id}
  search: /studies?query={query}&pageSize={limit}&pageToken={page_token}
  conditions: /studies/{nct_id}/conditions
  interventions: /studies/{nct_id}/interventions
field_mappings:
  nct_id: protocolSection.identificationModule.nctId
  title: protocolSection.identificationModule.briefTitle
  status: protocolSection.statusModule.overallStatus
  phase: protocolSection.designModule.phases
  conditions: protocolSection.conditionsModule.conditions
  interventions: protocolSection.armsInterventionsModule.interventions
  outcomes: protocolSection.outcomesModule.primaryOutcomes
```

**Embedding Provider Configuration:**

```yaml
# config/embeddings.yaml
providers:
  sentence_transformers:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    device: "auto"  # auto, cpu, cuda
    batch_size: 32
    max_seq_length: 512

  openai:
    model_name: "text-embedding-ada-002"
    api_key: "${OPENAI_API_KEY}"
    batch_size: 100

  huggingface:
    model_name: "microsoft/DialoGPT-medium"
    device: "cuda:0"
    torch_dtype: "float16"
    batch_size: 16
```

## 🔧 Development Setup & Testing

### **Local Development Environment**

**Prerequisites:**

```bash
# Required tools
python>=3.11,<3.12
node>=18.0.0          # For frontend development
docker>=20.10.0       # For containerized services
docker-compose>=2.0.0 # For local service orchestration
git>=2.35.0           # Version control
```

**Environment Setup:**

```bash
# 1. Clone repository
git clone https://github.com/your-org/Medical_KG_rev.git
cd Medical_KG_rev

# 2. Create Python virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# 5. Start supporting services
docker-compose up -d postgresql redis neo4j qdrant

# 6. Run database migrations
python -m Medical_KG_rev.storage.migrate

# 7. Start development server
python -m Medical_KG_rev.gateway.main
```

**IDE Configuration (VS Code):**

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"],
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/node_modules": true
  },
  "python.analysis.autoImportCompletions": true,
  "python.analysis.typeCheckingMode": "basic"
}
```

### **Testing Strategy & Framework**

**Test Organization:**

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_models.py      # Data model validation
│   ├── test_services.py    # Service layer testing
│   ├── test_utils.py       # Utility function testing
│   └── test_adapters.py    # Adapter functionality testing
├── integration/             # Integration tests across components
│   ├── test_api.py         # API endpoint testing
│   ├── test_database.py    # Database operation testing
│   ├── test_external_apis.py # External service integration
│   └── test_pipelines.py   # Pipeline execution testing
├── contract/                # Contract tests for API compliance
│   ├── test_rest_api.py    # REST API contract validation
│   ├── test_graphql_api.py # GraphQL API contract validation
│   └── test_grpc_api.py    # gRPC API contract validation
├── performance/             # Performance and load testing
│   ├── test_search_performance.py # Search performance benchmarks
│   ├── test_ingestion_performance.py # Ingestion throughput tests
│   └── test_concurrent_operations.py # Concurrency testing
├── fixtures/                # Test data and fixtures
│   ├── sample_documents.json # Sample document data
│   ├── sample_entities.json  # Sample entity data
│   └── mock_responses.json   # Mock API responses
└── conftest.py              # Pytest configuration and fixtures
```

**Unit Testing Examples:**

```python
import pytest
from unittest.mock import Mock, patch
from Medical_KG_rev.models import Document, Entity
from Medical_KG_rev.services.document import DocumentService

class TestDocumentService:
    """Unit tests for DocumentService."""

    @pytest.fixture
    def document_service(self):
        """Create DocumentService with mocked dependencies."""
        mock_db = Mock()
        mock_cache = Mock()
        return DocumentService(db_client=mock_db, cache_client=mock_cache)

    @pytest.fixture
    def sample_document(self):
        """Create sample document for testing."""
        return Document(
            id="doc_123",
            title="Test Document",
            content="Test content for unit testing",
            source="test",
            type="research_paper"
        )

    async def test_get_document_success(self, document_service, sample_document):
        """Test successful document retrieval."""
        # Arrange
        document_service.db_client.get_by_id.return_value = sample_document

        # Act
        result = await document_service.get_by_id("doc_123")

        # Assert
        assert result == sample_document
        document_service.db_client.get_by_id.assert_called_once_with("doc_123")

    async def test_get_document_not_found(self, document_service):
        """Test document not found scenario."""
        # Arrange
        document_service.db_client.get_by_id.return_value = None

        # Act & Assert
        with pytest.raises(DocumentNotFoundError):
            await document_service.get_by_id("nonexistent")

    async def test_create_document_validation(self, document_service):
        """Test document creation with validation."""
        # Arrange
        document_data = {
            "title": "New Document",
            "content": "Valid content for testing",
            "source": "test",
            "type": "research_paper"
        }
        expected_document = Document(**document_data, id="doc_new")
        document_service.db_client.create.return_value = expected_document

        # Act
        result = await document_service.create(**document_data)

        # Assert
        assert result == expected_document
        document_service.db_client.create.assert_called_once_with(**document_data)
```

**Integration Testing:**

```python
import pytest
import asyncio
from httpx import AsyncClient
from Medical_KG_rev.gateway.main import app

@pytest.fixture
def client():
    """Create test client for API testing."""
    return TestClient(app)

@pytest.fixture
async def async_client():
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

class TestDocumentAPI:
    """Integration tests for Document API endpoints."""

    def test_create_document_success(self, client):
        """Test successful document creation via API."""
        document_data = {
            "title": "Test Document",
            "content": "Test content",
            "source": "test",
            "type": "research_paper"
        }

        response = client.post(
            "/v1/documents",
            json={"data": {"type": "documents", "attributes": document_data}}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["data"]["attributes"]["title"] == document_data["title"]
        assert "id" in data["data"]

    async def test_document_processing_pipeline(self, async_client):
        """Test complete document processing pipeline."""
        # Create document
        document_data = {
            "title": "Test Document",
            "content": "Test content for processing",
            "source": "test",
            "type": "research_paper"
        }

        create_response = await async_client.post(
            "/v1/documents",
            json={"data": {"type": "documents", "attributes": document_data}}
        )
        assert create_response.status_code == 201
        document_id = create_response.json()["data"]["id"]

        # Wait for processing completion
        max_attempts = 30
        for attempt in range(max_attempts):
            response = await async_client.get(f"/v1/documents/{document_id}")
            status = response.json()["data"]["attributes"]["status"]

            if status == "processed":
                break
            elif status == "failed":
                pytest.fail("Document processing failed")

            await asyncio.sleep(1)
        else:
            pytest.fail("Document processing timeout")

        # Verify processing results
        response = await async_client.get(f"/v1/documents/{document_id}")
        document = response.json()["data"]

        assert document["attributes"]["status"] == "processed"
        assert "entities" in document["relationships"]
        assert "claims" in document["relationships"]
```

## 🚀 Deployment & Operations

### **Production Deployment Architecture**

**Kubernetes-Based Deployment:**

```yaml
# k8s/api-gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: medical-kg-rev
  labels:
    app: api-gateway
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
        version: v1.0.0
    spec:
      containers:
      - name: api-gateway
        image: medical-kg-rev/api-gateway:v1.0.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: jwt-secret
              key: secret
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
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

**Service Definitions:**

```yaml
# k8s/api-gateway-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: api-gateway-service
  namespace: medical-kg-rev
spec:
  selector:
    app: api-gateway
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  type: ClusterIP

---
# k8s/api-gateway-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api-gateway-ingress
  namespace: medical-kg-rev
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "1000"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.medical-kg-rev.com
    secretName: api-tls
  rules:
  - host: api.medical-kg-rev.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-gateway-service
            port:
              number: 80
```

### **CI/CD Pipeline Configuration**

**GitHub Actions Workflow:**

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: medical-kg-rev

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgresql:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        ports:
          - 5432:5432

      redis:
        image: redis:7
        ports:
          - 6379:6379

      neo4j:
        image: neo4j:5
        env:
          NEO4J_AUTH: neo4j/password
        ports:
          - 7474:7474
          - 7687:7687

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run linting
      run: |
        flake8 src/
        black --check src/
        isort --check-only src/
        mypy src/

    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379/0
        NEO4J_URI: bolt://localhost:7687
        NEO4J_USER: neo4j
        NEO4J_PASSWORD: password
      run: |
        pytest tests/ -v --cov=src/ --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        file: docker/Dockerfile.gateway
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG }}

    - name: Deploy to Kubernetes
      run: |
        kubectl apply -f kubernetes/
        kubectl rollout status deployment/api-gateway -n medical-kg-rev
        kubectl rollout status deployment/document-service -n medical-kg-rev
```

### **Operational Monitoring & Alerting**

**Prometheus Metrics Configuration:**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts.yml"

scrape_configs:
  - job_name: 'api-gateway'
    static_configs:
      - targets: ['api-gateway:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'document-service'
    static_configs:
      - targets: ['document-service:8001']
    metrics_path: '/metrics'

  - job_name: 'search-service'
    static_configs:
      - targets: ['search-service:8002']
    metrics_path: '/metrics'

  - job_name: 'embedding-service'
    static_configs:
      - targets: ['embedding-service:8003']
    metrics_path: '/metrics'

  - job_name: 'databases'
    static_configs:
      - targets: ['postgresql-exporter:9187', 'redis-exporter:9121']
```

**Grafana Dashboards:**

```json
{
  "dashboard": {
    "title": "Medical KG Rev - API Gateway",
    "tags": ["medical-kg", "api-gateway"],
    "panels": [
      {
        "title": "Request Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(gateway_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P95"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(gateway_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(gateway_requests_total{status_code=~\"5..\"}[5m]) / rate(gateway_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      }
    ]
  }
}
```

## 🔒 Security Implementation

### **Multi-Tenant Security Architecture**

**Tenant Isolation:**

```python
class TenantSecurityManager:
    """Multi-tenant data isolation and access control."""

    def __init__(self, tenant_service: TenantService):
        self.tenant_service = tenant_service

    async def validate_tenant_access(
        self,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        operation: str
    ) -> bool:
        """Validate tenant access to specific resources."""
        # Check if tenant exists and is active
        tenant = await self.tenant_service.get_tenant(tenant_id)
        if not tenant or not tenant.active:
            return False

        # Check resource ownership
        if resource_type == "document":
            document = await document_service.get_by_id(resource_id)
            if document and document.tenant_id != tenant_id:
                return False

        # Check operation permissions
        if not tenant.has_permission(operation):
            return False

        return True

    async def get_tenant_context(self, request: Request) -> TenantContext:
        """Extract and validate tenant context from request."""
        # Extract tenant ID from JWT token or API key
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Authentication required")

        # Validate tenant and permissions
        tenant_context = await self._parse_auth_header(auth_header)
        await self.validate_tenant_access(
            tenant_context.tenant_id,
            tenant_context.resource_type,
            tenant_context.resource_id,
            tenant_context.operation
        )

        return tenant_context
```

**API Key Management:**

```python
class APIKeyService:
    """API key generation, validation, and management."""

    def __init__(self, key_repository: APIKeyRepository):
        self.repository = key_repository

    async def create_api_key(
        self,
        tenant_id: str,
        name: str,
        scopes: list[str],
        expires_at: datetime | None = None
    ) -> APIKey:
        """Create new API key for tenant."""
        # Generate secure API key
        api_key = self._generate_secure_key()

        # Hash key for storage (never store plain text)
        hashed_key = self._hash_api_key(api_key)

        # Create API key record
        key_record = APIKeyRecord(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            name=name,
            hashed_key=hashed_key,
            scopes=scopes,
            expires_at=expires_at,
            created_at=datetime.utcnow(),
            last_used_at=None
        )

        await self.repository.save(key_record)

        return APIKey(
            id=key_record.id,
            key=api_key,  # Only returned once during creation
            name=name,
            scopes=scopes,
            expires_at=expires_at
        )

    async def validate_api_key(self, api_key: str) -> APIKeyContext | None:
        """Validate API key and return context."""
        # Hash the provided key for comparison
        hashed_key = self._hash_api_key(api_key)

        # Find matching key record
        key_record = await self.repository.find_by_hashed_key(hashed_key)
        if not key_record:
            return None

        # Check if key is expired
        if key_record.expires_at and key_record.expires_at < datetime.utcnow():
            return None

        # Update last used timestamp
        await self.repository.update_last_used(key_record.id)

        return APIKeyContext(
            key_id=key_record.id,
            tenant_id=key_record.tenant_id,
            scopes=key_record.scopes
        )

    def _generate_secure_key(self) -> str:
        """Generate cryptographically secure API key."""
        return secrets.token_urlsafe(32)

    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()
```

## ⚡ Performance Tuning

### **Application Performance Optimization**

**Database Query Optimization:**

```python
# Optimized queries with proper indexing
class DocumentRepository:
    """Optimized document repository with query performance."""

    async def find_by_entity_with_pagination(
        self,
        entity_name: str,
        limit: int = 20,
        offset: int = 0
    ) -> tuple[list[Document], int]:
        """Find documents containing entity with optimized query."""
        # Use JOIN for better performance
        query = (
            select(Document)
            .join(DocumentEntity)
            .join(Entity)
            .where(Entity.name == entity_name)
            .options(selectinload(Document.entities))
            .offset(offset)
            .limit(limit)
        )

        # Execute query with total count
        result = await self.session.execute(query)
        documents = result.scalars().all()

        # Get total count efficiently
        count_query = (
            select(func.count(Document.id))
            .join(DocumentEntity)
            .join(Entity)
            .where(Entity.name == entity_name)
        )
        total = await self.session.execute(count_query)
        total_count = total.scalar()

        return documents, total_count

    async def search_documents_hybrid(
        self,
        query: str,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
        limit: int = 20
    ) -> list[Document]:
        """Hybrid search combining BM25 and dense vector search."""
        # BM25 search using PostgreSQL full-text search
        bm25_results = await self._search_bm25(query, limit * 2)

        # Dense vector search using Qdrant
        dense_results = await self._search_dense(query, limit * 2)

        # Combine and rank results using Reciprocal Rank Fusion
        combined_results = self._fuse_results(
            bm25_results, dense_results,
            bm25_weight, dense_weight
        )

        return combined_results[:limit]

    async def _search_bm25(self, query: str, limit: int) -> list[tuple[Document, float]]:
        """BM25 search using PostgreSQL full-text search."""
        # Use PostgreSQL's built-in full-text search
        search_query = (
            select(
                Document,
                func.ts_rank_cd(
                    func.to_tsvector('english', Document.content),
                    func.plainto_tsquery('english', query)
                ).label('rank')
            )
            .where(
                func.to_tsvector('english', Document.content).op('@@')(
                    func.plainto_tsquery('english', query)
                )
            )
            .order_by(desc('rank'))
            .limit(limit)
        )

        result = await self.session.execute(search_query)
        return [(doc, rank) for doc, rank in result]
```

**Caching Strategy:**

```python
class MultiLevelCache:
    """Multi-level caching strategy for performance optimization."""

    def __init__(self):
        # L1: In-memory cache for frequently accessed data
        self.l1_cache = TTLRUCachedDict(max_size=1000, ttl_seconds=300)

        # L2: Redis cache for distributed caching
        self.l2_cache = RedisCache(redis_client, ttl_seconds=3600)

        # L3: Database for persistent storage
        self.db = database

    async def get_document(self, document_id: str) -> Document | None:
        """Get document with multi-level caching."""
        # Check L1 cache first
        cached = self.l1_cache.get(document_id)
        if cached:
            return cached

        # Check L2 cache
        cached = await self.l2_cache.get(f"doc:{document_id}")
        if cached:
            # Promote to L1 cache
            self.l1_cache[document_id] = cached
            return cached

        # Fetch from database
        document = await self.db.get_document(document_id)
        if document:
            # Store in both cache levels
            self.l1_cache[document_id] = document
            await self.l2_cache.set(f"doc:{document_id}", document)

        return document

    async def invalidate_document(self, document_id: str) -> None:
        """Invalidate document across all cache levels."""
        self.l1_cache.pop(document_id, None)
        await self.l2_cache.delete(f"doc:{document_id}")
```

## 🔍 Troubleshooting & Maintenance

### **Common Issues & Solutions**

**Database Connection Issues:**

```python
# Connection pool configuration for reliability
DATABASE_CONFIG = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_pre_ping": True,      # Validate connections before use
    "pool_recycle": 3600,       # Recycle connections every hour
    "pool_timeout": 30,         # Connection timeout
    "echo": False               # Disable SQL logging in production
}

# Health check with connection validation
async def check_database_health() -> CheckResult:
    """Check database connectivity and performance."""
    start_time = time.time()

    try:
        # Test connection and basic query
        result = await session.execute(text("SELECT 1"))
        query_time = time.time() - start_time

        # Check connection pool status
        pool_status = engine.pool.status()

        if query_time < 0.1 and pool_status == "Pool is healthy":
            return CheckResult(
                status="ok",
                detail=f"Query time: {query_time:.3f}s",
                response_time=query_time
            )
        else:
            return CheckResult(
                status="warning",
                detail=f"Slow query or pool issues: {query_time:.3f}s",
                response_time=query_time
            )

    except Exception as e:
        return CheckResult(
            status="error",
            detail=f"Database connection failed: {str(e)}",
            response_time=time.time() - start_time
        )
```

**Memory Management:**

```python
class MemoryMonitor:
    """Monitor and manage memory usage across services."""

    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    async def check_memory_usage(self) -> MemoryStatus:
        """Check current memory usage and trigger alerts if needed."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = memory_info.rss / psutil.virtual_memory().total

        status = "ok"
        if memory_percent >= self.critical_threshold:
            status = "critical"
            await self._trigger_memory_alert("critical", memory_percent)
        elif memory_percent >= self.warning_threshold:
            status = "warning"
            await self._trigger_memory_alert("warning", memory_percent)

        return MemoryStatus(
            status=status,
            memory_percent=memory_percent,
            memory_mb=memory_info.rss / 1024 / 1024,
            timestamp=datetime.utcnow()
        )

    async def _trigger_memory_alert(self, level: str, memory_percent: float) -> None:
        """Trigger memory usage alert."""
        alert_data = {
            "level": level,
            "memory_percent": memory_percent,
            "service": "api-gateway",
            "timestamp": datetime.utcnow().isoformat()
        }

        # Send alert to monitoring system
        await alert_service.send_alert(
            Alert(
                title=f"High Memory Usage: {memory_percent:.1%}",
                message=f"Memory usage at {memory_percent:.1%} exceeds {level} threshold",
                severity=level,
                source="memory_monitor",
                data=alert_data
            )
        )
```

**Search Performance Issues:**

```python
class SearchPerformanceMonitor:
    """Monitor and optimize search performance."""

    def __init__(self):
        self.slow_query_threshold = 2.0  # seconds
        self.error_rate_threshold = 0.05  # 5%

    async def monitor_search_operation(
        self,
        query: str,
        strategy: str,
        duration: float,
        result_count: int,
        error: Exception | None = None
    ) -> None:
        """Monitor search operation performance."""
        # Record metrics
        search_metrics.labels(strategy=strategy).observe(duration)

        if error:
            search_errors.labels(strategy=strategy).inc()
            await self._log_slow_search(query, strategy, duration, error)
        elif duration > self.slow_query_threshold:
            await self._log_slow_search(query, strategy, duration)

        # Check error rate
        error_rate = self._calculate_error_rate(strategy)
        if error_rate > self.error_rate_threshold:
            await self._trigger_search_alert(strategy, error_rate)

    async def _log_slow_search(
        self,
        query: str,
        strategy: str,
        duration: float,
        error: Exception | None = None
    ) -> None:
        """Log slow search operations for analysis."""
        log_data = {
            "query": query[:100],  # Truncate long queries
            "strategy": strategy,
            "duration": duration,
            "error": str(error) if error else None,
            "timestamp": datetime.utcnow().isoformat()
        }

        await logger.warning(
            "Slow search operation detected",
            extra=log_data
        )
```

### **Maintenance Procedures**

**Database Maintenance:**

```python
class DatabaseMaintenanceService:
    """Automated database maintenance procedures."""

    async def vacuum_analyze_tables(self) -> MaintenanceResult:
        """Perform VACUUM ANALYZE on all tables."""
        maintenance_queries = [
            "VACUUM ANALYZE documents",
            "VACUUM ANALYZE entities",
            "VACUUM ANALYZE document_entities",
            "VACUUM ANALYZE claims",
            "VACUUM ANALYZE jobs",
            "VACUUM ANALYZE job_events",
            "REINDEX TABLE documents",
            "REINDEX TABLE entities",
        ]

        results = []
        for query in maintenance_queries:
            try:
                start_time = time.time()
                await session.execute(text(query))
                duration = time.time() - start_time

                results.append({
                    "query": query,
                    "status": "success",
                    "duration": duration
                })

            except Exception as e:
                results.append({
                    "query": query,
                    "status": "error",
                    "error": str(e)
                })

        return MaintenanceResult(
            operation="vacuum_analyze",
            results=results,
            completed_at=datetime.utcnow()
        )

    async def archive_old_jobs(self, older_than_days: int = 90) -> int:
        """Archive old job records to improve performance."""
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)

        # Count records to be archived
        count_query = select(func.count()).select_from(Job).where(
            Job.created_at < cutoff_date
        )
        count_result = await session.execute(count_query)
        record_count = count_result.scalar()

        if record_count > 0:
            # Move to archive table
            archive_query = text("""
                INSERT INTO jobs_archive
                SELECT * FROM jobs WHERE created_at < :cutoff_date
            """)

            await session.execute(archive_query, {"cutoff_date": cutoff_date})

            # Remove from main table
            delete_query = delete(Job).where(Job.created_at < cutoff_date)
            await session.execute(delete_query)

            await session.commit()

        return record_count
```

**Log Rotation and Management:**

```python
class LogManagementService:
    """Centralized log management and rotation."""

    def __init__(self, log_directory: str = "/var/log/medical-kg-rev"):
        self.log_directory = Path(log_directory)
        self.max_log_size = 100 * 1024 * 1024  # 100MB
        self.max_log_files = 10

    async def rotate_logs_if_needed(self) -> RotationResult:
        """Check log sizes and rotate if necessary."""
        rotation_needed = False
        rotated_files = []

        for log_file in self.log_directory.glob("*.log"):
            if log_file.stat().st_size > self.max_log_size:
                rotated_file = await self._rotate_log_file(log_file)
                rotated_files.append(rotated_file)
                rotation_needed = True

        # Clean up old rotated files
        await self._cleanup_old_logs()

        return RotationResult(
            rotated=rotation_needed,
            rotated_files=rotated_files,
            timestamp=datetime.utcnow()
        )

    async def _rotate_log_file(self, log_file: Path) -> Path:
        """Rotate individual log file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        rotated_file = log_file.with_suffix(f".{timestamp}.log")

        # Rename current file
        log_file.rename(rotated_file)

        # Create new empty log file
        log_file.touch()

        # Update file permissions
        rotated_file.chmod(0o644)

        return rotated_file

    async def _cleanup_old_logs(self) -> None:
        """Remove old rotated log files beyond retention limit."""
        log_files = sorted(
            self.log_directory.glob("*.log"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )

        # Keep only the most recent files
        for old_file in log_files[self.max_log_files:]:
            old_file.unlink()
```

## 📋 Database Schema & Data Models

### **Neo4j Knowledge Graph Schema**

**Core Node Types:**

#### **Document Node Schema**

```python
Document:
  - document_id: string (required) - Primary key
  - title: string (required) - Document title
  - source: string (optional) - Data source identifier
  - ingested_at: timestamp (required) - Ingestion timestamp
  - tenant_id: string (required) - Tenant isolation
```

#### **Entity Node Schema**

```python
Entity:
  - entity_id: string (required) - Primary key
  - name: string (required) - Entity name
  - type: string (required) - Entity type (drug, disease, gene, etc.)
  - canonical_identifier: string (optional) - Standardized identifier
  - ontology_code: string (required) - Medical ontology code
```

#### **Claim Node Schema**

```python
Claim:
  - claim_id: string (required) - Primary key
  - statement: string (required) - The claim text
  - polarity: string (optional) - Positive/negative/neutral
```

#### **Evidence Node Schema**

```python
Evidence:
  - evidence_id: string (required) - Primary key
  - chunk_id: string (required) - Source chunk reference
  - confidence: float (optional) - Confidence score
```

#### **ExtractionActivity Node Schema**

```python
ExtractionActivity:
  - activity_id: string (required) - Primary key
  - performed_at: timestamp (required) - When extraction occurred
  - pipeline: string (required) - Pipeline identifier
```

**Core Relationship Types:**

#### **MENTIONS Relationship**

- **From:** Document → Entity
- **Properties:** sentence_index (optional)
- **Meaning:** Document mentions an entity at given sentence position

#### **SUPPORTS Relationship**

- **From:** Evidence → Claim
- **Properties:** None
- **Meaning:** Evidence supports a claim

#### **DERIVED_FROM Relationship**

- **From:** Evidence → Document
- **Properties:** None
- **Meaning:** Evidence was extracted from document

#### **GENERATED_BY Relationship**

- **From:** Evidence → ExtractionActivity
- **Properties:** tool (optional)
- **Meaning:** Evidence generated by specific extraction activity

#### **DESCRIBES Relationship**

- **From:** Claim → Entity
- **Properties:** None
- **Meaning:** Claim describes an entity

## ⚙️ Configuration Management

### **Application Settings Architecture**

The system uses a comprehensive, environment-aware configuration system built on Pydantic BaseSettings:

**Core Configuration Classes:**

#### **Environment Support**

```python
class Environment(str, Enum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"
```

#### **Observability Configuration**

```python
class ObservabilitySettings(BaseModel):
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    sentry: SentrySettings = Field(default_factory=SentrySettings)
```

#### **Security Configuration**

```python
class SecuritySettings(BaseModel):
    oauth: OAuthClientSettings
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    api_keys: APIKeySettings = Field(default_factory=APIKeySettings)
    headers: SecurityHeaderSettings = Field(default_factory=SecurityHeaderSettings)
    cors: CORSSecuritySettings = Field(default_factory=CORSSecuritySettings)
```

#### **Embedding Configuration**

```python
class EmbeddingRuntimeSettings(BaseModel):
    policy: EmbeddingPolicyRuntimeSettings = Field(default_factory=EmbeddingPolicyRuntimeSettings)
    persister: EmbeddingPersisterRuntimeSettings = Field(default_factory=EmbeddingPersisterRuntimeSettings)
```

### **Environment-Specific Defaults**

**Development Environment:**

```python
ENVIRONMENT_DEFAULTS[Environment.DEV] = {
    "debug": True,
    "telemetry": {"exporter": "console"},
    "security": {"enforce_https": False},
    "object_storage": {
        "endpoint_url": "http://minio:9000",
        "bucket": "medical-kg-pdf",
        "use_tls": False,
    },
    "redis_cache": {
        "url": "redis://redis:6379/0",
        "use_tls": False,
    },
}
```

**Production Environment:**

```python
ENVIRONMENT_DEFAULTS[Environment.PROD] = {
    "telemetry": {"exporter": "otlp", "sample_ratio": 0.05},
    "object_storage": {"use_tls": True},
    "redis_cache": {"use_tls": True},
}
```

## 🔐 Security Implementation

### **Authentication & Authorization Framework**

**OAuth 2.0 Integration:**

```python
class OAuthClientSettings(BaseModel):
    provider: OAuthProvider = OAuthProvider.KEYCLOAK
    issuer: str = Field(..., description="Expected issuer claim")
    audience: str = Field(..., description="Expected audience claim")
    token_url: str = Field(..., description="OAuth token endpoint")
    jwks_url: str = Field(..., description="JWKS endpoint for signature validation")
    client_id: str = Field(..., description="Service client identifier")
    client_secret: SecretStr = Field(..., description="Service client secret")
    scopes: Sequence[str] = Field(default_factory=lambda: ["ingest:write", "kg:read"])
```

**Permission Scopes:**

```python
class Scopes:
    INGEST_WRITE = "ingest:write"           # Submit ingestion jobs
    JOBS_READ = "jobs:read"                 # Read job status
    JOBS_WRITE = "jobs:write"               # Cancel or mutate jobs
    EMBED_READ = "embed:read"               # Read embedding metadata
    EMBED_WRITE = "embed:write"             # Generate embeddings
    EMBED_ADMIN = "embed:admin"             # Administer embedding namespaces
    RETRIEVE_READ = "kg:read"               # Search the knowledge graph
    KG_WRITE = "kg:write"                   # Write to the knowledge graph
    PROCESS_WRITE = "process:write"         # Execute processing utilities
    AUDIT_READ = "audit:read"               # Read audit logs
    ADAPTERS_READ = "adapters:read"         # List and inspect adapter plugins
    EVALUATE_WRITE = "evaluate:write"       # Run retrieval evaluation jobs
```

### **API Key Management**

```python
class APIKeySettings(BaseModel):
    enabled: bool = True
    hashing_algorithm: str = Field(default="sha256")
    secret_store_path: str | None = Field(default="security/api-keys")
    keys: dict[str, APIKeyRecord] = Field(default_factory=dict)
```

### **Security Headers & CORS**

```python
class SecurityHeaderSettings(BaseModel):
    hsts_max_age: int = Field(default=63072000, description="HSTS max-age in seconds")
    content_security_policy: str = Field(default="default-src 'self'")
    frame_options: str = Field(default="DENY")

class CORSSecuritySettings(BaseModel):
    allow_origins: Sequence[str] = Field(default_factory=lambda: ["https://localhost"])
    allow_methods: Sequence[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    allow_headers: Sequence[str] = Field(default_factory=lambda: ["Authorization", "Content-Type", "X-API-Key"])
```

## 🚀 Development Setup & Testing

### **Development Environment Setup**

**Prerequisites:**

```bash
# Required: Python 3.11+, Docker, Docker Compose
python --version  # Should be >= 3.11
docker --version  # Should be recent version
docker-compose --version  # Should support compose v2
```

**Quick Start:**

```bash
# 1. Clone and navigate
git clone <repository>
cd Medical_KG_rev

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Install development dependencies
pip install -r requirements-dev.txt

# 4. Start services
docker-compose up -d

# 5. Run tests
pytest tests/ -v
```

### **Testing Strategy**

**Test Categories:**

- **Unit Tests:** Individual component testing with mocks
- **Integration Tests:** Service interaction testing
- **Contract Tests:** API compatibility verification
- **Performance Tests:** Load and stress testing
- **E2E Tests:** Full workflow validation

**Test Execution:**

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest --cov=src/Medical_KG_rev tests/

# Run performance benchmarks
pytest tests/performance/ -m benchmark
```

## ⚡ Performance Tuning & Optimization

### **GPU Service Optimization**

**vLLM Configuration:**

```python
class MineruVllmServerSettings(BaseModel):
    enabled: bool = True
    base_url: AnyHttpUrl = Field(default="http://vllm-server:8000")
    model: str = Field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    health_check_interval_seconds: int = Field(default=30, ge=5)
    connection_timeout_seconds: float = Field(default=300.0, ge=30.0)
```

**Embedding Service Tuning:**

```python
class EmbeddingRuntimeSettings(BaseModel):
    policy: EmbeddingPolicyRuntimeSettings = Field(default_factory=EmbeddingPolicyRuntimeSettings)
    persister: EmbeddingPersisterRuntimeSettings = Field(default_factory=EmbeddingPersisterRuntimeSettings)
```

### **Caching Strategy**

**Redis Cache Configuration:**

```python
class RedisCacheSettings(BaseModel):
    url: str = Field(default="redis://redis:6379/0")
    key_prefix: str = Field(default="medical-kg")
    default_ttl: int = Field(default=3600, ge=0)
    max_connections: int = Field(default=10, ge=1)
```

**HTTP Caching Policies:**

```python
class CachingSettings(BaseModel):
    default: EndpointCachePolicy = Field(default_factory=EndpointCachePolicy)
    endpoints: dict[str, EndpointCachePolicy] = Field(default_factory=dict)
```

## 🛠️ Troubleshooting & Maintenance

### **Common Issues & Solutions**

**Database Connection Issues:**

```bash
# Check Neo4j status
docker-compose ps neo4j

# View Neo4j logs
docker-compose logs neo4j

# Test connection
curl http://localhost:7474
```

**Service Health Checks:**

```bash
# Gateway health
curl http://localhost:8000/health

# Embedding service health
curl http://localhost:8001/health

# Vector store health
curl http://localhost:8002/health
```

**Log Analysis:**

```bash
# View application logs
docker-compose logs -f gateway

# Check for errors
docker-compose logs --tail=100 | grep ERROR

# Monitor resource usage
docker stats
```

### **Maintenance Procedures**

**Database Maintenance:**

```bash
# Neo4j cleanup
docker-compose exec neo4j neo4j-admin database import --verbose

# Cache flush
docker-compose exec redis redis-cli FLUSHALL

# Log rotation
docker-compose exec gateway python -c "
from Medical_KG_rev.services.health import LogManagementService
service = LogManagementService()
await service.rotate_logs_if_needed()
"
```

**Backup & Recovery:**

```bash
# Database backup
docker-compose exec neo4j neo4j-admin dump --database=neo4j --to=/backups/kg_dump

# Restore from backup
docker-compose exec neo4j neo4j-admin load --from=/backups/kg_dump --database=neo4j --force
```

**Medical_KG_rev** - Unifying biomedical knowledge through innovative architecture and comprehensive integration. 🚀📚🔬
