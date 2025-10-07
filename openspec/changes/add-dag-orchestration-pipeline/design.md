# Design: DAG-Based Orchestration Pipeline

## Context

The current orchestration system has served well for the initial implementation, but hard-wiring pipeline stages in Python code creates friction as we scale:

1. **Topology changes require code edits**: Adding stages, reordering operations, or conditionally skipping steps requires Python changes and redeployment
2. **PDF two-phase gate is implicit**: The requirement to halt processing after PDF download until MinerU completes is enforced through manual coordination rather than explicit workflow gates
3. **Resilience policies are scattered**: Retry logic, timeouts, and circuit breakers are hard-coded in various service methods rather than centrally configured
4. **Component swapping is difficult**: Changing chunkers, embedders, or retrievers cascades through orchestration code because implementations are tightly coupled
5. **Lineage tracking is ad-hoc**: Understanding which stages touched a document requires parsing logs rather than querying a lineage system

**Stakeholders**: Backend engineers, ML engineers (GPU services), data engineers (ingestion), DevOps (monitoring)

**Constraints**:

- Must preserve local-first, non-streaming architecture (no Airflow scheduler, no distributed coordinator)
- Must maintain GPU fail-fast semantics (no implicit CPU fallback)
- Must support PDF two-phase gate with explicit halt/resume
- Must preserve existing hybrid retrieval (BM25 + SPLADE + dense)
- Must work with existing Kafka topics and Job Ledger

## Goals / Non-Goals

### Goals

1. **Externalize pipeline topology** so stage sequences are defined in declarative YAML files
2. **Type-safe stage contracts** using Python Protocols to allow swapping implementations without changing orchestration
3. **Local-first workflow engine** (Dagster) that provides UI, sensor-based gates, and retry logic without requiring distributed infrastructure
4. **Resilience as configuration** with named policies for retries, backoff, circuit breakers, and rate limits
5. **Portable observability** via CloudEvents + OpenLineage for stage-level tracing
6. **Haystack 2 integration** for text operations (chunking, embedding, indexing) while maintaining vendor flexibility

### Non-Goals

- Not introducing streaming architecture (Kafka remains for work queues, not streaming)
- Not replacing Kafka with Dagster's queue system (Kafka is the source of truth)
- Not migrating to cloud-managed workflow engines (Airflow, Temporal, Prefect)
- Not changing existing IR models, adapter contracts, or storage schemas
- Not adding real-time processing (pipelines remain batch-oriented)

## Decisions

### Decision 1: Dagster vs Alternatives (Airflow, Prefect, Temporal)

**Choice**: **Dagster 1.5.0+** (<https://dagster.io/>)

**Rationale**:

- **Local-first**: Dagster runs as a simple process with optional webserver; no scheduler required for our use case
- **Sensor-based gates**: Native support for polling external state (Job Ledger) to resume workflows
- **Typed ops**: Python-first API with strong typing matches our existing codebase
- **Asset materialization**: Tracks which data products (chunks, embeddings, graph nodes) were created by which runs
- **Minimal infrastructure**: PostgreSQL for run history (or can use existing Neo4j), MinIO for logs
- **Developer ergonomics**: Local UI at `localhost:3000` for debugging, no complex DAG syntax (unlike Airflow)

**Alternatives Considered**:

- **Airflow**: Requires scheduler, too heavyweight for local dev, complex DAG syntax, poor typing
- **Prefect**: Cloud-first design, less mature sensor support, requires Prefect Cloud or self-hosted server
- **Temporal**: Geared toward long-running workflows, requires separate server cluster, over-engineered for batch pipelines

**Implementation Example**:

```python
# src/Medical_KG_rev/orchestration/dagster/jobs.py
from dagster import job, op, sensor, RunRequest, SkipReason

@op
def ingest_op(context, adapter_request: AdapterRequest) -> list[RawPayload]:
    """Fetch data from adapter plugin."""
    stage = context.resources.ingest_stage
    stage_context = StageContext(
        tenant_id=adapter_request.tenant_id,
        doc_id=adapter_request.parameters.get("id"),
        correlation_id=context.run_id,
        metadata={}
    )
    return stage.execute(stage_context, adapter_request)

@op
def parse_op(context, payloads: list[RawPayload]) -> Document:
    """Parse raw payloads into IR Document."""
    stage = context.resources.parse_stage
    stage_context = StageContext(...)
    return stage.execute(stage_context, payloads)

@job
def auto_pipeline_job():
    """Non-PDF sources: ingest → parse → chunk → embed → index → extract → kg."""
    doc = parse_op(ingest_op())
    chunks = chunk_op(doc)
    embeddings = embed_op(chunks)
    index_receipt = index_op(embeddings)
    entities, claims = extract_op(doc)
    kg_receipt = kg_op(entities, claims)

@sensor(job=pdf_post_processing_job)
def pdf_ir_ready_sensor(context):
    """Poll Job Ledger for pdf_ir_ready=true, then start post-PDF stages."""
    ledger = context.resources.job_ledger
    ready_jobs = ledger.query({"pdf_ir_ready": True, "post_pdf_started": False})

    for entry in ready_jobs:
        yield RunRequest(
            run_key=f"pdf-post-{entry.job_id}",
            run_config={"ops": {"chunk_op": {"config": {"job_id": entry.job_id}}}}
        )
```

---

### Decision 2: Haystack 2 vs LlamaIndex vs Custom

**Choice**: **haystack-ai 2.0+** (<https://haystack.deepset.ai/>) for text operations

**Rationale**:

- **Component-based**: Modular design (document stores, retrievers, embedders, routers) matches our ports-and-adapters architecture
- **Vendor agnostic**: OpenSearch and FAISS integrations already built, easy to swap
- **OpenAI-compatible**: Supports vLLM endpoints for Qwen embeddings via `OpenAIDocumentEmbedder`
- **Flexible pipelines**: Can define custom components while reusing common ones (splitters, retrievers)
- **Production-ready**: Used by Deepset in commercial deployments, mature ecosystem

**Alternatives Considered**:

- **LlamaIndex**: More opinionated about RAG workflows, less flexible for our hybrid retrieval
- **Custom**: Full control but reinventing wheels for chunking/embedding/indexing logic

**Implementation Example**:

```python
# src/Medical_KG_rev/services/haystack/chunker.py
from haystack import component, Document as HaystackDoc
from haystack.components.preprocessors import DocumentSplitter
from Medical_KG_rev.orchestration.stages.contracts import ChunkStage, StageContext
from Medical_KG_rev.models.ir import Document, Chunk

@component
class IRToHaystackConverter:
    """Convert IR Document to Haystack Document format."""

    @component.output_types(documents=list[HaystackDoc])
    def run(self, ir_doc: Document) -> dict:
        haystack_docs = [
            HaystackDoc(content=section.text, meta={"section_id": section.id})
            for section in ir_doc.sections
        ]
        return {"documents": haystack_docs}

class HaystackChunker(ChunkStage):
    """Implements ChunkStage using Haystack DocumentSplitter."""

    def __init__(self, split_by: str = "sentence", split_length: int = 5):
        self.converter = IRToHaystackConverter()
        self.splitter = DocumentSplitter(split_by=split_by, split_length=split_length)

    def execute(self, ctx: StageContext, document: Document) -> list[Chunk]:
        # Convert IR → Haystack
        haystack_docs = self.converter.run(ir_doc=document)["documents"]

        # Split with Haystack
        split_result = self.splitter.run(documents=haystack_docs)

        # Convert back to IR Chunks with provenance
        chunks = [
            Chunk(
                chunk_id=f"{document.id}:chunk:{idx}",
                text=doc.content,
                metadata={"tenant_id": ctx.tenant_id, **doc.meta},
                provenance={"source_doc": document.id, "stage": "chunk"}
            )
            for idx, doc in enumerate(split_result["documents"])
        ]
        return chunks
```

---

### Decision 3: Resilience Library Stack

**Choice**:

- **tenacity 8.2.0+** (<https://tenacity.readthedocs.io/>) for retries and backoff
- **pybreaker 1.0.0+** (<https://github.com/danielfm/pybreaker>) for circuit breakers
- **aiolimiter 1.1.0+** (<https://github.com/mjpieters/aiolimiter>) for async rate limiting

**Rationale**:

- **Composable**: Each library handles one concern, can be combined at stage boundary
- **Battle-tested**: tenacity used by OpenStack, pybreaker by major Python projects
- **Async-native**: aiolimiter designed for asyncio, matches our async services
- **Configuration-driven**: Policies can be defined in YAML and applied via decorators

**Alternatives Considered**:

- **resilience4j (Java)**: Not Python, would require JVM
- **Custom retry logic**: Already scattered across codebase, goal is to centralize
- **Hystrix**: Deprecated by Netflix, pybreaker is spiritual successor

**Implementation Example**:

```python
# src/Medical_KG_rev/orchestration/resilience/policies.py
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from pybreaker import CircuitBreaker
from aiolimiter import AsyncLimiter

def create_retry_policy(config: ResiliencePolicyConfig):
    """Factory for tenacity retry decorators from config."""
    return retry(
        stop=stop_after_attempt(config.max_attempts),
        wait=wait_exponential_jitter(
            initial=config.backoff_initial_seconds,
            max=config.backoff_max_seconds,
            jitter=config.backoff_jitter_seconds
        ),
        reraise=True
    )

def create_circuit_breaker(config: CircuitBreakerConfig):
    """Factory for pybreaker circuit breakers from config."""
    return CircuitBreaker(
        fail_max=config.failure_threshold,
        reset_timeout=config.reset_timeout_seconds,
        name=config.name
    )

async def rate_limited_call(limiter: AsyncLimiter, func, *args, **kwargs):
    """Execute function with rate limiting."""
    async with limiter:
        return await func(*args, **kwargs)

# Apply to stage execution
class ResilientStageExecutor:
    def __init__(self, stage: ChunkStage, policy: ResiliencePolicyConfig):
        self.stage = stage
        self.retry_decorator = create_retry_policy(policy)
        self.circuit_breaker = create_circuit_breaker(policy.circuit_breaker)
        self.rate_limiter = AsyncLimiter(policy.rate_limit_per_second, 1.0) if policy.rate_limit_per_second else None

    async def execute(self, ctx: StageContext, *args, **kwargs):
        @self.retry_decorator
        @self.circuit_breaker
        async def _execute():
            if self.rate_limiter:
                return await rate_limited_call(self.rate_limiter, self.stage.execute, ctx, *args, **kwargs)
            return await self.stage.execute(ctx, *args, **kwargs)

        return await _execute()
```

---

### Decision 4: CloudEvents + OpenLineage vs Custom Observability

**Choice**: **CloudEvents 1.0** for event envelope + **OpenLineage** (optional) for lineage

**Rationale**:

- **CloudEvents** is CNCF standard, portable across systems (can send to Kafka, HTTP, gRPC)
- **Structured schema**: Consistent event format (`type`, `source`, `subject`, `data`) simplifies parsing
- **OpenLineage** provides standard for lineage (job runs, datasets, facets) with Marquez UI
- **Optional OpenLineage**: Can emit CloudEvents without OpenLineage, add lineage later if needed
- **Integrates with existing observability**: CloudEvents stream feeds Prometheus via Kafka exporter

**Alternatives Considered**:

- **Custom JSON events**: No standard, requires documentation, harder to integrate with tools
- **OpenTelemetry only**: Spans are good for latency but not for data lineage
- **DataHub**: Heavyweight, requires separate service, over-engineered for our needs

**Implementation Example**:

```python
# src/Medical_KG_rev/orchestration/observability/cloudevents.py
from cloudevents.http import CloudEvent
from cloudevents.kafka import to_binary

def create_stage_started_event(ctx: StageContext, stage_name: str) -> CloudEvent:
    """Create CloudEvent for stage start."""
    return CloudEvent({
        "type": "org.medicalkg.orchestration.stage.started",
        "source": f"orchestration/{stage_name}",
        "subject": ctx.doc_id,
        "datacontenttype": "application/json",
        "data": {
            "job_id": ctx.metadata.get("job_id"),
            "tenant_id": ctx.tenant_id,
            "correlation_id": ctx.correlation_id,
            "stage": stage_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    })

def create_stage_completed_event(ctx: StageContext, stage_name: str, duration_ms: int, output_count: int) -> CloudEvent:
    """Create CloudEvent for stage completion."""
    return CloudEvent({
        "type": "org.medicalkg.orchestration.stage.completed",
        "source": f"orchestration/{stage_name}",
        "subject": ctx.doc_id,
        "data": {
            "job_id": ctx.metadata.get("job_id"),
            "stage": stage_name,
            "duration_ms": duration_ms,
            "output_count": output_count,
            "success": True
        }
    })

# Publish to Kafka
async def publish_cloudevent(kafka: KafkaClient, event: CloudEvent):
    """Publish CloudEvent to orchestration.events.v1 topic."""
    headers, body = to_binary(event)
    await kafka.publish(
        topic="orchestration.events.v1",
        value=body,
        headers=headers.items()
    )
```

---

### Decision 5: Pipeline Topology YAML Structure

**Choice**: Declarative YAML with stages, gates, policies

**Format**:

```yaml
# config/orchestration/pipelines/pdf-two-phase.yaml
version: "1.0"
name: "pdf-two-phase"
description: "Pipeline for PDF-bound sources with MinerU gate"
applicable_sources:
  - pmc-fulltext
  - unpaywall
  - core

stages:
  - name: ingest
    type: IngestStage
    policy: default

  - name: download_pdf
    type: DownloadStage
    policy: polite-api
    depends_on: [ingest]

  # GATE: Wait for MinerU to process PDF and set pdf_ir_ready=true
  - name: pdf_gate
    type: Gate
    condition:
      ledger_field: pdf_ir_ready
      equals: true
    resume_stage: chunk

  - name: chunk
    type: ChunkStage
    policy: gpu-bound
    depends_on: [pdf_gate]

  - name: embed
    type: EmbedStage
    policy: gpu-bound
    depends_on: [chunk]

  - name: index
    type: IndexStage
    policy: default
    depends_on: [embed]

  - name: extract
    type: ExtractStage
    policy: gpu-bound
    depends_on: [index]

  - name: kg
    type: KGStage
    policy: default
    depends_on: [extract]
```

**Rationale**:

- **Human-readable**: Engineers can understand pipeline flow without reading code
- **Versionable**: Topology changes tracked in git, can diff and roll back
- **Composition**: Can define common stage sequences and reuse (e.g., post-PDF stages shared across pipelines)
- **Validation**: Pydantic schema validates topology on load

---

### Decision 6: GPU Fail-Fast Enforcement

**Choice**: Stages with GPU requirements check availability at initialization and fail immediately if unavailable

**Rationale**:

- **Preserves existing semantics**: System already designed for GPU-only operation (MinerU, SPLADE, Qwen)
- **Fail-fast prevents silent degradation**: No implicit CPU fallback that produces low-quality results
- **Clear error messages**: Job marked as `gpu_unavailable` in ledger with actionable error

**Implementation**:

```python
# src/Medical_KG_rev/services/haystack/embedder.py
class HaystackEmbedder(EmbedStage):
    def __init__(self, vllm_endpoint: str, model_name: str):
        # Fail-fast GPU check
        gpu_manager = GpuManager(min_memory_mb=8192)
        if not gpu_manager.is_available():
            raise GpuNotAvailableError(
                f"GPU required for embedding service (Qwen {model_name}). "
                f"No CUDA devices found. This service will not start."
            )

        self.embedder = OpenAIDocumentEmbedder(
            api_base_url=vllm_endpoint,
            model=model_name,
            api_key="EMPTY"  # vLLM doesn't require auth
        )
        logger.info(f"HaystackEmbedder initialized with GPU-backed vLLM endpoint: {vllm_endpoint}")

    def execute(self, ctx: StageContext, chunks: list[Chunk]) -> EmbeddingBatch:
        # Convert to Haystack documents
        haystack_docs = [Document(content=chunk.text, meta={"chunk_id": chunk.chunk_id}) for chunk in chunks]

        # Embed with GPU acceleration
        try:
            result = self.embedder.run(documents=haystack_docs)
        except Exception as e:
            # If GPU OOM, fail loudly
            if "CUDA out of memory" in str(e):
                raise GpuOutOfMemoryError(f"GPU OOM during embedding: {e}") from e
            raise

        return EmbeddingBatch(embeddings=result["embeddings"], metadata={"gpu_used": True})
```

---

## Risks / Trade-offs

### Risk 1: Learning Curve for Dagster

**Risk**: Team unfamiliar with Dagster concepts (ops, jobs, sensors, resources)

**Mitigation**:

- Comprehensive documentation with examples (`docs/guides/dagster-orchestration.md`)
- Video walkthrough of Dagster UI for team onboarding
- Gradual rollout with feature flag (Phase 1: test tenant, Phase 2: 50%, Phase 3: 100%)
- Pair programming sessions for first custom pipeline implementations

### Risk 2: Haystack 2 Maturity

**Risk**: Haystack 2.0 is new (released late 2023), potential for breaking changes

**Mitigation**:

- Stage contracts (Python Protocols) insulate rest of system from Haystack specifics
- Can swap Haystack components for custom implementations without touching orchestration
- Pin Haystack version in requirements.txt, test upgrades in staging
- Monitor Haystack GitHub for breaking change announcements

### Risk 3: Migration Complexity

**Risk**: Two-phase rollout requires maintaining both legacy and Dagster orchestration

**Mitigation**:

- Feature flag (`MK_USE_DAGSTER`) allows instant rollback if issues arise
- Compatibility shim routes `submit_job` calls to appropriate backend
- Comprehensive integration tests for both paths
- Clear deprecation timeline (legacy removed in v0.3.0)

### Risk 4: Dagster Operational Overhead

**Risk**: Dagster adds webserver and daemon processes to monitor

**Mitigation**:

- Local-first design: Dagster runs as simple process, no complex scheduler
- Health check endpoints for both services
- Prometheus metrics for Dagster job/stage execution
- Runbook for Dagster operations (start/stop, scaling, troubleshooting)

---

## Implementation Plan (Hard Cutover)

### Phase 1: Build New Architecture (Week 1-2)

1. **Audit & Delete Plan**: Create `LEGACY_DECOMMISSION_CHECKLIST.md`
2. **Add Dependencies**: dagster, haystack-ai, tenacity, pybreaker, aiolimiter to requirements.txt
3. **Implement Stage Contracts**: Python Protocols for all 8 stage types
4. **Create YAML Topologies**: auto.yaml, pdf-two-phase.yaml with resilience policies
5. **Implement Dagster Jobs**: auto_pipeline_job, pdf_two_phase_job, pdf_ir_ready_sensor
6. **Build Haystack Wrappers**: HaystackChunker, HaystackEmbedder, HaystackIndexWriter
7. **Atomic Deletions**: Delete legacy code in same commits as new implementations
   - Commit 1: Add Dagster jobs + delete `orchestrator.py`, `worker.py`
   - Commit 2: Add HaystackChunker + delete custom chunkers
   - Commit 3: Add resilience decorators + delete custom retry/circuit breaker logic
8. **Test Migration**: Delete legacy tests, create new Dagster/Haystack tests

### Phase 2: Integration Testing (Week 3-4)

1. **End-to-End Tests**: Auto pipeline (ClinicalTrials) and PDF two-phase (PMC)
2. **Performance Benchmarks**: Validate throughput ≥100 docs/sec, P95 latency <500ms
3. **GPU Fail-Fast**: Verify no CPU fallback for GPU-required stages
4. **CloudEvents Validation**: Confirm all stage lifecycle events emitted
5. **Sensor Testing**: Verify pdf_ir_ready_sensor triggers post-PDF stages
6. **Codebase Validation**: Verify ≥30% code reduction, no dead legacy code

### Phase 3: Production Deployment (Week 5-6)

1. **Deploy to Production**: Single feature branch merge, no legacy code remains
2. **Monitor Observability**: CloudEvents stream, Prometheus metrics, Grafana dashboards
3. **Validate Operations**: Dagster UI accessible, sensors running, jobs completing
4. **Emergency Rollback**: Revert entire feature branch if critical issues (no legacy to fall back to)
5. **Retrospective**: Document lessons learned, codebase reduction achieved

---

## Configuration Management Implementation

### Pydantic Models for Pipeline Topology

```python
# src/Medical_KG_rev/orchestration/config/models.py
from enum import Enum
from pydantic import BaseModel, Field, field_validator

class BackoffStrategy(str, Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    NONE = "none"

class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""
    failure_threshold: int = Field(ge=3, le=10)
    reset_timeout_seconds: int = Field(ge=30, le=600)

class ResiliencePolicyConfig(BaseModel):
    """Named resilience policy loaded from YAML."""
    name: str
    max_attempts: int = Field(ge=1, le=10)
    backoff_strategy: BackoffStrategy
    backoff_initial_seconds: float = Field(ge=0.1, le=10.0)
    backoff_max_seconds: float = Field(ge=1.0, le=300.0)
    backoff_jitter_seconds: float = Field(ge=0.0, le=5.0)
    timeout_seconds: int = Field(ge=1, le=600)
    circuit_breaker: CircuitBreakerConfig | None = None
    rate_limit_per_second: float | None = Field(default=None, ge=0.1, le=100.0)

class GateCondition(BaseModel):
    """Gate condition for conditional pipeline execution."""
    ledger_field: str = Field(pattern=r"^[a-z_]+$")
    equals: bool | str | int

class StageDefinition(BaseModel):
    """Single stage in pipeline topology."""
    name: str = Field(pattern=r"^[a-z_]+$")
    type: str  # IngestStage, ParseStage, ChunkStage, etc.
    policy: str  # Reference to resilience policy name
    depends_on: list[str] = Field(default_factory=list)
    condition: GateCondition | None = None

    @field_validator("depends_on")
    @classmethod
    def validate_dependencies(cls, v):
        if len(v) != len(set(v)):
            raise ValueError("Duplicate dependencies not allowed")
        return v

class PipelineTopologyConfig(BaseModel):
    """Complete pipeline topology loaded from YAML."""
    version: str = Field(pattern=r"^\d+\.\d+$")
    name: str = Field(pattern=r"^[a-z-]+$")
    description: str
    applicable_sources: list[str]
    stages: list[StageDefinition]

    @field_validator("stages")
    @classmethod
    def validate_dag(cls, stages):
        """Ensure stages form a DAG (no cycles)."""
        # Build dependency graph
        graph = {s.name: set(s.depends_on) for s in stages}

        # Topological sort to detect cycles
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        for stage in stages:
            if stage.name not in visited:
                if has_cycle(stage.name):
                    raise ValueError(f"Cycle detected in pipeline: {stage.name}")

        return stages

class PipelineConfigLoader:
    """Loads and caches pipeline topology configurations."""

    def __init__(self, config_dir: Path = Path("config/orchestration/pipelines")):
        self.config_dir = config_dir
        self._cache: dict[str, PipelineTopologyConfig] = {}

    def load(self, pipeline_name: str) -> PipelineTopologyConfig:
        """Load pipeline config from YAML with validation."""
        if pipeline_name in self._cache:
            return self._cache[pipeline_name]

        yaml_path = self.config_dir / f"{pipeline_name}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {yaml_path}")

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        config = PipelineTopologyConfig(**data)
        self._cache[pipeline_name] = config
        return config
```

---

## Security & Authentication Details

### Dagster UI OAuth Integration

```python
# src/Medical_KG_rev/orchestration/dagster/auth.py
from dagster import DagsterInstance
from dagster_cloud import DagsterCloudAgentInstance
from Medical_KG_rev.auth.jwt import JWTAuthenticator

class DagsterOAuthConfig:
    """OAuth configuration for Dagster webserver."""

    def __init__(self, settings: AppSettings):
        self.jwt_authenticator = JWTAuthenticator(
            jwks_url=settings.auth.jwks_url,
            audience="medical-kg-api",
            issuer="medical-kg-auth"
        )
        self.required_scopes = ["admin:read", "admin:write"]

    def get_instance_config(self) -> dict:
        """Generate Dagster instance YAML config with OAuth."""
        return {
            "run_launcher": {
                "module": "dagster.core.launcher",
                "class": "DefaultRunLauncher"
            },
            "run_storage": {
                "module": "dagster_postgres.run_storage",
                "class": "PostgresRunStorage",
                "config": {
                    "postgres_url": os.getenv("DAGSTER_POSTGRES_URL")
                }
            },
            "event_log_storage": {
                "module": "dagster_postgres.event_log",
                "class": "PostgresEventLogStorage",
                "config": {
                    "postgres_url": os.getenv("DAGSTER_POSTGRES_URL")
                }
            },
            "schedule_storage": {
                "module": "dagster_postgres.schedule_storage",
                "class": "PostgresScheduleStorage",
                "config": {
                    "postgres_url": os.getenv("DAGSTER_POSTGRES_URL")
                }
            },
            "telemetry": {"enabled": False},
            "auth": {
                "type": "jwt",
                "jwt_secret": os.getenv("MK_JWT_SECRET_KEY"),
                "jwt_algorithm": "HS256",
                "required_claims": {
                    "aud": "medical-kg-api",
                    "scopes": self.required_scopes
                }
            }
        }

# ops/k8s/base/configmap-dagster.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: dagster-instance
  namespace: medical-kg
data:
  dagster.yaml: |
    run_launcher:
      module: dagster.core.launcher
      class: DefaultRunLauncher

    run_storage:
      module: dagster_postgres.run_storage
      class: PostgresRunStorage
      config:
        postgres_url:
          env: DAGSTER_POSTGRES_URL

    auth:
      type: jwt
      jwt_secret:
        env: MK_JWT_SECRET_KEY
      jwt_algorithm: HS256
      required_claims:
        aud: medical-kg-api
        scopes: ["admin:read", "admin:write"]
```

---

## Data Migration Implementation

### Job Ledger Migration Script

```python
# scripts/migrate_job_ledger_for_dagster.py
"""Migrate Job Ledger schema to support Dagster orchestration."""
import logging
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.config import get_settings

logger = logging.getLogger(__name__)

def migrate_ledger_schema():
    """Add new fields to existing Job Ledger entries."""
    settings = get_settings()
    ledger = JobLedger(storage_backend=settings.ledger_storage)

    # New fields with defaults
    new_fields = {
        "pdf_downloaded": False,
        "pdf_ir_ready": False,
        "current_stage": None,
        "pipeline_name": "auto",  # Default to auto for existing jobs
        "retry_count_per_stage": {},
        "post_pdf_started": False
    }

    logger.info("Starting Job Ledger migration for Dagster support")

    # Iterate all existing entries
    migrated_count = 0
    error_count = 0

    for entry in ledger.list_all():
        try:
            # Check if already migrated
            if hasattr(entry, "pipeline_name"):
                logger.debug(f"Job {entry.job_id} already migrated, skipping")
                continue

            # Add new fields
            for field, default_value in new_fields.items():
                setattr(entry, field, default_value)

            # Infer pipeline_name from existing metadata
            if "pdf_url" in entry.metadata or entry.dataset in ["pmc-fulltext", "unpaywall", "core"]:
                entry.pipeline_name = "pdf-two-phase"

            # Update storage
            ledger.update(entry)
            migrated_count += 1

            if migrated_count % 100 == 0:
                logger.info(f"Migrated {migrated_count} entries...")

        except Exception as e:
            logger.error(f"Failed to migrate job {entry.job_id}: {e}")
            error_count += 1

    logger.info(f"Migration complete: {migrated_count} migrated, {error_count} errors")

    if error_count > 0:
        raise RuntimeError(f"Migration completed with {error_count} errors")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate_ledger_schema()
```

---

## Performance Benchmarking Specification

### Benchmark Implementation

```python
# tests/performance/benchmark_dagster_vs_legacy.py
"""Performance comparison: Dagster vs Legacy orchestration."""
import time
import statistics
from dataclasses import dataclass
from typing import Literal

@dataclass
class BenchmarkResult:
    """Results from orchestration benchmark."""
    orchestration_type: Literal["legacy", "dagster"]
    job_count: int
    total_duration_seconds: float
    throughput_jobs_per_second: float
    stage_timings: dict[str, list[float]]  # stage -> list of durations
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_count: int

def run_benchmark(orchestration_type: Literal["legacy", "dagster"], job_count: int = 100) -> BenchmarkResult:
    """Run benchmark with specified orchestration type."""
    # Setup
    if orchestration_type == "legacy":
        orch = create_legacy_orchestrator()
    else:
        orch = create_dagster_orchestrator()

    # Submit jobs
    job_ids = []
    start = time.time()

    for i in range(job_count):
        job_id = orch.submit_job(
            tenant_id="benchmark",
            dataset="clinicaltrials",
            item={"id": f"NCT{i:08d}"}
        )
        job_ids.append(job_id)

    # Wait for all completions
    stage_timings = {stage: [] for stage in ["ingest", "parse", "chunk", "embed", "index"]}
    latencies = []
    error_count = 0

    for job_id in job_ids:
        try:
            result = wait_for_completion(job_id, timeout=60)
            latencies.append(result.total_duration_ms)

            # Collect stage timings
            for stage, duration in result.stage_durations.items():
                stage_timings[stage].append(duration)
        except TimeoutError:
            error_count += 1

    end = time.time()
    total_duration = end - start

    # Calculate percentiles
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies) // 2]
    p95 = latencies_sorted[int(len(latencies) * 0.95)]
    p99 = latencies_sorted[int(len(latencies) * 0.99)]

    return BenchmarkResult(
        orchestration_type=orchestration_type,
        job_count=job_count,
        total_duration_seconds=total_duration,
        throughput_jobs_per_second=job_count / total_duration,
        stage_timings=stage_timings,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        error_count=error_count
    )

def compare_performance():
    """Compare legacy vs Dagster orchestration."""
    legacy_result = run_benchmark("legacy", job_count=100)
    dagster_result = run_benchmark("dagster", job_count=100)

    print(f"Legacy throughput: {legacy_result.throughput_jobs_per_second:.2f} jobs/sec")
    print(f"Dagster throughput: {dagster_result.throughput_jobs_per_second:.2f} jobs/sec")
    print(f"Overhead: {((dagster_result.p95_latency_ms / legacy_result.p95_latency_ms) - 1) * 100:.1f}%")

    # Assert acceptable overhead (≤10%)
    assert dagster_result.p95_latency_ms <= legacy_result.p95_latency_ms * 1.1, \
        f"Dagster P95 latency ({dagster_result.p95_latency_ms}ms) exceeds 10% overhead threshold"

    # Assert throughput parity
    assert dagster_result.throughput_jobs_per_second >= legacy_result.throughput_jobs_per_second * 0.9, \
        f"Dagster throughput ({dagster_result.throughput_jobs_per_second:.2f}) below 90% of legacy"
```

---

## Open Questions

### Q1: Should we support dynamic pipeline generation?

**Current Answer**: No for MVP. Pipelines are static YAML files. If needed later, can add Python API to generate topologies programmatically.

### Q2: How do we handle pipeline versioning across deployments?

**Current Answer**: Pipeline configs versioned in `config/orchestration/versions/YYYY-MM-DD/`. Job Ledger stores `pipeline_version` field. Old jobs run with old topologies, new jobs use latest.

### Q3: Should CloudEvents be synchronous or async to Kafka?

**Current Answer**: Async (fire-and-forget) to avoid blocking stage execution. If Kafka publish fails, log warning but don't fail job.

### Q4: Can we run multiple Dagster jobs in parallel?

**Current Answer**: Yes, Dagster supports concurrent job execution. Resource contention (GPU, database connections) managed by existing resource pools and rate limiters.

### Q5: How do we test sensor-based gates without waiting for real MinerU?

**Current Answer**: Integration tests mock Job Ledger updates. Set `pdf_ir_ready=true` programmatically, verify sensor triggers post-PDF stages.

---

## Alternative Designs Considered

### Alternative 1: Airflow-Based Orchestration

**Pros**: Industry standard, rich ecosystem, mature scheduler
**Cons**: Heavyweight for local dev, complex DAG syntax, poor Python typing, requires scheduler process
**Verdict**: Rejected - too much operational overhead for our local-first requirements

### Alternative 2: Temporal Workflows

**Pros**: Strong consistency, durable execution, built-in retries
**Cons**: Requires separate Temporal server cluster, geared toward long-running workflows (hours/days), over-engineered for batch pipelines
**Verdict**: Rejected - infrastructure complexity outweighs benefits

### Alternative 3: Custom DAG Engine

**Pros**: Full control, minimal dependencies, tailored to our needs
**Cons**: Reinventing wheels (DAG execution, sensors, UI, retry logic), maintenance burden, no ecosystem support
**Verdict**: Rejected - Dagster provides 90% of what we need out-of-box

---

## Success Metrics

1. **Pipeline visibility**: 100% of jobs visualized in Dagster UI with stage-level status
2. **Topology agility**: New pipeline created and deployed in <1 hour (vs <1 day with code changes)
3. **Resilience configurability**: Policy changes applied via YAML edit, no code deployment
4. **GPU fail-fast enforcement**: 0 jobs fall back to CPU for GPU-required stages
5. **Lineage completeness**: Every stage execution produces CloudEvent, 100% traceable
6. **Migration success**: Legacy orchestration removed in v0.3.0 without production incidents
