# Orchestration Spec Delta

## ADDED Requirements

### Requirement: Declarative Pipeline Topology

The orchestration system SHALL load pipeline definitions from YAML files located in `config/orchestration/pipelines/`. Each pipeline definition MUST specify:

- `version`: Semantic version string (e.g., "1.0")
- `name`: Unique pipeline identifier (e.g., "auto", "pdf-two-phase")
- `description`: Human-readable description
- `applicable_sources`: List of adapter names this pipeline applies to
- `stages`: Ordered list of stage definitions with:
  - `name`: Stage identifier
  - `type`: Stage contract type (IngestStage, ParseStage, ChunkStage, etc.)
  - `policy`: Reference to named resilience policy
  - `depends_on`: List of prerequisite stage names (optional)
  - `condition`: Gate condition for conditional execution (optional)

#### Scenario: Auto pipeline loaded from YAML

- **GIVEN** a file `config/orchestration/pipelines/auto.yaml` exists
- **WHEN** the orchestration system initializes
- **THEN** the auto pipeline is loaded with all 8 stages (ingest, parse, ir_validation, chunk, embed, index, extract, kg)
- **AND** stage dependencies are validated to form a DAG (no cycles)
- **AND** all referenced resilience policies exist in `config/orchestration/resilience.yaml`

#### Scenario: PDF two-phase pipeline with gate

- **GIVEN** a file `config/orchestration/pipelines/pdf-two-phase.yaml` with a gate stage
- **WHEN** the pipeline is loaded
- **THEN** the gate condition `pdf_ir_ready=true` is parsed from YAML
- **AND** the resume stage `chunk` is identified as the first post-gate stage
- **AND** the Dagster sensor is configured to poll the Job Ledger for this condition

---

### Requirement: Typed Stage Contracts

Each pipeline stage MUST implement a Python Protocol defining its input and output types. The system SHALL provide the following standard contracts:

- `IngestStage`: `execute(ctx: StageContext, request: AdapterRequest) -> list[RawPayload]`
- `ParseStage`: `execute(ctx: StageContext, payloads: list[RawPayload]) -> Document`
- `ChunkStage`: `execute(ctx: StageContext, document: Document) -> list[Chunk]`
- `EmbedStage`: `execute(ctx: StageContext, chunks: list[Chunk]) -> EmbeddingBatch`
- `IndexStage`: `execute(ctx: StageContext, batch: EmbeddingBatch) -> IndexReceipt`
- `ExtractStage`: `execute(ctx: StageContext, document: Document) -> tuple[list[Entity], list[Claim]]`
- `KGStage`: `execute(ctx: StageContext, entities: list[Entity], claims: list[Claim]) -> GraphWriteReceipt`

All stage implementations MUST satisfy their respective Protocol via structural subtyping.

#### Scenario: Custom chunker implements ChunkStage

- **GIVEN** a custom chunker class `SemanticChunker`
- **WHEN** the class defines `execute(ctx: StageContext, document: Document) -> list[Chunk]`
- **THEN** the class satisfies the `ChunkStage` Protocol
- **AND** the orchestrator can invoke it without knowing the concrete implementation
- **AND** static type checkers (mypy) validate the contract

#### Scenario: Stage contract violation detected

- **GIVEN** a stage implementation with wrong method signature
- **WHEN** the Dagster job is defined with this stage
- **THEN** a runtime error is raised during job initialization
- **AND** the error message indicates which Protocol method is missing or mismatched

---

### Requirement: Dagster-Based Job Execution

The orchestration system SHALL use Dagster (<https://dagster.io/>) version 1.5.0+ as the workflow engine. The system MUST define:

- **auto_pipeline_job**: Dagster job for non-PDF sources with ops for all 8 stages executed in sequence
- **pdf_two_phase_job**: Dagster job for PDF sources with pre-PDF ops (ingest, download) and post-PDF ops (chunk, embed, index, extract, kg)
- **pdf_ir_ready_sensor**: Dagster sensor polling the Job Ledger every 10 seconds for `pdf_ir_ready=true`, triggering post-PDF ops when condition met

Each Dagster op MUST wrap a stage contract implementation, apply resilience policies, and emit CloudEvents on start/finish/failure.

#### Scenario: Auto pipeline job execution

- **GIVEN** a job submission for ClinicalTrials.gov adapter (non-PDF)
- **WHEN** `submit_to_dagster(job_id, "auto")` is called
- **THEN** the `auto_pipeline_job` is launched in Dagster
- **AND** all 8 stages execute in sequence: ingest → parse → ir_validation → chunk → embed → index → extract → kg
- **AND** intermediate outputs are passed between ops via Dagster's internal storage
- **AND** the Dagster UI shows real-time progress for each stage

#### Scenario: PDF two-phase job with sensor resume

- **GIVEN** a job submission for PMC full-text (PDF source)
- **WHEN** `submit_to_dagster(job_id, "pdf-two-phase")` is called
- **THEN** the pre-PDF ops (ingest, download) execute
- **AND** the job pauses after download, setting `pdf_downloaded=true` in Job Ledger
- **WHEN** MinerU completes and sets `pdf_ir_ready=true`
- **THEN** the `pdf_ir_ready_sensor` detects the condition within 10 seconds
- **AND** the post-PDF ops (chunk, embed, index, extract, kg) execute
- **AND** the Job Ledger is updated with each stage completion

---

### Requirement: Resilience Policy Configuration

The system SHALL load named resilience policies from `config/orchestration/resilience.yaml`. Each policy MUST specify:

- `max_attempts`: Maximum retry attempts (1-10)
- `backoff_strategy`: Exponential, linear, or none
- `backoff_initial_seconds`: Initial backoff delay (0.1-10.0)
- `backoff_max_seconds`: Maximum backoff delay (1.0-300.0)
- `backoff_jitter_seconds`: Jitter range for randomization (0.0-5.0)
- `timeout_seconds`: Per-attempt timeout (1-600)
- `circuit_breaker`: Optional circuit breaker config with:
  - `failure_threshold`: Failures before opening (3-10)
  - `reset_timeout_seconds`: Time before half-open retry (30-600)
- `rate_limit_per_second`: Optional requests per second limit (0.1-100.0)

Policies SHALL be applied at stage execution boundaries using tenacity, pybreaker, and aiolimiter libraries.

#### Scenario: Default policy applied to parse stage

- **GIVEN** the parse stage references policy "default" in `auto.yaml`
- **WHEN** the parse op executes and fails with a transient error (HTTP 503)
- **THEN** the stage retries up to 3 times with exponential backoff (1s, 2s, 4s)
- **AND** jitter is added to prevent thundering herd
- **AND** Prometheus metric `dagster_retry_attempts_total{stage="parse"}` increments

#### Scenario: GPU-bound policy enforces fail-fast

- **GIVEN** the embed stage references policy "gpu-bound"
- **WHEN** the embed op executes and detects no GPU available
- **THEN** the stage fails immediately with `GpuNotAvailableError`
- **AND** no retries are attempted (max_attempts=1)
- **AND** the job is marked as `gpu_unavailable` in Job Ledger
- **AND** a CloudEvent `stage.failed` with reason "no_gpu" is emitted

#### Scenario: Circuit breaker opens after threshold

- **GIVEN** the ingest stage references policy "polite-api" with circuit breaker (failure_threshold=5)
- **WHEN** the adapter fails 5 consecutive times (e.g., upstream API down)
- **THEN** the circuit breaker opens
- **AND** subsequent attempts fail immediately without calling the adapter
- **AND** after reset_timeout_seconds (60s), the circuit enters half-open state
- **AND** one test request is allowed through; if it succeeds, circuit closes

---

### Requirement: CloudEvents Stage Lifecycle

Each stage execution MUST emit CloudEvents (<https://cloudevents.io/>) version 1.0 to the Kafka topic `orchestration.events.v1`. The system SHALL produce:

- `org.medicalkg.orchestration.stage.started`: Emitted when stage begins, with `data.stage`, `data.doc_id`, `data.correlation_id`
- `org.medicalkg.orchestration.stage.completed`: Emitted on success, with `data.duration_ms`, `data.output_count`, `data.gpu_used`
- `org.medicalkg.orchestration.stage.failed`: Emitted on failure, with `data.error_message`, `data.retry_count`, `data.policy_name`
- `org.medicalkg.orchestration.stage.retrying`: Emitted before retry, with `data.attempt_number`, `data.backoff_ms`

All events MUST include `subject` field with document ID and `source` field with stage name.

#### Scenario: Chunk stage emits lifecycle events

- **GIVEN** a document being chunked in the auto pipeline
- **WHEN** the chunk stage starts execution
- **THEN** a CloudEvent `stage.started` is published with `subject=doc_id`, `source=orchestration/chunk`
- **WHEN** chunking completes after 1.2 seconds producing 42 chunks
- **THEN** a CloudEvent `stage.completed` is published with `data.duration_ms=1200`, `data.output_count=42`, `data.gpu_used=false`

#### Scenario: Embed stage retries and emits retry event

- **GIVEN** an embed stage with max_attempts=3
- **WHEN** the first attempt fails with a timeout
- **THEN** a CloudEvent `stage.retrying` is published with `data.attempt_number=1`, `data.backoff_ms=2000`
- **WHEN** the second attempt succeeds
- **THEN** a CloudEvent `stage.completed` is published with `data.retry_count=1`

---

### Requirement: Haystack Component Integration

For text operations (chunking, embedding, indexing), the system SHOULD use Haystack 2.0+ (<https://haystack.deepset.ai/>) components wrapped to satisfy stage contracts. The system MUST provide:

- `HaystackChunker`: Wraps `DocumentSplitter` from `haystack.components.preprocessors`
- `HaystackEmbedder`: Wraps `OpenAIDocumentEmbedder` pointing to vLLM endpoint for Qwen
- `HaystackSparseExpander`: Custom component for SPLADE expansion on GPU
- `HaystackIndexWriter`: Wraps `OpenSearchDocumentWriter` and `FAISSDocumentWriter` for dual indexing
- `HaystackRetriever`: Wraps `OpenSearchBM25Retriever`, `FAISSEmbeddingRetriever`, and `RRFFusionRanker`

All wrappers MUST convert between IR models and Haystack `Document` format, preserving metadata and provenance.

#### Scenario: HaystackChunker converts IR to Haystack and back

- **GIVEN** an IR Document with 3 sections
- **WHEN** `HaystackChunker.execute(ctx, document)` is called
- **THEN** the IR Document is converted to 3 Haystack Documents
- **AND** Haystack `DocumentSplitter` splits them into 12 chunks
- **AND** the 12 Haystack chunks are converted back to IR `Chunk[]` with provenance metadata
- **AND** each chunk retains `tenant_id`, `source_doc_id`, and `section_id`

#### Scenario: HaystackEmbedder uses vLLM for Qwen

- **GIVEN** a vLLM server running Qwen-3 on GPU at `http://localhost:8000`
- **WHEN** `HaystackEmbedder.execute(ctx, chunks)` is called with 32 chunks
- **THEN** the chunks are batched into 1 request (batch_size=32)
- **AND** the embedder sends a POST to `http://localhost:8000/v1/embeddings` with OpenAI-compatible format
- **AND** the response vectors (dimension 512) are returned as `EmbeddingBatch`
- **AND** GPU utilization metric is recorded

---

## MODIFIED Requirements

### Requirement: Job Ledger Tracking

The Job Ledger MUST be extended to support pipeline execution state. New fields:

- `pdf_downloaded: bool` - Set to true after PDF retrieval stage completes
- `pdf_ir_ready: bool` - Set to true after MinerU processing completes
- `current_stage: str` - Name of the currently executing stage (e.g., "chunk", "embed")
- `pipeline_name: str` - Name of the pipeline being executed (e.g., "auto", "pdf-two-phase")
- `retry_count_per_stage: dict[str, int]` - Map of stage name to retry attempts

The ledger API MUST support querying by gate conditions (e.g., `query({"pdf_ir_ready": True, "post_pdf_started": False})`).

#### Scenario: PDF gate condition query

- **GIVEN** 100 jobs in the ledger, 10 with `pdf_ir_ready=true` and `post_pdf_started=false`
- **WHEN** the `pdf_ir_ready_sensor` queries for resume-eligible jobs
- **THEN** exactly 10 job IDs are returned
- **AND** the sensor starts post-PDF ops for each job
- **AND** the ledger is updated with `post_pdf_started=true` to prevent duplicate processing

#### Scenario: Stage retry count tracking

- **GIVEN** a chunk stage that failed 2 times and succeeded on the 3rd attempt
- **WHEN** the job completes
- **THEN** the ledger field `retry_count_per_stage["chunk"]` equals 2
- **AND** this data is available for debugging and metrics

---

## REMOVED Requirements

### Requirement: Hardcoded Pipeline Stages

**Reason**: Replaced by declarative YAML topology

**Migration**: Existing `Orchestrator._register_default_pipelines()` logic will be deprecated. Pipeline definitions in `config/orchestration/pipelines/` replace hardcoded stage sequences.

---

## RENAMED Requirements

- FROM: `Requirement: Pipeline Execution`
- TO: `Requirement: Dagster-Based Job Execution`

(Content updated to reflect Dagster as the execution engine instead of custom Python orchestration)
