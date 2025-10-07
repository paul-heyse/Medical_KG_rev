# Add DAG-Based Orchestration Pipeline

## Why

The current orchestration system hard-wires pipeline stage sequences in Python code, making it difficult to:

1. **Visualize and understand** the ingestion flow without reading implementation details
2. **Modify pipeline topology** without code changes (e.g., adding new stages, reordering operations)
3. **Handle complex gates** like the PDF two-phase path where processing must halt after download and resume only when GPU-backed MinerU produces IR
4. **Swap underlying implementations** without cascading changes (e.g., switching chunkers, embedders, or retrievers)
5. **Configure resilience policies** (retries, backoff, circuit breakers, rate limits) per-stage without editing code
6. **Track lineage and provenance** in a portable format that works across tools

This creates technical debt and operational friction, especially as we scale to more data sources, add new extraction types, and need to debug failed jobs across multi-stage pipelines.

## What Changes

### 1. **Externalized Pipeline Topology**

- **Declarative DAG files** in `config/orchestration/pipelines/` define stage sequences for:
  - `auto.yaml` - Non-PDF sources: ingest → parse → IR → chunk → embed → index → extract → KG
  - `pdf-two-phase.yaml` - PDF sources with explicit gate: download → **GATE(pdf_ir_ready)** → chunk → embed → index → extract → KG
  - Custom pipelines per source type (e.g., `clinical-trials.yaml`, `pmc-fulltext.yaml`)

- **Gate conditions** expressed as Job Ledger field predicates (e.g., `pdf_downloaded=true`, `pdf_ir_ready=true`)

- **Pipeline versioning** via timestamped configs in `config/orchestration/versions/` for auditability

### 2. **Typed Stage Contracts (Python Protocols)**

- Every stage defines **input/output contracts** using Python Protocols:
  - `IngestStage`: `AdapterRequest` → `RawPayload[]`
  - `ParseStage`: `RawPayload[]` → `Document` (IR)
  - `ChunkStage`: `Document` → `Chunk[]`
  - `EmbedStage`: `Chunk[]` → `EmbeddingBatch`
  - `IndexStage`: `EmbeddingBatch` → `IndexReceipt`
  - `ExtractStage`: `Document` → `Entity[]` + `Claim[]`
  - `KGStage`: `Entity[]` + `Claim[]` → `GraphWriteReceipt`

- **Ports-and-adapters pattern**: Stage contracts are thin, stable boundaries allowing swapping of underlying implementations (Haystack components, LlamaIndex, custom code) without changing orchestration

### 3. **Dagster as Local Workflow Engine**

- **Library**: Dagster 1.5.0+ (<https://dagster.io/>)
- **Rationale**: Provides local-first DAG execution, sensor-based gate handling, UI for runs/retries, and minimal infrastructure requirements (no Airflow scheduler, no distributed coordinator)

- **Two lightweight jobs**:
  - `auto_pipeline_job` - Fully automatic for non-PDF sources
  - `pdf_two_phase_job` - Runs in two steps with sensor polling Job Ledger for `pdf_ir_ready` state

- **Sensors** poll the Job Ledger; when `pdf_ir_ready=true`, they start the "post-PDF" half of the graph

- **Local UI** at `http://localhost:3000` for visualizing runs, retries, and stage-level failures

### 4. **Haystack 2 for Text Operations**

- **Library**: haystack-ai 2.0+ (<https://haystack.deepset.ai/>)
- **Rationale**: Provides composable components (document stores, retrievers, embedders, routers) that encapsulate vendor-specific details while keeping stage contracts stable

- **Wrapped Haystack components**:
  - `HaystackChunker` wraps `DocumentSplitter` for semantic chunking
  - `HaystackEmbedder` wraps `OpenAIDocumentEmbedder` for Qwen via vLLM (OpenAI-compatible endpoint)
  - `HaystackSparseExpander` wraps custom SPLADE expansion
  - `HaystackIndexWriter` wraps `OpenSearchDocumentWriter` + `FAISSDocumentWriter`
  - `HaystackRetriever` wraps `OpenSearchBM25Retriever` + `FAISSEmbeddingRetriever` + `RRFFusionRanker`

- **Preserves hybrid retrieval**: BM25 + SPLADE (OpenSearch) + Dense (FAISS) with fusion ranking

### 5. **Resilience as Configuration**

- **Named policies** in `config/orchestration/resilience.yaml`:
  - `default`: 3 retries, exponential backoff with jitter, 30s timeout
  - `gpu-bound`: 1 retry (fail-fast), no backoff, 60s timeout, circuit breaker after 5 failures
  - `polite-api`: 10 retries, linear backoff, 10s timeout, rate limit 5 req/s

- **Libraries**:
  - `tenacity>=8.2.0` (<https://tenacity.readthedocs.io/>) - Retries and backoff at stage boundary
  - `pybreaker>=1.0.0` (<https://github.com/danielfm/pybreaker>) - Circuit breakers for unreliable externals
  - `aiolimiter>=1.1.0` (<https://github.com/mjpieters/aiolimiter>) - Async rate limiting for polite API calls

- **GPU fail-fast enforcement**: If MinerU, SPLADE, or Qwen report "no GPU", job marked failed immediately with no implicit CPU fallback

### 6. **CloudEvents + OpenLineage for Observability**

- **CloudEvents 1.0** (<https://cloudevents.io/>) - Portable event envelope for stage lifecycle:
  - Stage start: `{"type": "stage.started", "source": "orchestration", "data": {"doc_id": "...", "stage": "chunk"}}`
  - Stage finish: `{"type": "stage.completed", "data": {"duration_ms": 1234, "chunks_produced": 42}}`
  - Stage failure: `{"type": "stage.failed", "data": {"error": "...", "retry_count": 2}}`

- **OpenLineage** (<https://openlineage.io/>) - Optional lineage records for Marquez visualization:
  - Job runs with parent/child relationships
  - Dataset reads/writes per stage
  - Facets for GPU usage, model versions, retry attempts

- **Kafka topic**: `orchestration.events.v1` for CloudEvents stream

### 7. **AsyncAPI for Queue Documentation**

- **AsyncAPI 3.0 spec** in `docs/asyncapi.yaml` documents:
  - `ingest.requests.v1` - Ingestion job submissions
  - `ingest.results.v1` - Ingestion completion notifications
  - `orchestration.events.v1` - Stage lifecycle CloudEvents
  - `mineru.queue.v1` - PDF processing requests for GPU nodes

- **Not a streaming mandate**: Describes existing Kafka topics for clarity, not introducing streaming architecture

### 8. **Testing with respx**

- **Library**: respx 0.20.0+ (<https://lundberg.github.io/respx/>) - Mock HTTP calls in tests
- **Usage**: Validate resilience behavior (retries, circuit breakers), adapter calls, model endpoint interactions without live external dependencies

## Impact

### Affected Capabilities

1. **orchestration** - Complete refactor from hardcoded stages to DAG-based execution
2. **ingestion** - Pipeline definitions externalized, plugin execution via Dagster
3. **retrieval** - Chunking, embedding, indexing wrapped in Haystack components
4. **observability** - CloudEvents + OpenLineage for stage-level tracing
5. **configuration** - New resilience policies, pipeline topology configs

### Affected Code

**Core Modules**:

- `src/Medical_KG_rev/orchestration/` - New Dagster job definitions, sensor logic, stage wrappers
- `src/Medical_KG_rev/services/` - Haystack component adapters for chunking, embedding, indexing
- `config/orchestration/` - Pipeline topology YAMLs, resilience policies
- `tests/orchestration/` - New Dagster tests, respx-based adapter mocks

**Dependencies**:

- **Added**: dagster>=1.5.0, haystack-ai>=2.0.0, tenacity>=8.2.0, pybreaker>=1.0.0, aiolimiter>=1.1.0, cloudevents>=1.9.0, openlineage-python>=1.0.0, respx>=0.20.0
- **Modified**: orchestration service interfaces, ingestion pipeline execution

### Breaking Changes

- **BREAKING**: Orchestrator API changes from `execute_pipeline(job_id)` to `submit_to_dagster(job_id, pipeline_name)`
- **BREAKING**: Stage handlers now receive typed `StageContext` instead of raw `dict[str, object]`
- **BREAKING**: Resilience policies must be defined in config; hardcoded retries removed

### Migration Path

1. **Phase 1**: Add Dagster alongside existing orchestration (feature flag `MK_USE_DAGSTER=false`)
2. **Phase 2**: Migrate non-PDF pipelines to Dagster, validate with integration tests
3. **Phase 3**: Migrate PDF two-phase pipeline with sensor-based gate
4. **Phase 4**: Remove legacy orchestration code (v0.3.0 release)

### Benefits

- **Visibility**: Dagster UI shows real-time pipeline execution, stage timings, failure points
- **Flexibility**: Add/remove/reorder stages by editing YAML, no code changes
- **Reliability**: Centralized resilience policies, circuit breakers prevent cascading failures
- **Debuggability**: CloudEvents + OpenLineage provide complete audit trail for job troubleshooting
- **Testability**: respx mocks enable comprehensive stage-level testing without live services
- **Extensibility**: Haystack components can be swapped (e.g., LlamaIndex chunks, HuggingFace embedders) without touching orchestration

### Risks

- **Learning curve**: Team must learn Dagster concepts (jobs, ops, sensors, resources)
- **Migration complexity**: Two-phase rollout required to avoid disrupting production ingestion
- **Operational overhead**: Dagster UI adds another service to monitor (mitigated by local-first design)
- **Haystack maturity**: Version 2.0 is new; potential for breaking changes (mitigated by stage contracts insulating rest of system)

### Mitigation

- **Documentation**: Comprehensive guides for pipeline authoring, stage implementation, resilience tuning
- **Feature flags**: Gradual rollout with ability to revert to legacy orchestration
- **Integration tests**: Full end-to-end tests for both pipeline paths (auto, PDF two-phase)
- **Monitoring**: CloudEvents stream feeds existing Prometheus/OpenTelemetry observability
