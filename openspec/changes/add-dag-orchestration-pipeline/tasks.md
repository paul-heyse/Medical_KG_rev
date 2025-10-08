# Implementation Tasks: DAG-Based Orchestration Pipeline

## CRITICAL: Hard Cutover Strategy

**No Legacy Compatibility** - This is a complete replacement, not a migration:

- ❌ No feature flags (`MK_USE_DAGSTER` removed from scope)
- ❌ No compatibility shims or legacy code preservation
- ❌ No gradual rollout or A/B testing
- ✅ Delete legacy orchestration in same commits as new implementation
- ✅ Single feature branch with full replacement
- ✅ Rollback = revert entire branch (not toggle back to legacy)

---

## 1. Legacy Code Decommissioning Plan

**Purpose**: Systematically eliminate all bespoke orchestration code being replaced by Dagster/Haystack

### 1.1 Legacy Code Inventory (Audit Phase)

- [x] 1.1.1 Identify all files in `src/Medical_KG_rev/orchestration/` to be deleted:
- [x] `orchestrator.py` (lines 1-176) - Replace with Dagster jobs
- [x] `worker.py` (lines 62-110) - Replace with Dagster ops
  - [ ] `pipeline.py` - Replace with YAML topology configs
  - [ ] `profiles.py` - Replace with per-pipeline YAML configs
- [x] 1.1.2 Identify all files in `src/Medical_KG_rev/services/` with bespoke stage logic:
  - [ ] `services/retrieval/indexing_service.py` - Replace with HaystackIndexWriter
  - [ ] `services/embedding/service.py` (bespoke retry logic) - Replace with tenacity decorators
  - [x] `services/retrieval/chunking.py` (custom splitters) - Replace with HaystackChunker
- [x] 1.1.3 Create deletion checklist: `LEGACY_DECOMMISSION_CHECKLIST.md` with:
  - File paths to delete
  - Imports to update
  - Tests to delete/replace
  - Documentation to update

### 1.2 Dependency Analysis (Pre-Delete Validation)

- [x] 1.2.1 Run `grep -r "from.*orchestrator import" src/` to find all imports of legacy orchestrator
- [x] 1.2.2 Run `grep -r "Orchestrator\(" src/` to find all instantiations
- [x] 1.2.3 Run `grep -r "execute_pipeline" src/` to find all calls to legacy API
- [x] 1.2.4 Run `grep -r "IngestWorker\|MappingWorker" src/` to find worker usage
- [x] 1.2.5 Document all dependencies in `LEGACY_DEPENDENCIES.md` with replacement plan

### 1.3 Delegation to Open-Source Libraries (Validation)

- [x] 1.3.1 **Chunking**: Verify all chunking logic moved to Haystack `DocumentSplitter`
  - [x] Audit: `src/Medical_KG_rev/services/retrieval/chunking.py` - which methods stay?
  - [x] Decision: Keep profile detection, delegate splitting to Haystack
  - [x] Delete: Custom `SemanticSplitter`, `SlidingWindow` implementations
  - [x] Verify: All chunking tests pass with Haystack backend
- [ ] 1.3.2 **Embedding**: Verify all embedding calls use Haystack `OpenAIDocumentEmbedder`
  - [ ] Audit: `src/Medical_KG_rev/services/embedding/service.py` - which methods stay?
  - [ ] Decision: Keep namespace management, delegate embedding to Haystack
  - [ ] Delete: Custom embedding loops, batch processing logic (Haystack handles)
  - [ ] Verify: GPU fail-fast preserved in Haystack wrapper
- [ ] 1.3.3 **Retry Logic**: Verify all retries use tenacity decorators
  - [ ] Audit: Search for `for attempt in range(max_retries)` patterns
  - [ ] Decision: Delete all custom retry loops
  - [ ] Replace: With `@retry_on_failure` decorator from resilience policies
  - [ ] Verify: Retry behavior identical (exponential backoff, jitter)
- [ ] 1.3.4 **Circuit Breakers**: Verify all failure handling uses pybreaker
  - [ ] Audit: Search for custom failure counting logic
  - [ ] Decision: Delete all bespoke circuit breaker implementations
  - [ ] Replace: With `@circuit_breaker` decorator from resilience policies
  - [ ] Verify: Circuit state transitions match expected behavior
- [ ] 1.3.5 **Rate Limiting**: Verify all throttling uses aiolimiter
  - [ ] Audit: Search for `time.sleep()` or custom rate limit logic
  - [ ] Decision: Delete all bespoke rate limiting
  - [ ] Replace: With `@rate_limit` decorator from resilience policies
  - [ ] Verify: Request rates honor configured limits

### 1.4 Atomic Deletion (Commit Strategy)

- [ ] 1.4.1 Create commit plan with atomic deletions:
  - [ ] Commit 1: Add Dagster jobs + delete `orchestrator.py` + delete `worker.py`
  - [ ] Commit 2: Add HaystackChunker + delete custom chunker implementations
  - [ ] Commit 3: Add HaystackEmbedder + delete custom embedding loops
  - [ ] Commit 4: Add resilience decorators + delete all custom retry/circuit breaker logic
  - [ ] Commit 5: Add HaystackIndexWriter + delete `indexing_service.py`
  - [ ] Commit 6: Update all imports to point to new implementations
  - [ ] Commit 7: Delete all legacy tests, add new Dagster/Haystack tests
- [ ] 1.4.2 Run full test suite after each commit to ensure no regressions
- [ ] 1.4.3 Document deleted code statistics in commit messages (lines removed, files deleted)

### 1.5 Import Cleanup (Post-Delete)

- [x] 1.5.1 Update `src/Medical_KG_rev/orchestration/__init__.py`:
  - [x] Remove: `from .orchestrator import Orchestrator, OrchestrationError`
  - [x] Remove: `from .worker import IngestWorker, MappingWorker`
  - [x] Add: `from .dagster import DagsterOrchestrator, submit_to_dagster`
- [x] 1.5.2 Update `src/Medical_KG_rev/gateway/services.py`:
  - [x] Remove: `self.orchestrator = Orchestrator(...)`
  - [x] Add: `self.orchestrator = DagsterOrchestrator(...)`
- [ ] 1.5.3 Run `ruff check --select F401` to find unused imports
- [ ] 1.5.4 Run `mypy src/` to verify no type errors from deletions

### 1.6 Test Migration (Delete and Replace)

- [x] 1.6.1 Delete legacy orchestration tests:
  - [x] `tests/orchestration/test_orchestrator.py` (legacy API tests)
  - [x] `tests/orchestration/test_workers.py` (worker-specific tests)
  - [x] `tests/orchestration/test_integration.py` (references `execute_pipeline`)
- [ ] 1.6.2 Create new Dagster tests:
  - [ ] `tests/orchestration/test_dagster_jobs.py` (auto, PDF two-phase)
  - [ ] `tests/orchestration/test_dagster_sensors.py` (pdf_ir_ready_sensor)
  - [ ] `tests/orchestration/test_stage_contracts.py` (Protocol compliance)
- [ ] 1.6.3 Verify test coverage ≥90% for new orchestration code
- [ ] 1.6.4 Delete all references to `Orchestrator` in test fixtures

### 1.7 Documentation Updates (Reflect Deletions)

- [ ] 1.7.1 Update `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md`:
  - [ ] Remove: Section 6.2 "Legacy Pipeline Stages"
  - [ ] Add: Section 6.2 "Dagster Job Definitions"
  - [ ] Remove: All references to `Orchestrator.execute_pipeline`
  - [ ] Add: All references to `submit_to_dagster`
- [ ] 1.7.2 Update `docs/guides/orchestration-pipelines.md`:
  - [ ] Remove: Legacy examples with hardcoded stages
  - [ ] Add: YAML topology examples with Dagster
- [ ] 1.7.3 Update `README.md`:
  - [x] Remove: "Start background workers" (no longer needed)
  - [x] Add: "Start Dagster daemon" command
- [ ] 1.7.4 Create `DELETED_CODE.md` documenting what was removed and why

### 1.8 Codebase Size Validation

- [ ] 1.8.1 Measure codebase before changes:
  - [ ] Run `cloc src/Medical_KG_rev/orchestration/ src/Medical_KG_rev/services/` (baseline)
  - [ ] Record: Total lines of Python code, number of files
- [ ] 1.8.2 Measure codebase after changes:
  - [ ] Run `cloc` on same paths (post-deletion)
  - [ ] Record: Lines removed, files deleted, net reduction
- [ ] 1.8.3 Validate codebase shrinkage:
  - [ ] Assert: Total lines reduced by ≥30% (bespoke → library delegation)
  - [ ] Assert: Number of files reduced by ≥20% (consolidation)
  - [ ] Document: Actual numbers in `CODEBASE_REDUCTION_REPORT.md`

---

## 2. Foundation & Dependencies

- [x] 1.1 Add **dagster>=1.5.0** to `requirements.txt` and `pyproject.toml`
- [x] 1.2 Add **haystack-ai>=2.0.0** with OpenSearch and FAISS extras
- [x] 1.3 Add **tenacity>=8.2.0** for retry decorators
- [x] 1.4 Add **pybreaker>=1.0.0** for circuit breaker pattern
- [x] 1.5 Add **aiolimiter>=1.1.0** for async rate limiting
- [x] 1.6 Add **cloudevents>=1.9.0** for event envelope format
- [x] 1.7 Add **openlineage-python>=1.0.0** for lineage tracking (optional)
- [x] 1.8 Add **respx>=0.20.0** to `requirements-dev.txt` for HTTP mocking in tests

## 2. Stage Contracts (Python Protocols)

- [x] 2.1 Define `StageContext` dataclass with tenant_id, doc_id, correlation_id, metadata
- [x] 2.2 Define `IngestStage` Protocol: `execute(ctx: StageContext, request: AdapterRequest) -> list[RawPayload]`
- [x] 2.3 Define `ParseStage` Protocol: `execute(ctx: StageContext, payloads: list[RawPayload]) -> Document`
- [x] 2.4 Define `ChunkStage` Protocol: `execute(ctx: StageContext, document: Document) -> list[Chunk]`
- [x] 2.5 Define `EmbedStage` Protocol: `execute(ctx: StageContext, chunks: list[Chunk]) -> EmbeddingBatch`
- [x] 2.6 Define `IndexStage` Protocol: `execute(ctx: StageContext, batch: EmbeddingBatch) -> IndexReceipt`
- [x] 2.7 Define `ExtractStage` Protocol: `execute(ctx: StageContext, document: Document) -> tuple[list[Entity], list[Claim]]`
- [x] 2.8 Define `KGStage` Protocol: `execute(ctx: StageContext, entities: list[Entity], claims: list[Claim]) -> GraphWriteReceipt`
- [x] 2.9 Create `src/Medical_KG_rev/orchestration/stages/contracts.py` with all protocols
- [x] 2.10 Add type hints and docstrings for each protocol method

## 3. Pipeline Configuration Schema

- [x] 3.1 Define `PipelineTopologyConfig` Pydantic model for YAML parsing
- [x] 3.2 Define `StageDefinition` with name, stage_type, policy_ref, dependencies
- [x] 3.3 Define `GateDefinition` with condition (ledger field predicates), resume_stage
- [x] 3.4 Define `ResiliencePolicyConfig` with max_attempts, backoff_strategy, timeout_seconds, circuit_breaker_config, rate_limit_config
- [x] 3.5 Create schema validation for pipeline YAML files
- [x] 3.6 Add JSON Schema export for documentation (`docs/guides/pipeline-schema.json`)
- [x] 3.7 Write comprehensive config validation tests

## 4. Resilience Configuration

- [x] 4.1 Create `config/orchestration/resilience.yaml` with named policies:
  - [x] 4.1.1 `default`: 3 retries, exponential backoff with jitter, 30s timeout
  - [x] 4.1.2 `gpu-bound`: 1 retry, no backoff, 60s timeout, circuit breaker (5 failures, 60s reset)
  - [x] 4.1.3 `polite-api`: 10 retries, linear backoff, 10s timeout, 5 req/s rate limit
- [x] 4.2 Implement `ResiliencePolicyLoader` to load and validate policies from config
- [x] 4.3 Create `tenacity.retry` decorator factory from policy config
- [x] 4.4 Create `pybreaker.CircuitBreaker` factory from policy config
- [x] 4.5 Create `aiolimiter.AsyncLimiter` factory from policy config
- [x] 4.6 Implement policy application at stage execution boundary
- [x] 4.7 Add Prometheus metrics for retry attempts, circuit breaker state changes, rate limit waits
- [x] 4.8 Write resilience policy unit tests with respx mocks

## 5. Pipeline Topology Definitions

- [x] 5.1 Create `config/orchestration/pipelines/auto.yaml`:
  - [x] 5.1.1 Define stages: ingest, parse, ir_validation, chunk, embed, index, extract, kg
  - [x] 5.1.2 Set resilience policies per stage
  - [x] 5.1.3 Add metadata (version, description, applicable_sources)
- [x] 5.2 Create `config/orchestration/pipelines/pdf-two-phase.yaml`:
  - [x] 5.2.1 Define pre-PDF stages: ingest, download
  - [x] 5.2.2 Add GATE(pdf_ir_ready) with ledger condition
  - [x] 5.2.3 Define post-PDF stages: chunk, embed, index, extract, kg
  - [x] 5.2.4 Set gpu-bound policy for chunk (uses SPLADE), embed (uses Qwen), extract (uses LLM)
- [x] 5.3 Create `config/orchestration/pipelines/clinical-trials.yaml` (auto pipeline variant)
- [x] 5.4 Create `config/orchestration/pipelines/pmc-fulltext.yaml` (PDF pipeline variant)
- [x] 5.5 Add pipeline versioning: copy to `config/orchestration/versions/YYYY-MM-DD/`
- [x] 5.6 Implement `PipelineLoader` to load and cache topology configs
- [x] 5.7 Write pipeline config validation tests

## 6. Haystack Component Wrappers

- [x] 6.1 Implement `HaystackChunker` wrapping `DocumentSplitter`:
  - [x] 6.1.1 Satisfy `ChunkStage` protocol
  - [x] 6.1.2 Convert IR `Document` to Haystack `Document` format
  - [x] 6.1.3 Apply semantic chunking with coherence scoring
  - [x] 6.1.4 Convert back to IR `Chunk[]` with provenance
- [x] 6.2 Implement `HaystackEmbedder` wrapping `OpenAIDocumentEmbedder`:
  - [x] 6.2.1 Satisfy `EmbedStage` protocol
  - [x] 6.2.2 Point to local vLLM endpoint (Qwen-3 via OpenAI-compatible API)
  - [x] 6.2.3 Batch processing with GPU utilization tracking
  - [x] 6.2.4 Fail-fast if GPU unavailable
- [x] 6.3 Implement `HaystackSparseExpander` for SPLADE:
  - [x] 6.3.1 Custom Haystack component for sparse vector generation
  - [x] 6.3.2 GPU-backed expansion term generation
  - [x] 6.3.3 Fail-fast GPU requirement
- [x] 6.4 Implement `HaystackIndexWriter` wrapping `OpenSearchDocumentWriter` + `FAISSDocumentWriter`:
  - [x] 6.4.1 Satisfy `IndexStage` protocol
  - [x] 6.4.2 Dual write to OpenSearch (BM25 + SPLADE) and FAISS (dense)
  - [x] 6.4.3 Tenant-aware index naming
  - [x] 6.4.4 Transactional semantics (both succeed or both fail)
- [x] 6.5 Implement `HaystackRetriever` for hybrid search:
  - [x] 6.5.1 Wrap `OpenSearchBM25Retriever`, `FAISSEmbeddingRetriever`
  - [x] 6.5.2 Integrate `RRFFusionRanker` for result merging
  - [x] 6.5.3 Maintain existing retrieval API compatibility
- [x] 6.6 Write comprehensive Haystack wrapper unit tests with mocks
- [ ] 6.7 Write integration tests with real OpenSearch + FAISS instances

## 7. Dagster Job Definitions

- [ ] 7.1 Create `src/Medical_KG_rev/orchestration/dagster/` package
- [ ] 7.2 Define Dagster resources:
  - [ ] 7.2.1 `plugin_manager_resource` for adapter discovery
  - [ ] 7.2.2 `job_ledger_resource` for state tracking
  - [ ] 7.2.3 `kafka_resource` for event publishing
  - [ ] 7.2.4 `haystack_pipeline_resource` for component initialization
- [ ] 7.3 Implement `auto_pipeline_job`:
  - [ ] 7.3.1 Define ops for each stage (ingest, parse, chunk, embed, index, extract, kg)
  - [ ] 7.3.2 Wire op dependencies from `auto.yaml` topology
  - [ ] 7.3.3 Apply resilience policies via decorators
  - [ ] 7.3.4 Emit CloudEvents on stage start/finish/failure
- [ ] 7.4 Implement `pdf_two_phase_job`:
  - [ ] 7.4.1 Define pre-PDF ops (ingest, download)
  - [ ] 7.4.2 Define sensor `pdf_ir_ready_sensor` polling Job Ledger
  - [ ] 7.4.3 Define post-PDF ops (chunk, embed, index, extract, kg)
  - [ ] 7.4.4 Wire sensor to resume post-PDF graph when gate condition met
- [ ] 7.5 Implement `@op` wrappers for each stage protocol:
  - [ ] 7.5.1 `ingest_op` calls `IngestStage.execute`
  - [ ] 7.5.2 `parse_op` calls `ParseStage.execute`
  - [ ] 7.5.3 `chunk_op` calls `ChunkStage.execute`
  - [ ] 7.5.4 `embed_op` calls `EmbedStage.execute`
  - [ ] 7.5.5 `index_op` calls `IndexStage.execute`
  - [ ] 7.5.6 `extract_op` calls `ExtractStage.execute`
  - [ ] 7.5.7 `kg_op` calls `KGStage.execute`
- [ ] 7.6 Implement Dagster `Definitions` with all jobs, resources, sensors
- [ ] 7.7 Add Dagster configuration file (`dagster.yaml`) for local dev
- [ ] 7.8 Write Dagster job tests using `dagster.execute_in_process`

## 8. Job Ledger Integration

- [ ] 8.1 Add new ledger fields:
  - [ ] 8.1.1 `pdf_downloaded: bool` (set after PDF retrieval)
  - [ ] 8.1.2 `pdf_ir_ready: bool` (set after MinerU processing)
  - [ ] 8.1.3 `current_stage: str` (e.g., "chunk", "embed")
  - [ ] 8.1.4 `pipeline_name: str` (e.g., "auto", "pdf-two-phase")
  - [ ] 8.1.5 `retry_count_per_stage: dict[str, int]` (track retries by stage)
- [ ] 8.2 Implement ledger update logic in Dagster ops
- [ ] 8.3 Implement sensor query logic for gate conditions
- [ ] 8.4 Add ledger migration script for new fields
- [ ] 8.5 Write ledger integration tests

## 9. CloudEvents & OpenLineage

- [ ] 9.1 Define `CloudEventFactory` for stage lifecycle events:
  - [ ] 9.1.1 `stage.started` with stage name, doc_id, timestamp
  - [ ] 9.1.2 `stage.completed` with duration_ms, output_count
  - [ ] 9.1.3 `stage.failed` with error message, retry_count
  - [ ] 9.1.4 `stage.retrying` with backoff_ms, attempt_number
- [ ] 9.2 Implement CloudEvents publishing to Kafka topic `orchestration.events.v1`
- [ ] 9.3 Define OpenLineage facets:
  - [ ] 9.3.1 `GPUUtilizationFacet` with gpu_memory_used, gpu_utilization_percent
  - [ ] 9.3.2 `ModelVersionFacet` with model_name, model_version
  - [ ] 9.3.3 `RetryAttemptFacet` with retry_count, backoff_strategy
- [ ] 9.4 Implement `OpenLineageEmitter` for job runs (optional, feature-flagged)
- [ ] 9.5 Add CloudEvents schema to `docs/asyncapi.yaml`
- [ ] 9.6 Write CloudEvents emission tests
- [ ] 9.7 Write OpenLineage lineage validation tests

## 10. AsyncAPI Documentation

- [ ] 10.1 Update `docs/asyncapi.yaml` with new topics:
  - [ ] 10.1.1 `orchestration.events.v1` (CloudEvents stream)
  - [ ] 10.1.2 `mineru.queue.v1` (PDF processing requests)
- [ ] 10.2 Define message schemas for each topic
- [ ] 10.3 Add operation descriptions (publish, subscribe)
- [ ] 10.4 Generate AsyncAPI HTML docs (`asyncapi generate html`)
- [ ] 10.5 Add AsyncAPI validation to CI/CD

## 11. Migration & Compatibility

- [ ] 11.1 Add feature flag `MK_USE_DAGSTER=false` (default off for Phase 1)
- [ ] 11.2 Implement `OrchestrationStrategy` enum (LEGACY, DAGSTER)
- [ ] 11.3 Create compatibility shim `submit_job` routing to Dagster when enabled
- [ ] 11.4 Maintain legacy `Orchestrator.execute_pipeline` for backward compatibility
- [ ] 11.5 Add deprecation warnings to legacy orchestration methods
- [ ] 11.6 Document migration path in `docs/guides/dagster-migration.md`
- [ ] 11.7 Create automated migration script for ledger fields
- [ ] 11.8 Write integration tests for both legacy and Dagster paths

## 12. Testing with respx

- [ ] 12.1 Create `tests/orchestration/test_dagster_jobs.py`:
  - [ ] 12.1.1 Test auto pipeline end-to-end with mocked HTTP calls (adapters, model endpoints)
  - [ ] 12.1.2 Test PDF two-phase pipeline with sensor activation
  - [ ] 12.1.3 Test resilience policies (retries, circuit breakers)
  - [ ] 12.1.4 Test GPU fail-fast behavior
- [ ] 12.2 Create respx fixtures for common API mocks:
  - [ ] 12.2.1 ClinicalTrials.gov API responses
  - [ ] 12.2.2 OpenAlex API responses
  - [ ] 12.2.3 vLLM embedding endpoint responses
  - [ ] 12.2.4 MinerU service responses
- [ ] 12.3 Write resilience behavior tests:
  - [ ] 12.3.1 Retry on transient failure (HTTP 5xx)
  - [ ] 12.3.2 Circuit breaker opens after threshold
  - [ ] 12.3.3 Rate limiting delays requests
- [ ] 12.4 Write stage contract compliance tests (verify all implementations satisfy protocols)
- [ ] 12.5 Write Haystack wrapper tests with mocked components
- [ ] 12.6 Achieve >90% test coverage for orchestration module

## 13. Observability Integration

- [ ] 13.1 Add Prometheus metrics:
  - [ ] 13.1.1 `dagster_job_duration_seconds{pipeline, status}`
  - [ ] 13.1.2 `dagster_stage_duration_seconds{stage, status}`
  - [ ] 13.1.3 `dagster_retry_attempts_total{stage, reason}`
  - [ ] 13.1.4 `dagster_circuit_breaker_state{stage, state}`
  - [ ] 13.1.5 `dagster_rate_limit_wait_seconds{stage}`
- [ ] 13.2 Add OpenTelemetry spans for Dagster ops
- [ ] 13.3 Integrate CloudEvents stream with existing Prometheus/Grafana
- [ ] 13.4 Create Grafana dashboard for Dagster job metrics
- [ ] 13.5 Add alerting rules for pipeline failures, circuit breaker opens
- [ ] 13.6 Write observability integration tests

## 14. Documentation

- [ ] 14.1 Create `docs/guides/dagster-orchestration.md`:
  - [ ] 14.1.1 Overview of DAG-based architecture
  - [ ] 14.1.2 Pipeline topology YAML reference
  - [ ] 14.1.3 Stage contract protocol reference
  - [ ] 14.1.4 Resilience policy configuration guide
  - [ ] 14.1.5 Haystack component integration guide
- [ ] 14.2 Create `docs/guides/pipeline-authoring.md`:
  - [ ] 14.2.1 How to create a new pipeline YAML
  - [ ] 14.2.2 How to implement a custom stage
  - [ ] 14.2.3 How to test stages with respx
  - [ ] 14.2.4 How to configure resilience policies
- [ ] 14.3 Create `docs/guides/pdf-two-phase-gate.md`:
  - [ ] 14.3.1 Explanation of PDF gate mechanism
  - [ ] 14.3.2 MinerU integration details
  - [ ] 14.3.3 Sensor polling logic
  - [ ] 14.3.4 Troubleshooting stalled jobs
- [ ] 14.4 Update `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md` with Dagster architecture
- [ ] 14.5 Add code examples for common tasks (submit job, check status, retry failed stage)
- [ ] 14.6 Create video walkthrough of Dagster UI for team onboarding

## 15. Deployment & Operations

- [ ] 15.1 Add Dagster to Docker Compose (`ops/docker-compose.yml`):
  - [ ] 15.1.1 `dagster-webserver` service (UI on port 3000)
  - [ ] 15.1.2 `dagster-daemon` service (sensors, schedules)
- [ ] 15.2 Add Dagster to Kubernetes deployment:
  - [ ] 15.2.1 Deployment for dagster-webserver
  - [ ] 15.2.2 Deployment for dagster-daemon
  - [ ] 15.2.3 ConfigMap for dagster.yaml
  - [ ] 15.2.4 ServiceAccount with appropriate RBAC
- [ ] 15.3 Configure Dagster storage:
  - [ ] 15.3.1 PostgreSQL for run history (or existing Neo4j)
  - [ ] 15.3.2 MinIO for logs and artifacts
- [ ] 15.4 Add health check endpoints for Dagster services
- [ ] 15.5 Update CI/CD to run Dagster tests
- [ ] 15.6 Create runbook for Dagster operations (start/stop, scaling, troubleshooting)

## 16. Production Deployment

- [ ] 16.1 Deploy to production with feature branch merge (no legacy code remains)
- [ ] 16.2 Validate auto pipeline end-to-end with ClinicalTrials.gov adapter
- [ ] 16.3 Validate PDF two-phase pipeline with PMC full-text source
- [ ] 16.4 Monitor CloudEvents stream, Prometheus metrics, error rates for 24 hours
- [ ] 16.5 Verify Dagster UI accessible, sensors running, jobs completing
- [ ] 16.6 Performance validation: throughput ≥100 docs/sec, P95 latency <500ms
- [ ] 16.7 Codebase validation: confirm ≥30% code reduction achieved
- [ ] 16.8 Emergency rollback plan: revert entire feature branch if critical issues within 48 hours
- [ ] 16.9 Conduct retrospective and document lessons learned
- [ ] 16.10 Update `CODEBASE_REDUCTION_REPORT.md` with final metrics

## 17. Dependency Management & Version Pinning

- [ ] 17.1 Create dependency compatibility matrix for Dagster + Haystack + resilience libs
- [ ] 17.2 Pin exact versions in `requirements.txt`:
  - [ ] 17.2.1 `dagster==1.5.14` (latest stable in 1.5.x series)
  - [ ] 17.2.2 `dagster-postgres==0.21.14` (matching Dagster version)
  - [ ] 17.2.3 `haystack-ai==2.0.1` (latest stable in 2.0.x series)
  - [ ] 17.2.4 `tenacity==8.2.3`
  - [ ] 17.2.5 `pybreaker==1.0.2`
  - [ ] 17.2.6 `aiolimiter==1.1.0`
  - [ ] 17.2.7 `cloudevents==1.9.0`
  - [ ] 17.2.8 `openlineage-python==1.1.0` (optional)
- [ ] 17.3 Test upgrade path from Dagster 1.5.x to 1.6.x
- [ ] 17.4 Document breaking changes in Haystack 2.0.x → 2.1.x
- [ ] 17.5 Create Dependabot config for automated security updates
- [ ] 17.6 Set up CI job to test against upcoming library versions

## 18. Error Taxonomy & Handling

- [ ] 18.1 Define Dagster-specific error classes:
  - [ ] 18.1.1 `DagsterPipelineConfigError` (invalid YAML topology)
  - [ ] 18.1.2 `DagsterStageTimeoutError` (stage exceeded timeout)
  - [ ] 18.1.3 `DagsterGateConditionError` (gate condition never met)
  - [ ] 18.1.4 `DagsterResourceUnavailableError` (GPU, Kafka, ledger unavailable)
- [ ] 18.2 Map Dagster failure modes to Job Ledger states
- [ ] 18.3 Define CloudEvent error codes for each failure type
- [ ] 18.4 Implement automatic retry for transient failures (Kafka publish, ledger update)
- [ ] 18.5 Implement dead letter queue for unrecoverable stage failures
- [ ] 18.6 Add error correlation across stage boundaries via correlation_id
- [ ] 18.7 Create error runbook with troubleshooting steps for common failures
- [ ] 18.8 Write tests for each error scenario (timeout, GPU OOM, gate timeout)

## 19. Operational Runbook

- [ ] 19.1 Create `docs/runbooks/dagster-operations.md` with sections:
  - [ ] 19.1.1 Starting/stopping Dagster services (webserver, daemon)
  - [ ] 19.1.2 Checking Dagster health (UI, API, Prometheus metrics)
  - [ ] 19.1.3 Investigating failed jobs (UI logs, CloudEvents, ledger state)
  - [ ] 19.1.4 Manually triggering post-PDF stages (sensor override)
  - [ ] 19.1.5 Draining job queue before maintenance
  - [ ] 19.1.6 Recovering from Dagster database corruption
  - [ ] 19.1.7 Emergency rollback: reverting feature branch (Git revert procedure)
- [ ] 19.2 Create troubleshooting decision tree for common issues:
  - [ ] 19.2.1 Job stuck at PDF gate → Check MinerU status, ledger state, sensor logs
  - [ ] 19.2.2 High retry rates → Check circuit breaker state, upstream API health
  - [ ] 19.2.3 CloudEvents not appearing → Check Kafka topic, consumer lag
  - [ ] 19.2.4 GPU stage failures → Check GPU availability, vLLM endpoint health
- [ ] 19.3 Define on-call escalation paths for Dagster issues
- [ ] 19.4 Create runbook for Dagster version upgrades
- [ ] 19.5 Document backup/restore procedures for Dagster PostgreSQL
- [ ] 19.6 Create runbook for Dagster UI access control (adding/removing users)

## 20. Monitoring & Alerting Specifications

- [ ] 20.1 Create Grafana dashboard `Medical_KG_Dagster_Overview.json`:
  - [ ] 20.1.1 Panel: Job throughput (jobs/second) by pipeline
  - [ ] 20.1.2 Panel: P50/P95/P99 latency per stage
  - [ ] 20.1.3 Panel: Retry rate by stage
  - [ ] 20.1.4 Panel: Circuit breaker state (open/closed/half-open)
  - [ ] 20.1.5 Panel: Sensor activity (poll rate, trigger count)
  - [ ] 20.1.6 Panel: Job Ledger state distribution
- [ ] 20.2 Define Prometheus alerting rules:
  - [ ] 20.2.1 Alert: `DagsterJobFailureRateHigh` (>5% over 5 minutes)
  - [ ] 20.2.2 Alert: `DagsterStageLatencyHigh` (P95 > SLO for 10 minutes)
  - [ ] 20.2.3 Alert: `DagsterSensorStalled` (no triggers for 5 minutes)
  - [ ] 20.2.4 Alert: `DagsterCircuitBreakerOpen` (any circuit open for >5 minutes)
  - [ ] 20.2.5 Alert: `DagsterJobQueueBacklog` (>100 jobs queued)
- [ ] 20.3 Integrate CloudEvents with existing log aggregation (Loki)
- [ ] 20.4 Create CloudEvent-based alerts for critical failures
- [ ] 20.5 Set up PagerDuty integration for critical Dagster alerts
- [ ] 20.6 Define SLO dashboards for Dagster orchestration (99.9% availability)

**Total Tasks**: 215 across 20 work streams
