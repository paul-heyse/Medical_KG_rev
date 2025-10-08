# Orchestration Pipelines Guide

This guide describes the Dagster-based orchestration stack that replaced the
legacy worker pipeline. The gateway and CLI interact with Dagster definitions
under `Medical_KG_rev.orchestration.dagster` and Haystack components under
`Medical_KG_rev.orchestration.haystack`.

## Dagster Architecture

- **Stage contracts** – `StageContext`, `ChunkStage`, `EmbedStage`, and other
  protocols live in `Medical_KG_rev.orchestration.stages.contracts`. Dagster ops
  call these protocols so stage implementations remain framework-agnostic.
- **StageFactory** – `StageFactory` resolves stage definitions from topology
  YAML files. The default factory wires Haystack chunking, embedding, and
  indexing components while falling back to lightweight stubs for unit tests.
- **Runtime module** – `Medical_KG_rev.orchestration.dagster.runtime` defines
  jobs, resources, and helper utilities (`DagsterOrchestrator`,
  `submit_to_dagster`). Jobs call the appropriate stage implementation and
  update the job ledger after each op.
- **Haystack wrappers** – `Medical_KG_rev.orchestration.haystack.components`
  adapts Haystack classes to the stage protocols. The chunker converts IR
  documents into Haystack documents, the embedder produces dense vectors (with
  optional sparse expansion), and the index writer dual writes to OpenSearch and
  FAISS.

### Stage Plugin System

- **StagePlugin base class** – `StagePlugin` in
  `Medical_KG_rev.orchestration.stages.plugins` exposes typed metadata,
  dependency declarations, lifecycle hooks (`initialize`, `cleanup`,
  `health_check`), and helpers (`create_registration`) so plugins can register
  stage builders with a few lines of code.
- **Lifecycle management** – `StagePluginManager` keeps a registry of
  `StagePluginRegistration` instances per stage type, orders them using declared
  dependencies, and exposes `unregister()`, `describe_plugins()`, and
  `check_health()` for operational tooling.
- **Dependency-aware resolution** – Registrations can depend on other plugins by
  referencing the fully qualified metadata name (e.g., `core-stage.chunk`). The
  manager topologically sorts registrations before attempting builds, so
  fallbacks are only invoked after primary providers fail.
- **Health diagnostics** – Health signals returned from `StagePlugin.health_check`
  are cached on each registration state and surfaced through
  `describe_plugins()`, simplifying automation hooks for dashboards and alerts.

## Pipeline Configuration

- **Topology YAML** – Pipelines are described in
  `config/orchestration/pipelines/*.yaml`. Each stage lists `name`, `type`,
  optional `policy`, dependencies, and a free-form `config` block. Gates define
  resume conditions, e.g., `pdf_ir_ready=true` for two-phase PDF ingestion.
- **Resilience policies** – `config/orchestration/resilience.yaml` contains
  shared retry, circuit breaker, and rate limiting definitions. The runtime
  loads these into Tenacity, PyBreaker, and aiolimiter objects.
- **Version manifest** – `config/orchestration/versions/*` tracks pipeline
  revisions. `PipelineConfigLoader` loads and caches versions to provide
  deterministic orchestration.

## Execution Flow

1. **Job submission** – The gateway builds a `StageContext` and calls
   `submit_to_dagster`. The Dagster run stores the initial state using the job
   ledger resource.
2. **Stage execution** – Each op resolves the stage implementation via
   `StageFactory`. Resilience policies wrap the execution and emit metrics on
   retries, circuit breaker state changes, and rate limiting delays.
3. **Ledger updates** – Ops record progress to the job ledger (`current_stage`,
   attempt counts, gate metadata). Sensors poll the ledger for gate conditions
   (e.g., `pdf_ir_ready=true`) and resume downstream stages.
4. **Outputs** – Stage results are added to the Dagster run state and surfaced
   to the gateway through the ledger/SSE stream. Haystack components persist
   embeddings and metadata in downstream storage systems.

## Troubleshooting

- **Stage resolution errors** – Verify the stage `type` in the topology YAML
  matches the stage types advertised by `create_stage_plugin_manager`. Unknown stage
  types raise `StageResolutionError` during job execution.
- **Resilience misconfiguration** – Check `config/orchestration/resilience.yaml`
  for required fields (attempts, backoff, circuit breaker thresholds). Invalid
  policies raise validation errors at load time.
- **Gate stalls** – Inspect the job ledger entry to confirm gate metadata is
  set (e.g., `pdf_ir_ready` for PDF pipelines). Sensors poll every ten seconds
  and record trigger counts in the ledger metadata.
- **Missing embeddings** – Ensure the embed stage resolved the Haystack
  embedder; stubs return deterministic values for test runs but do not persist
  to OpenSearch/FAISS.

## Operational Notes

- Run Dagster locally with
  `dagster dev -m Medical_KG_rev.orchestration.dagster.runtime` to access the UI
  and sensors.
- The gateway uses `StageFactory` directly for synchronous operations (chunking
  and embedding APIs) to avoid spinning up full Dagster runs.
- Dagster daemon processes handle sensors and schedules. Ensure the daemon has
  access to the same configuration volume as the webserver and gateway.
- CloudEvents and OpenLineage emission hooks live alongside the Dagster jobs
  and reuse the resilience policy loader for consistent telemetry metadata.

