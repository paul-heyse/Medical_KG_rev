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

## Pipeline Configuration

- **Topology YAML** – Pipelines are described in
  `config/orchestration/pipelines/*.yaml`. Each stage lists `name`, `type`,
  optional `policy`, dependencies, and a free-form `config` block.
  Gate stages add a `gate` reference that points to a named entry in the
  `gates:` section. Gate definitions declare the ledger conditions, timeout,
  polling interval, and resume stage for two-phase execution.
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
   attempt counts, gate metadata). Gate evaluations record status, attempts,
   and phase readiness so sensors can resume downstream stages deterministically.
4. **Outputs** – Stage results are added to the Dagster run state and surfaced
   to the gateway through the ledger/SSE stream. Haystack components persist
   embeddings and metadata in downstream storage systems.

## Gate-Aware Execution

- **Condition syntax** – Each `GateDefinition` contains one or more clauses
  under `condition.clauses`. Supported operators are `equals`, `exists`, and
  `changed`. Clauses are combined with `condition.mode` (`all`/`any`). Field
  paths support root attributes such as `pdf_ir_ready`, `status`, or nested
  `metadata.*` entries stored in the job ledger.
- **Timeout handling** – Gates poll the ledger until the condition is satisfied
  or the configured `timeout_seconds` elapses. Timeouts raise
  `GateConditionError`, update the ledger with gate status, and emit metrics for
  observability. The Dagster run completes without executing post-gate stages so
  sensors can resume later.
- **Phase tracking** – Stage definitions are assigned numeric phases during
  topology validation. Gate stages unlock the next phase by setting
  `phase_index` and `phase_ready` in the run state and ledger. Resume runs
  should set `context.phase` to the unlocked phase (for example, `phase-2`) so
  pre-gate stages are skipped automatically.
- **Metrics and logging** – Gate evaluation outcomes increment
  `orchestration_gate_evaluation_total` with the gate name and status. Phase
  transitions emit `orchestration_phase_transition_total`. Structured log lines
  prefixed with `dagster.stage.gate_*` describe skip reasons, failures, and
  successful unlocks.

## Sensors and Resumption

- The `pdf_ir_ready_sensor` watches ledger entries for `pdf_ir_ready=true` and
  `status=processing`. When triggered it creates a Dagster run with
  `context.phase=phase-2`, forwards the original adapter payload, and tags the
  resume stage/phase for observability.
- Resume runs inherit ledger metadata (correlation ID, payload, gate status)
  so monitoring dashboards can tie both phases together. The orchestrator only
  marks a job `completed` when the final phase finishes with `phase_ready=true`.
- Gate metadata lives under `metadata["gate.<name>.*"]` in the ledger. Use this
  to debug stalled jobs, confirm resume stages, and correlate gate attempts.

## Troubleshooting

- **Stage resolution errors** – Verify the stage `type` in the topology YAML
  matches the keys registered in `build_default_stage_factory`. Unknown stage
  types raise `StageResolutionError` during job execution.
- **Resilience misconfiguration** – Check `config/orchestration/resilience.yaml`
  for required fields (attempts, backoff, circuit breaker thresholds). Invalid
  policies raise validation errors at load time.
- **Gate stalls** – Inspect the job ledger entry to confirm gate metadata is
  set (for example, `metadata["gate.pdf_ir_ready.status"]`). The ledger records
  attempts, reasons, and last values per clause. Sensors poll every ten seconds
  and resume the pipeline automatically once `phase_ready` flips to `true`.
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

