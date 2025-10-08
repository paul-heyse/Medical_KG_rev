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
  optional `policy`, dependencies, and a free-form `config` block. Gates define
  resume conditions, e.g., `pdf_ir_ready=true` for two-phase PDF ingestion.
- **Resilience policies** – `config/orchestration/resilience.yaml` contains
  shared retry, circuit breaker, and rate limiting definitions. The runtime
  loads these into Tenacity, PyBreaker, and aiolimiter objects.
- **Version manifest** – `config/orchestration/versions/*` tracks pipeline
  revisions. `PipelineConfigLoader` loads and caches versions to provide
  deterministic orchestration.

## Gate-Aware Two-Phase Execution

- **Phases** – Every stage is tagged with a `phase` (`pre-gate`, `gate`,
  `post-gate`). Runtime transitions are emitted via the
  `orchestration_phase_transitions_total` metric and persisted in the job
  ledger for observability.
- **Gate definitions** – Gate metadata lives alongside the stage topology. Each
  gate specifies the controlling `stage`, `resume_stage`, timeout, retry
  behaviour, and a list of ledger predicates (e.g., `pdf_ir_ready equals true`).
  Gate evaluation outcomes are emitted to
  `orchestration_gate_evaluations_total`, timeouts to
  `orchestration_gate_timeouts_total`.
- **Runtime behaviour** – Gate stages poll the ledger until their clauses are
  satisfied. Successful gates capture a structured result in the run state and
  ledger (`metadata.gate.<name>`), which the resume sensor uses to skip
  download work and seed the correct downstream phase.
- **Tooling** – Run `medkg-gates pdf-two-phase` to render a human-readable gate
  report (or `--json` for automation). This CLI is backed by
  `Medical_KG_rev.orchestration.tools.gate_debugger` and validates gate
  definitions against the live topology cache.

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
  matches the keys registered in `build_default_stage_factory`. Unknown stage
  types raise `StageResolutionError` during job execution.
- **Resilience misconfiguration** – Check `config/orchestration/resilience.yaml`
  for required fields (attempts, backoff, circuit breaker thresholds). Invalid
  policies raise validation errors at load time.
- **Gate stalls** – Use `medkg-gates <pipeline>` to confirm the expected resume
  stage and predicates. At runtime, inspect `metadata.gate.<gate>` within the
  job ledger and the `orchestration_gate_*` metrics to determine whether the
  gate failed its clauses or timed out waiting for ledger updates.
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

