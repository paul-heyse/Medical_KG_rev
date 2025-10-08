# Orchestration Pipelines Guide

This guide describes the Dagster-based orchestration stack that replaced the
legacy worker pipeline. The gateway and CLI interact with Dagster definitions
under `Medical_KG_rev.orchestration.dagster` and Haystack components under
`Medical_KG_rev.orchestration.haystack`. The pipeline now supports declarative
gate stages, two-phase execution, and rich observability for resume flows.

## Dagster Architecture

- **Stage contracts** – `StageContext`, `ChunkStage`, `EmbedStage`, and other
  protocols live in `Medical_KG_rev.orchestration.stages.contracts`. Dagster ops
  call these protocols so stage implementations remain framework-agnostic.
- **StageFactory** – `StageFactory` resolves stage definitions from topology
  YAML files. The registry also wires the ledger-aware `GateStage` that evaluates
  declarative conditions before allowing downstream phases to run.
- **Runtime module** – `Medical_KG_rev.orchestration.dagster.runtime` defines
  jobs, resources, sensors, and helper utilities (`DagsterOrchestrator`,
  `submit_to_dagster`). Jobs call the appropriate stage implementation, update
  the job ledger after each op, and emit structured gate telemetry.
- **Haystack wrappers** – `Medical_KG_rev.orchestration.haystack.components`
  adapts Haystack classes to the stage protocols. The chunker converts IR
  documents into Haystack documents, the embedder produces dense vectors (with
  optional sparse expansion), and the index writer dual writes to OpenSearch and
  FAISS.

## Pipeline Configuration

- **Topology YAML** – Pipelines are described in
  `config/orchestration/pipelines/*.yaml`. Each stage lists `name`, `type`,
  optional `policy`, dependencies, and a free-form `config` block.
- **Phases and gates** – Set `phase_order` to declare execution phases and add
  gate stages with `type: gate`. A matching entry in the `gates` section provides
  resume metadata, polling configuration, and declarative condition clauses.
- **Resilience policies** – `config/orchestration/resilience.yaml` contains
  shared retry, circuit breaker, and rate limiting definitions. The runtime
  loads these into Tenacity, PyBreaker, and aiolimiter objects.
- **Version manifest** – `config/orchestration/versions/*` tracks pipeline
  revisions. `PipelineConfigLoader` loads and caches versions to provide
  deterministic orchestration.

Example (excerpt from `pdf-two-phase.yaml`):

```yaml
phase_order:
  - pre-gate
  - gate
  - post-gate
stages:
  - name: gate_pdf_ir_ready
    type: gate
    depends_on: [download]
    phase: gate
    gate: pdf_ir_ready
gates:
  - name: pdf_ir_ready
    resume_stage: chunk
    resume_phase: post-gate
    timeout_seconds: 900
    poll_interval_seconds: 10
    conditions:
      - all:
          - field: pdf_downloaded
            operator: equals
            value: true
      - any:
          - field: pdf_ir_ready
            operator: equals
            value: true
```

## Gate Definitions and Condition Syntax

Gate clauses combine `all` (logical AND) and `any` (logical OR) lists of
predicates. Supported operators include:

- `equals` / `not_equals` – strict equality checks.
- `exists` – asserts presence (or absence) of a ledger/state attribute.
- `changed` – compares the current value to the previous evaluation.
- `in` – membership check for iterable values.

Each gate can specify optional timeout, polling interval, and retry metadata.
`resume_stage` identifies the first downstream stage to run when the gate passes
and `resume_phase` chooses the phase that should resume. Validation enforces
that gates reference existing stages, resume in later phases, and only gate
stages declare gate metadata.

## Phase-Aware Execution Flow

1. **Job submission** – The gateway builds a `StageContext` and calls
   `submit_to_dagster`. The Dagster run stores the initial state using the job
   ledger resource.
2. **Phase tracking** – The runtime partitions the pipeline into ordered phases
   using `PipelinePhasePlan`. Phase transitions emit `orchestration_phase_transitions`
   metrics and update in-memory phase markers.
3. **Gate evaluation** – When a gate stage executes, the ledger-aware
   `GateStage` polls ledger state until the declarative conditions pass, time
   out, or exhaust retries. Results are persisted to the ledger metadata and
   recorded via `orchestration_gate_*` Prometheus metrics.
4. **Downstream execution** – On success, the runtime resumes the configured
   phase, skips previously completed stages, and continues updating the ledger
   (`current_stage`, attempt counts, output metadata).
5. **Outputs** – Stage results are added to the Dagster run state and surfaced to
   the gateway through the ledger/SSE stream. Haystack components persist
   embeddings and metadata in downstream storage systems.

## Resume Sensors and Metadata

`pdf_ir_ready_sensor` inspects the job ledger for gate completions. When a gate
passes, the sensor emits a Dagster `RunRequest` containing:

- The resume stage and phase.
- Gate status, attempts, duration, and error details (if any).
- The original adapter request payload for downstream stages.

Resume runs skip previously completed phases, bind the job correlation ID, and
emit structured logs for gate evaluation success/failure.

## Gate Tooling

Use the developer tooling script to validate and debug gate configurations:

```bash
# Validate topology structure
python scripts/orchestration_gate_tool.py validate config/orchestration/pipelines/pdf-two-phase.yaml

# Visualise phases, stages, and gate clauses
python scripts/orchestration_gate_tool.py visualise config/orchestration/pipelines/pdf-two-phase.yaml --show-conditions

# Evaluate a gate definition against a ledger snapshot
python scripts/orchestration_gate_tool.py debug config/orchestration/pipelines/pdf-two-phase.yaml pdf_ir_ready --ledger ledger.json --state state.json
```

The `debug` subcommand accepts inline JSON or file paths and can persist the
mutated gate state with `--dump-gate-state` to emulate multiple evaluations.

## Troubleshooting

- **Stage resolution errors** – Verify the stage `type` in the topology YAML
  matches the keys registered in `build_default_stage_factory`. Unknown stage
  types raise `StageResolutionError` during job execution.
- **Resilience misconfiguration** – Check `config/orchestration/resilience.yaml`
  for required fields (attempts, backoff, circuit breaker thresholds). Invalid
  policies raise validation errors at load time.
- **Gate stalls** – Inspect the job ledger entry or run the debug CLI to confirm
  gate predicates and ledger fields. Prometheus counters
  (`orchestration_gate_evaluations_total`, `orchestration_gate_timeouts_total`)
  provide aggregated insight.
- **Phase mismatch during resume** – Review the gate metadata stored in the
  ledger (`gate.<name>.resume_phase`) and ensure the topology declares a valid
  downstream phase.
- **Missing embeddings** – Ensure the embed stage resolved the Haystack embedder
  and that the resume run progressed to the post-gate phase.

## Operational Notes

- Run Dagster locally with
  `dagster dev -m Medical_KG_rev.orchestration.dagster.runtime` to access the UI
  and sensors.
- The gateway uses `StageFactory` directly for synchronous operations (chunking
  and embedding APIs) to avoid spinning up full Dagster runs.
- Dagster daemon processes handle sensors and schedules. Ensure the daemon has
  access to the same configuration volume as the webserver and gateway.
- Gate metrics are exported via `metrics.py`. Scrape them alongside existing
  MinerU counters to monitor phase transitions, evaluation durations, and
  timeout rates.

