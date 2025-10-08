# Connect Ledger State Management — Orchestration

## MODIFIED Requirements

### Requirement: Stage lifecycle events SHALL update the job ledger through a dedicated state manager.
- **Previous Behavior**: Dagster stage ops called the in-memory ledger directly, updating retry counters and metadata in-line inside each stage wrapper.
- **New Behavior**: Stage ops MUST call a shared `LedgerStateManager` that records retries, marks stage failures, and enriches metadata for successful stages.

#### Scenario: Stage success records attempts, counts, duration, and completion timestamp
- **GIVEN** a job ledger entry for `job-123`
- **AND** the orchestrator executes stage `chunk`
- **WHEN** the stage succeeds on the second attempt producing 5 outputs and taking 1.2 seconds
- **THEN** the ledger metadata SHALL include `stage.chunk.attempts=2`
- **AND** `stage.chunk.output_count=5`
- **AND** `stage.chunk.duration_ms=1200`
- **AND** `stage.chunk.completed_at` SHALL be populated with an ISO timestamp.

#### Scenario: Stage failure records reason centrally
- **GIVEN** a job ledger entry for `job-123`
- **WHEN** the orchestrator raises an error during stage `embed`
- **THEN** the state manager SHALL mark the ledger entry `status="failed"`, `stage="embed"`, and `error_reason` set to the exception text.

### Requirement: Dagster run submission SHALL prepare ledger metadata before execution.
- **Previous Behavior**: `submit()` attempted to update ledger metadata inline and did not normalise context payloads.
- **New Behavior**: A ledger state manager SHALL populate pipeline name, pipeline version, context metadata, adapter request, payload snapshot, and mark the job status as `processing` with stage `bootstrap` before executing the Dagster job.

#### Scenario: Run preparation persists context and adapter request metadata
- **GIVEN** a queued ledger entry `job-789`
- **WHEN** `submit()` is invoked for pipeline `pdf-two-phase` version `v2` with context metadata `{ "source": "tests" }`
- **AND** adapter request `{ "adapter": "mineru" }`
- **AND** payload `{ "seed": 7 }`
- **THEN** the ledger entry SHALL have `pipeline_name="pdf-two-phase"`, `status="processing"`, and `stage="bootstrap"`
- **AND** metadata SHALL contain keys `pipeline_version`, `context`, `adapter_request`, and `payload` populated with the submitted values.

### Requirement: Job attempts and retries SHALL be tracked via the state manager.
- **Previous Behavior**: Retry counters were incremented directly against `JobLedger` inside the retry hook.
- **New Behavior**: Retry hooks MUST call the state manager so that retry metrics remain centralised and missing-ledger scenarios are handled gracefully.

#### Scenario: Retry increments per-stage counter
- **GIVEN** a ledger entry with `retry_count_per_stage["chunk"] = 1`
- **WHEN** the stage retry hook fires for stage `chunk`
- **THEN** the state manager SHALL increment `retry_count_per_stage["chunk"]` to `2`
- **AND** keep the job status as `processing`.
