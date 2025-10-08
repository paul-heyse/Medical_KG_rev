# Orchestration Capability: Spec Delta

## MODIFIED Requirements

### Requirement: PDF Two-Phase Pipeline with Manual Gate (Modified)

The PDF two-phase pipeline SHALL enforce an explicit manual gate (`postpdf-start`) after MinerU processing, replacing any automatic progression to chunking/embedding.

**Previous Behavior**: After MinerU completed, the pipeline automatically continued to chunking/embedding (or behavior was ambiguous).

**New Behavior**: After MinerU completes and ledger reaches `pdf_ir_ready=true`, the pipeline SHALL HALT until explicit `postpdf-start` trigger.

#### Scenario: PDF pipeline halts at MinerU gate

- **GIVEN** a PDF ingestion job with MinerU processing complete
- **WHEN** the ledger is updated to `pdf_ir_ready=true`
- **THEN** the orchestrator does NOT automatically schedule chunking
- **AND** the job remains in state `pdf_ir_ready` until manual intervention
- **AND** a CloudEvent `mineru.gate.waiting` is emitted

#### Scenario: postpdf-start trigger resumes pipeline

- **GIVEN** a job in state `pdf_ir_ready=true`
- **WHEN** an operator calls `POST /v1/jobs/{job_id}/postpdf-start`
- **THEN** the orchestrator validates the ledger state is `pdf_ir_ready`
- **AND** schedules chunking stage with context from MinerU IR
- **AND** updates ledger to `postpdf_start_triggered=true`
- **AND** emits CloudEvent `postpdf.start.triggered`

#### Scenario: Dagster sensor polls for pdf_ir_ready

- **GIVEN** the Dagster `pdf_ir_ready_sensor` configured with poll_interval=30s
- **WHEN** the sensor polls the Job Ledger
- **THEN** the sensor detects jobs with `pdf_ir_ready=true` and `postpdf_start_triggered=false`
- **AND** for jobs waiting >5 minutes, the sensor triggers `postpdf-start` automatically
- **AND** the sensor records `trigger_source="auto_sensor"` in ledger metadata

#### Scenario: postpdf-start validation prevents invalid triggers

- **GIVEN** a job in state `chunking` (already past the gate)
- **WHEN** an operator attempts `POST /v1/jobs/{job_id}/postpdf-start`
- **THEN** the system raises `InvalidStateError("Job {job_id} is not in pdf_ir_ready state (current: chunking)")`
- **AND** the ledger remains unchanged
- **AND** HTTP 400 Bad Request is returned

---

### Requirement: Job Ledger Schema for PDF Gate (Modified)

The Job Ledger schema SHALL include fields tracking the PDF two-phase gate: `pdf_downloaded`, `pdf_ir_ready`, `postpdf_start_triggered`, `mineru_bbox_map`.

**Previous Behavior**: Ledger only tracked high-level states (queued, processing, completed, failed).

**New Behavior**: Ledger SHALL explicitly track PDF gate progression with boolean flags and MinerU metadata.

#### Scenario: Ledger tracks PDF gate progression

- **GIVEN** a PDF ingestion job created
- **WHEN** the PDF is downloaded from Unpaywall
- **THEN** ledger is updated: `pdf_downloaded=true, pdf_ir_ready=false, postpdf_start_triggered=false`
- **WHEN** MinerU completes successfully
- **THEN** ledger is updated: `pdf_ir_ready=true, mineru_bbox_map={...}`
- **WHEN** `postpdf-start` is triggered
- **THEN** ledger is updated: `postpdf_start_triggered=true, postpdf_triggered_at="2025-10-07T14:30:00Z"`

#### Scenario: Ledger MinerU bbox map for provenance

- **GIVEN** MinerU processed a 10-page PDF
- **WHEN** the ledger is updated after MinerU success
- **THEN** `mineru_bbox_map` includes page/bbox mappings for all blocks
- **AND** downstream chunking can use this map for `page_bbox` in chunks
- **AND** the map is serialized as JSON in the ledger

#### Scenario: Ledger migration adds new fields with defaults

- **GIVEN** 1000 existing job ledger entries without PDF gate fields
- **WHEN** the ledger migration script runs
- **THEN** all entries have `pdf_downloaded=false, pdf_ir_ready=false, postpdf_start_triggered=false` added
- **AND** entries with `status="completed"` from PDF sources infer `pdf_downloaded=true, pdf_ir_ready=true, postpdf_start_triggered=true`
