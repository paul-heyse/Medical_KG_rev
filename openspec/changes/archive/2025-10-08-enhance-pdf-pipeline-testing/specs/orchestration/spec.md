# Orchestration Specification Delta — enhance-pdf-pipeline-testing

## MODIFIED Requirements

### Requirement: Dagster SHALL expose the pdf_ir_ready resume contract for MinerU jobs

#### Scenario: Resuming pdf-two-phase after MinerU signals readiness
- GIVEN a ledger entry for pipeline `pdf-two-phase` marked `pdf_ir_ready=true`
- WHEN the pdf_ir_ready_sensor evaluates the ledger
- THEN it MUST emit a `RunRequest` carrying the stored pipeline version, correlation payload, and a `medical_kg.resume_stage="chunk"` tag

#### Scenario: Guarding the pdf-two-phase topology gate configuration
- GIVEN the pipeline loader parses `config/orchestration/pipelines/pdf-two-phase.yaml`
- WHEN the topology is validated
- THEN it MUST include a gate named `pdf_ir_ready` that resumes stage `chunk`
- AND the gate condition MUST watch the `pdf_ir_ready` ledger field with timeout `900` seconds and poll interval `10.0` seconds
- AND the topology stages MUST include `download` and `gate_pdf_ir_ready` to enforce MinerU gating semantics

