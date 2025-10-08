# Proposal: enhance-pdf-pipeline-testing

This implementation-focused change tightens regression coverage for the PDF ingestion pipeline by exercising the gateway routing
logic, validating ledger metadata for MinerU resumptions, and asserting the Dagster resume sensor contract. No production code is
modified beyond test scaffolding; the goal is to lock in behaviour defined by the existing topology specifications.

## Objectives

1. Confirm gateway routing surfaces the dedicated `pdf-two-phase` topology when PDF payloads arrive, and that ledger entries persist
   the correlation metadata required by the resume sensor.
2. Strengthen the PDF resume sensor test to guard the pipeline version tagging and resume stage semantics.
3. Introduce a topology regression test so the `pdf_ir_ready` gate remains correctly defined.

## Out of Scope

- Changes to the pipeline topology YAML or runtime orchestrator implementations.
- Updates to the MinerU adapter or download stages.

## Validation Strategy

- New unit tests for the gateway and topology loaders.
- Enhanced sensor regression assertions.
