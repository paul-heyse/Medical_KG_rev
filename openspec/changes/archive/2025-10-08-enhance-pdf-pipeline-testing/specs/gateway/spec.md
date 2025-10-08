# Gateway Specification Delta — enhance-pdf-pipeline-testing

## MODIFIED Requirements

### Requirement: Gateway SHALL route PDF ingestion jobs to the dedicated MinerU pipeline

#### Scenario: Selecting pdf-two-phase for PMC datasets
- GIVEN an ingestion request targeting dataset `pmc`
- AND the Dagster orchestrator exposes the `pdf-two-phase` topology with PMC in its applicable sources
- WHEN the gateway resolves the pipeline for the incoming item
- THEN `pdf-two-phase` MUST be selected
- AND the job ledger entry MUST persist the topology version and adapter request metadata used to resume the job

#### Scenario: Falling back to pdf-two-phase for PDF document payloads
- GIVEN an ingestion request targeting a dataset without an explicit topology mapping
- AND the item metadata declares `document_type="pdf"`
- WHEN the gateway resolves the pipeline
- THEN `pdf-two-phase` MUST be selected to guarantee MinerU gating is engaged

#### Scenario: Using the default auto pipeline for non-PDF payloads
- GIVEN an ingestion request targeting an unmapped dataset
- AND the item metadata is not flagged as a PDF
- WHEN the gateway resolves the pipeline
- THEN the generic `auto` topology MUST be selected

