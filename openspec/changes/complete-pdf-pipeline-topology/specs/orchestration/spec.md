## ADDED Requirements

### Requirement: Complete PDF Two-Phase Pipeline

The orchestration system SHALL support a complete PDF two-phase pipeline topology that includes all necessary stages for PDF document processing with proper gate-based execution control.

#### Scenario: PDF pipeline with download stage

- **GIVEN** a PDF document ingestion request
- **WHEN** the pdf-two-phase pipeline executes
- **THEN** it includes a download stage that acquires the PDF file
- **AND** updates the ledger with `pdf_downloaded=true`
- **AND** triggers MinerU processing to set `pdf_ir_ready=true`

#### Scenario: Gate-controlled two-phase execution

- **GIVEN** a PDF pipeline with a gate_pdf_ir_ready gate
- **WHEN** the gate condition is evaluated
- **THEN** it checks if `pdf_ir_ready=true` in the ledger
- **AND** if true, allows execution to proceed to post-gate stages
- **AND** if false, halts execution until condition is met

### Requirement: Download Stage Implementation

The orchestration system SHALL include a download stage that can acquire PDF files from URLs and update document state accordingly.

#### Scenario: PDF URL extraction and download

- **GIVEN** document metadata contains PDF URLs (e.g., from OpenAlex best_oa_location)
- **WHEN** the download stage executes
- **THEN** it extracts and downloads the PDF file
- **AND** stores the file for subsequent MinerU processing
- **AND** updates the ledger with download completion status

#### Scenario: Download error handling

- **GIVEN** a download stage encounters an error
- **WHEN** the download fails or URL is invalid
- **THEN** it logs the error and updates ledger with failure status
- **AND** allows pipeline execution to continue or fail based on configuration
- **AND** provides retry capability for transient failures

### Requirement: Gate Stage Implementation

The orchestration system SHALL support gate stages that control pipeline execution flow based on external conditions.

#### Scenario: Ledger-based gate conditions

- **GIVEN** a gate stage with ledger field conditions
- **WHEN** the gate is evaluated
- **THEN** it queries the ledger for the specified conditions
- **AND** raises `GateConditionError` if conditions are not met
- **AND** allows execution to proceed when conditions are satisfied

#### Scenario: Gate timeout handling

- **GIVEN** a gate stage with a timeout configuration
- **WHEN** the gate condition is not met within the timeout period
- **THEN** it raises a timeout error
- **AND** logs the timeout for monitoring and debugging
- **AND** allows configuration of timeout behavior (fail vs retry)

## MODIFIED Requirements

### Requirement: Pipeline Topology Schema

The pipeline topology schema SHALL support gate definitions and download stages in addition to existing stage types.

#### Scenario: Enhanced pipeline configuration

- **GIVEN** a pipeline topology YAML file
- **WHEN** it includes gate and download stage definitions
- **THEN** the system validates the gate conditions and stage configurations
- **AND** builds the appropriate Dagster execution graph
- **AND** handles two-phase execution with gate-controlled flow

## REMOVED Requirements

### Requirement: Incomplete PDF Pipeline Support

**Reason**: Replaced by complete PDF pipeline implementation with proper gate handling and download capabilities
**Migration**: Existing partial PDF pipeline configurations will be enhanced with missing stage definitions and gate logic while maintaining compatibility with existing execution patterns
