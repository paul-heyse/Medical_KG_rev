## ADDED Requirements

### Requirement: Intelligent Pipeline Resolution

The gateway SHALL intelligently resolve pipeline topologies based on document metadata and characteristics, with configurable rules for different document types and datasets.

#### Scenario: PDF document pipeline selection

- **GIVEN** a client submits an ingestion request with `document_type="pdf"`
- **WHEN** the gateway processes the request
- **THEN** it resolves to the `pdf-two-phase` pipeline topology
- **AND** passes the resolved pipeline to the orchestrator for execution

#### Scenario: Non-PDF document pipeline selection

- **GIVEN** a client submits an ingestion request without PDF metadata
- **WHEN** the gateway processes the request
- **THEN** it falls back to the `auto` pipeline topology
- **AND** applies any dataset-specific resolution rules

### Requirement: Configurable Resolution Rules

The gateway SHALL support YAML-based configuration of pipeline resolution rules, allowing administrators to define custom logic for pipeline selection.

#### Scenario: Dataset-specific pipeline rules

- **GIVEN** an ingestion request for a specific dataset
- **WHEN** resolution rules include dataset-specific mappings
- **THEN** the gateway applies those rules to select the appropriate pipeline
- **AND** falls back to default rules if no specific match is found

#### Scenario: Rule configuration validation

- **GIVEN** a pipeline resolution configuration file
- **WHEN** the gateway loads the configuration
- **THEN** it validates all rules for correctness
- **AND** provides clear error messages for malformed rules
- **AND** supports hot-reload of configuration changes

## MODIFIED Requirements

### Requirement: Ingestion Job Submission

The gateway ingestion job submission process SHALL include pipeline resolution as a standard step before orchestrator execution.

#### Scenario: Enhanced job submission flow

- **GIVEN** a valid ingestion request
- **WHEN** the gateway processes the request
- **THEN** it first resolves the appropriate pipeline topology
- **AND** validates the pipeline exists and is available
- **AND** submits the job with the resolved pipeline to the orchestrator

## REMOVED Requirements

### Requirement: Hardcoded Pipeline Selection

**Reason**: Replaced by intelligent resolution system to support multiple pipeline types and dynamic selection based on document characteristics
**Migration**: Existing hardcoded pipeline selection logic is replaced by configuration-driven resolution while maintaining the same external API behavior
