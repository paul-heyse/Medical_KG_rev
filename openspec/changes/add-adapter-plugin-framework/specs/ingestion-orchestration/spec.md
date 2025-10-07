# Ingestion Orchestration Specification Deltas

## MODIFIED Requirements

### Requirement: Adapter Selection

The orchestrator SHALL select adapters dynamically using the plugin manager by querying adapter metadata (domain, capabilities, supported identifiers). Adapter selection MUST consider tenant-specific configuration, adapter health status, and cost estimates.

#### Scenario: Adapter selection by identifier

- **GIVEN** a user requests ingestion with identifier "NCT04267848"
- **WHEN** the orchestrator queries available adapters
- **THEN** the plugin manager returns ClinicalTrialsAdapter (supports "NCT" identifiers)
- **AND** the orchestrator verifies adapter health is "healthy"
- **AND** the orchestrator checks tenant quota for estimated cost

#### Scenario: Fallback adapter on failure

- **GIVEN** primary adapter (Unpaywall) fails health check
- **WHEN** orchestrator needs to fetch PDF for DOI "10.1234/example"
- **THEN** orchestrator selects fallback adapter (CORE) with "pdf_support" capability
- **AND** job continues with fallback adapter

## ADDED Requirements

### Requirement: Plugin Manager Integration

The orchestrator MUST use the centralized **Pluggy-based** plugin manager for all adapter discovery and instantiation. Manual adapter registry imports SHALL NOT be used. The orchestrator SHALL cache plugin manager results for 5 minutes to avoid repeated entry point scanning. The plugin manager SHALL use Pluggy's `PluginManager` class to coordinate hook calls across all registered adapters.

#### Scenario: Orchestrator startup adapter discovery

- **GIVEN** application startup with multiple adapters installed
- **WHEN** orchestrator initializes
- **THEN** Pluggy plugin manager scans entry points and registers all adapters
- **AND** orchestrator caches adapter metadata
- **AND** startup completes within 2 seconds (no expensive operations)

#### Scenario: Adapter version tracking

- **GIVEN** an ingestion job completes successfully
- **WHEN** the job is recorded in the ledger
- **THEN** the ledger stores adapter name and version
- **AND** version information is available for debugging and auditing

### Requirement: Dynamic Pipeline Construction

The orchestrator SHALL support dynamic pipeline construction based on adapter capabilities and data source requirements. Multi-domain ingestion jobs SHALL automatically route documents to appropriate domain-specific processing stages.

#### Scenario: Multi-domain ingestion

- **GIVEN** an ingestion job with biomedical and financial documents
- **WHEN** the orchestrator constructs the pipeline
- **THEN** biomedical documents route to medical validation stage
- **AND** financial documents route to XBRL validation stage
- **AND** both domains share common chunking and embedding stages
