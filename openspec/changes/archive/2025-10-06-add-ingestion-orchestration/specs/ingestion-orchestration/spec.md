# Ingestion & Orchestration Specification

## ADDED Requirements

### Requirement: Kafka Message Broker

The system SHALL use Apache Kafka for asynchronous job orchestration with defined topics for requests, results, and mapping events.

#### Scenario: Kafka topic creation

- **WHEN** system initializes
- **THEN** topics ingest.requests.v1, ingest.results.v1, mapping.events.v1 MUST be created

#### Scenario: Message publishing

- **WHEN** ingestion job is requested
- **THEN** message MUST be published to ingest.requests.v1 with job payload

### Requirement: Job Ledger System

The system SHALL maintain a ledger tracking job state, document processing stages, and metadata for idempotency.

#### Scenario: Job state transitions

- **WHEN** job is created
- **THEN** ledger MUST track transitions queued → processing → completed/failed

#### Scenario: Idempotency checking

- **WHEN** duplicate job is submitted
- **THEN** ledger MUST detect and return existing job ID

### Requirement: Two-Phase Pipeline

The system SHALL support two-phase processing for GPU-bound operations (metadata first, then content processing).

#### Scenario: Auto-pipeline execution

- **WHEN** non-PDF document is ingested
- **THEN** pipeline MUST execute metadata → chunk → embed → index automatically

#### Scenario: Manual pipeline for PDFs

- **WHEN** PDF document is ingested
- **THEN** pipeline MUST execute metadata → PDF fetch → MinerU parse → postpdf processing

### Requirement: Multi-Adapter Chaining

The system SHALL support orchestrating multiple adapters in sequence for enrichment workflows.

#### Scenario: Literature enrichment chain

- **WHEN** OpenAlex returns open-access paper
- **THEN** orchestrator MUST chain Unpaywall → CORE → PDF download → MinerU parsing
