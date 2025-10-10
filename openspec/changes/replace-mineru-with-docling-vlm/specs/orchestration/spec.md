## ADDED Requirements

### Requirement: Docling VLM PDF Processing Pipeline

The system SHALL provide a VLM-based PDF processing pipeline using Docling with Gemma3 12B model as an alternative to the MinerU OCR pipeline.

#### Scenario: PDF processing with Docling VLM

- **WHEN** a PDF document is submitted for processing
- **AND** the Docling VLM feature flag is enabled
- **THEN** the system SHALL use Docling[vlm] with Gemma3 12B for document analysis
- **AND** SHALL extract structured content including text, tables, and figures
- **AND** SHALL maintain backward compatibility with existing document formats

#### Scenario: VLM processing fallback handling

- **WHEN** Docling VLM processing fails due to model limitations
- **THEN** the system SHALL provide clear error reporting
- **AND** SHALL maintain graceful degradation without system failure
- **AND** SHALL log detailed error information for debugging

#### Scenario: GPU resource management for VLM

- **WHEN** multiple PDF processing requests arrive simultaneously
- **THEN** the system SHALL manage GPU memory allocation for Gemma3 12B
- **AND** SHALL implement request queuing to prevent resource exhaustion
- **AND** SHALL provide monitoring for GPU utilization and memory usage

## MODIFIED Requirements

### Requirement: PDF Processing Pipeline Stages

The PDF processing pipeline stages SHALL support both MinerU and Docling VLM processing modes through feature flag configuration.

#### Scenario: Feature flag controlled processing mode

- **WHEN** the Docling VLM feature flag is enabled
- **THEN** PDF download and gate stages SHALL use Docling readiness checks
- **AND** SHALL bypass MinerU-specific processing requirements
- **AND** SHALL maintain the same external API interface

#### Scenario: Backward compatibility during migration

- **WHEN** processing existing documents created with MinerU
- **THEN** the system SHALL continue to support document access and retrieval
- **AND** SHALL not require document reprocessing for compatibility
- **AND** SHALL maintain provenance tracking for both processing methods

## REMOVED Requirements

### Requirement: MinerU-Specific Processing Dependencies

**Reason**: Replaced by Docling VLM processing
**Migration**: Existing MinerU-processed documents remain accessible; new documents use Docling

The system no longer requires MinerU CLI and vLLM server for PDF processing when Docling VLM mode is active.
