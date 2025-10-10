## ADDED Requirements

### Requirement: Docling VLM Service Integration

The system SHALL provide a DoclingVLMService that integrates Docling's vision-language model capabilities with the Gemma3 12B model for PDF document processing.

#### Scenario: VLM service initialization

- **WHEN** the DoclingVLMService starts up
- **THEN** it SHALL verify Gemma3 12B model availability
- **AND** SHALL check GPU resource requirements (~24GB VRAM)
- **AND** SHALL initialize model warm-up procedures
- **AND** SHALL register health check endpoints

#### Scenario: PDF processing with VLM

- **WHEN** a PDF document is submitted for VLM processing
- **THEN** the service SHALL use Docling[vlm] with Gemma3 12B
- **AND** SHALL extract structured content including text, tables, and figures
- **AND** SHALL return results in the same format as MinerU processing
- **AND** SHALL include provenance information for VLM processing

#### Scenario: Batch processing with VLM

- **WHEN** multiple PDF documents are submitted for processing
- **THEN** the service SHALL implement efficient batching for Gemma3 12B
- **AND** SHALL manage GPU memory allocation across batches
- **AND** SHALL provide progress tracking for batch operations
- **AND** SHALL handle partial failures gracefully

#### Scenario: Error handling and recovery

- **WHEN** VLM processing encounters errors
- **THEN** the service SHALL provide detailed error classification
- **AND** SHALL implement retry logic for transient failures
- **AND** SHALL maintain circuit breaker patterns for persistent issues
- **AND** SHALL log comprehensive error information for debugging

## MODIFIED Requirements

### Requirement: PDF Processing Service Interface

The PDF processing service interface SHALL support both MinerU and Docling VLM backends through configuration.

#### Scenario: Service backend selection

- **WHEN** the service receives a PDF processing request
- **THEN** it SHALL determine the appropriate backend based on feature flags
- **AND** SHALL route requests to either MinerU or Docling VLM
- **AND** SHALL maintain consistent response formats across backends
- **AND** SHALL provide backend identification in response metadata

#### Scenario: Performance monitoring across backends

- **WHEN** processing requests with either backend
- **THEN** the service SHALL collect performance metrics
- **AND** SHALL compare processing times and accuracy
- **AND** SHALL provide monitoring dashboards for backend comparison
- **AND** SHALL alert on significant performance deviations

## REMOVED Requirements

### Requirement: MinerU Service Dependency

**Reason**: Replaced by Docling VLM service
**Migration**: DoclingVLMService provides equivalent functionality with improved accuracy

The system no longer requires the MinerU service for PDF processing when Docling VLM mode is active.
