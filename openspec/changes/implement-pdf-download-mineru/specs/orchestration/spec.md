## ADDED Requirements

### Requirement: PDF Download and Processing Pipeline

The orchestration system SHALL support downloading PDF files from document metadata and processing them through MinerU to generate structured document representations.

#### Scenario: PDF URL extraction and download

- **GIVEN** document metadata contains PDF URLs (e.g., from OpenAlex best_oa_location)
- **WHEN** the download stage executes
- **THEN** it extracts the PDF URL from metadata
- **AND** downloads the PDF file with proper error handling
- **AND** stores the file for subsequent MinerU processing
- **AND** updates the ledger with download completion status

#### Scenario: MinerU PDF processing

- **GIVEN** a downloaded PDF file and available GPU resources
- **WHEN** the MinerU stage executes
- **THEN** it invokes MinerU CLI with the PDF file
- **AND** parses the structured output into document IR format
- **AND** updates the ledger with `pdf_ir_ready=true`
- **AND** triggers subsequent pipeline stages

### Requirement: PDF State Management

The job ledger SHALL track PDF processing state including download status, MinerU processing completion, and error conditions.

#### Scenario: PDF processing state transitions

- **GIVEN** a document with PDF metadata
- **WHEN** processing progresses through stages
- **THEN** the ledger tracks `pdf_url`, `pdf_downloaded`, `pdf_ir_ready` states
- **AND** maintains processing history and timestamps
- **AND** supports state queries for sensor-based resumption

#### Scenario: PDF processing error states

- **GIVEN** a PDF processing failure occurs
- **WHEN** the error is recorded in the ledger
- **THEN** it includes error type, message, and retry information
- **AND** supports error recovery and state cleanup
- **AND** prevents infinite retry loops for permanent failures

## MODIFIED Requirements

### Requirement: Document Metadata Enhancement

Document models SHALL include PDF-specific metadata fields to support download and processing workflows.

#### Scenario: Enhanced document representation

- **GIVEN** a document with PDF content
- **WHEN** the document is processed through the pipeline
- **THEN** PDF URLs and processing status are included in metadata
- **AND** processing history is maintained for audit trails
- **AND** PDF-specific fields are properly validated and serialized

## REMOVED Requirements

### Requirement: Metadata-Only PDF Processing

**Reason**: Replaced by complete PDF download and MinerU processing to support full document analysis and structured output generation
**Migration**: Existing metadata-only processing continues to work while PDF processing capabilities are added for documents that include PDF URLs
