# Implementation Tasks: PDF Download and MinerU Integration

## 1. PDF URL Extraction Enhancement

### 1.1 OpenAlex Adapter Enhancement

- [ ] 1.1.1 Update OpenAlex adapter to extract `best_oa_location.url_for_pdf` from metadata
- [ ] 1.1.2 Add PDF URL validation and accessibility checking
- [ ] 1.1.3 Include PDF metadata (file size, content type) in document representation
- [ ] 1.1.4 Handle cases where no PDF URL is available

### 1.2 Document Metadata Schema Updates

- [ ] 1.2.1 Extend `Document` model to include PDF metadata fields
- [ ] 1.2.2 Add `pdf_url`, `pdf_size`, `pdf_content_type` fields
- [ ] 1.2.3 Update document serialization for PDF metadata
- [ ] 1.2.4 Add validation for PDF metadata fields

## 2. Download Stage Implementation

### 2.1 PDF Download Service

- [ ] 2.1.1 Create `PdfDownloadService` for URL-based PDF acquisition
- [ ] 2.1.2 Implement download with retry logic and progress tracking
- [ ] 2.1.3 Add file validation (size, content type, corruption checks)
- [ ] 2.1.4 Support resumable downloads for large files

### 2.2 Download Stage Integration

- [ ] 2.2.1 Create `DownloadStage` class implementing orchestration protocol
- [ ] 2.2.2 Extract PDF URLs from document metadata
- [ ] 2.2.3 Download PDFs to temporary storage location
- [ ] 2.2.4 Update ledger with `pdf_downloaded=true` and file location
- [ ] 2.2.5 Handle download failures and retry scenarios

### 2.3 Storage Integration

- [ ] 2.3.1 Integrate with MinIO for PDF file storage
- [ ] 2.3.2 Implement file cleanup for failed downloads
- [ ] 2.3.3 Add file access logging for audit trails
- [ ] 2.3.4 Support multiple storage backends (local, S3, etc.)

## 3. MinerU Integration

### 3.1 MinerU Processing Service

- [ ] 3.1.1 Create `MineruProcessingService` for PDF-to-IR conversion
- [ ] 3.1.2 Implement MinerU CLI invocation with proper parameters
- [ ] 3.1.3 Handle MinerU output parsing and validation
- [ ] 3.1.4 Add processing progress tracking and timeout handling

### 3.2 MinerU Stage Integration

- [ ] 3.2.1 Create `MineruStage` class implementing orchestration protocol
- [ ] 3.2.2 Trigger MinerU processing on downloaded PDFs
- [ ] 3.2.3 Parse MinerU output into structured document format
- [ ] 3.2.4 Update ledger with `pdf_ir_ready=true` on successful processing
- [ ] 3.2.5 Handle MinerU failures and partial processing

### 3.3 GPU Resource Management

- [ ] 3.3.1 Integrate with GPU service manager for MinerU resource allocation
- [ ] 3.3.2 Implement concurrent processing limits based on GPU memory
- [ ] 3.3.3 Add GPU utilization monitoring and metrics
- [ ] 3.3.4 Handle GPU unavailability with appropriate fallbacks

## 4. Ledger State Management

### 4.1 PDF State Tracking

- [ ] 4.1.1 Extend ledger schema to include PDF processing fields
- [ ] 4.1.2 Add `pdf_url`, `pdf_downloaded`, `pdf_ir_ready` fields
- [ ] 4.1.3 Implement state transition validation and consistency checks
- [ ] 4.1.4 Add PDF processing history and timestamps

### 4.2 State Transition Logic

- [ ] 4.2.1 Create PDF state machine with valid transitions
- [ ] 4.2.2 Implement state transition validation
- [ ] 4.2.3 Add automatic state cleanup for failed operations
- [ ] 4.2.4 Support state queries for sensor-based resumption

## 5. Error Handling and Recovery

### 5.1 Download Error Handling

- [ ] 5.1.1 Implement retry logic for transient download failures
- [ ] 5.1.2 Handle network timeouts and connection errors
- [ ] 5.1.3 Add circuit breaker for repeatedly failing URLs
- [ ] 5.1.4 Log download attempts and failures for debugging

### 5.2 MinerU Error Handling

- [ ] 5.2.1 Handle MinerU CLI failures and exit codes
- [ ] 5.2.2 Implement timeout handling for long-running processing
- [ ] 5.2.3 Add GPU memory error detection and recovery
- [ ] 5.2.4 Support partial processing recovery where possible

### 5.3 Comprehensive Error Recovery

- [ ] 5.3.1 Implement rollback for failed PDF operations
- [ ] 5.3.2 Add dead letter queue handling for unrecoverable errors
- [ ] 5.3.3 Create error classification system (retryable vs permanent)
- [ ] 5.3.4 Implement exponential backoff for retryable failures

## 6. Testing and Validation

### 6.1 Unit Tests for PDF Processing

- [ ] 6.1.1 Test PDF URL extraction from various metadata formats
- [ ] 6.1.2 Test download service with mocked HTTP responses
- [ ] 6.1.3 Test MinerU service integration with mock CLI
- [ ] 6.1.4 Test ledger state management and transitions

### 6.2 Integration Tests

- [ ] 6.2.1 Test complete PDF download and MinerU processing flow
- [ ] 6.2.2 Test error scenarios and recovery mechanisms
- [ ] 6.2.3 Test state transitions and sensor triggering
- [ ] 6.2.4 Test concurrent PDF processing limits

### 6.3 End-to-End Validation

- [ ] 6.3.1 Test full PDF pipeline from ingestion to completion
- [ ] 6.3.2 Validate MinerU output quality and completeness
- [ ] 6.3.3 Test performance under load with multiple PDFs
- [ ] 6.3.4 Test error recovery and state cleanup

## 7. Monitoring and Observability

### 7.1 PDF Processing Metrics

- [ ] 7.1.1 Track download success/failure rates by source
- [ ] 7.1.2 Monitor MinerU processing times and throughput
- [ ] 7.1.3 Track GPU utilization during PDF processing
- [ ] 7.1.4 Add PDF file size and processing time correlations

### 7.2 Enhanced Logging and Tracing

- [ ] 7.2.1 Add structured logging for PDF download operations
- [ ] 7.2.2 Include correlation IDs for download → processing → indexing flow
- [ ] 7.2.3 Log MinerU processing details and output validation
- [ ] 7.2.4 Add distributed tracing for PDF pipeline execution

### 7.3 Alerting and Monitoring

- [ ] 7.3.1 Add alerts for download failure rate thresholds
- [ ] 7.3.2 Monitor MinerU service health and responsiveness
- [ ] 7.3.3 Track PDF processing queue depth and backlog
- [ ] 7.3.4 Implement PDF processing SLA monitoring

**Total Tasks**: 60 across 7 work streams

**Risk Assessment:**

- **High Risk**: PDF processing involves external services and file I/O which can fail
- **Medium Risk**: GPU resource management complexity could affect reliability

**Rollback Plan**: If critical issues arise, disable PDF processing while keeping metadata-only ingestion functional.
