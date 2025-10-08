# Implementation Tasks: Gateway Pipeline Resolution

## 1. Pipeline Resolution Architecture

### 1.1 Create Resolution Configuration Schema

- [ ] 1.1.1 Define `PipelineResolutionConfig` Pydantic model for YAML parsing
- [ ] 1.1.2 Define `ResolutionRule` with conditions and pipeline mapping
- [ ] 1.1.3 Define `DocumentMatcher` for PDF detection and metadata matching
- [ ] 1.1.4 Add schema validation for resolution rules

### 1.2 Implement Resolution Engine

- [ ] 1.2.1 Create `PipelineResolver` class with rule evaluation logic
- [ ] 1.2.2 Implement PDF detection: `document_type="pdf"` condition
- [ ] 1.2.3 Add fallback pipeline selection for unmatched documents
- [ ] 1.2.4 Add confidence scoring for resolution decisions

### 1.3 Configuration File Structure

- [ ] 1.3.1 Create `config/gateway/pipeline-resolution.yaml` with rules:
  - [ ] 1.3.1.1 PDF document rule → `pdf-two-phase` pipeline
  - [ ] 1.3.1.2 Default rule → `auto` pipeline for non-PDF
  - [ ] 1.3.1.3 Future extensibility for dataset-specific rules
- [ ] 1.3.2 Add configuration loading and validation
- [ ] 1.3.3 Add hot-reload capability for resolution rules

## 2. Gateway Integration

### 2.1 Update Gateway Services

- [ ] 2.1.1 Modify `GatewayService.submit_ingestion_job()` to use `PipelineResolver`
- [ ] 2.1.2 Add pipeline resolution before orchestrator submission
- [ ] 2.1.3 Enhance error handling for resolution failures
- [ ] 2.1.4 Add logging for resolution decisions

### 2.2 Update Ingestion Request Processing

- [ ] 2.2.1 Extract document metadata from ingestion requests
- [ ] 2.2.2 Apply resolution rules to determine pipeline topology
- [ ] 2.2.3 Validate pipeline exists before submission
- [ ] 2.2.4 Pass resolved pipeline name to orchestrator

## 3. Testing and Validation

### 3.1 Unit Tests

- [ ] 3.1.1 Test `PipelineResolver` with various document scenarios
- [ ] 3.1.2 Test PDF detection logic with different metadata formats
- [ ] 3.1.3 Test fallback behavior for unmatched documents
- [ ] 3.1.4 Test configuration validation and error handling

### 3.2 Integration Tests

- [ ] 3.2.1 Test end-to-end pipeline resolution via gateway API
- [ ] 3.2.2 Test PDF document triggers pdf-two-phase pipeline
- [ ] 3.2.3 Test non-PDF documents use auto pipeline
- [ ] 3.2.4 Test error scenarios and edge cases

### 3.3 Configuration Tests

- [ ] 3.3.1 Test resolution rule loading and validation
- [ ] 3.3.2 Test configuration hot-reload functionality
- [ ] 3.3.3 Test malformed configuration handling

## 4. Documentation and Examples

### 4.1 Developer Documentation

- [ ] 4.1.1 Update `docs/guides/gateway-integration.md` with pipeline resolution
- [ ] 4.1.2 Document resolution rule configuration format
- [ ] 4.1.3 Add troubleshooting guide for resolution failures

### 4.2 Code Examples

- [ ] 4.2.1 Create example resolution rule configurations
- [ ] 4.2.2 Add integration test examples for different document types
- [ ] 4.2.3 Document API usage patterns for pipeline resolution

## 5. Observability and Monitoring

### 5.1 Metrics Collection

- [ ] 5.1.1 Add metrics for pipeline resolution decisions
- [ ] 5.1.2 Track resolution rule hit rates
- [ ] 5.1.3 Monitor resolution failure rates
- [ ] 5.1.4 Add pipeline selection latency metrics

### 5.2 Logging Enhancement

- [ ] 5.2.1 Add structured logging for resolution decisions
- [ ] 5.2.2 Include document metadata in resolution logs
- [ ] 5.2.3 Add correlation IDs for resolution tracing
- [ ] 5.2.4 Log resolution rule evaluation process

**Total Tasks**: 28 across 5 work streams

**Risk Assessment:**

- **Low Risk**: Changes are additive and maintain backward compatibility
- **Medium Risk**: Pipeline resolution logic could affect existing workflows if misconfigured

**Rollback Plan**: If issues arise, revert to hardcoded pipeline selection while maintaining new configuration structure.
