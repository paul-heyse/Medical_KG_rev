# Implementation Tasks: Complete PDF Pipeline Topology

## 1. Pipeline Topology Enhancement

### 1.1 Complete PDF Two-Phase Pipeline Configuration

- [ ] 1.1.1 Update `config/orchestration/pipelines/pdf-two-phase.yaml`:
  - [ ] 1.1.1.1 Add adapter configuration for ingest stage (OpenAlex for metadata)
  - [ ] 1.1.1.2 Define download stage configuration (URL extraction and download logic)
  - [ ] 1.1.1.3 Configure gate_pdf_ir_ready with proper condition logic
  - [ ] 1.1.1.4 Add stage dependencies reflecting two-phase execution
- [ ] 1.1.2 Add pipeline metadata (version, description, applicable document types)
- [ ] 1.1.3 Add validation for complete pipeline configuration

### 1.2 Gate Definition Integration

- [ ] 1.2.1 Define gate_pdf_ir_ready with ledger condition:
  - [ ] 1.2.1.1 Check `pdf_ir_ready=true` for current document
  - [ ] 1.2.1.2 Set timeout for gate waiting (e.g., 30 minutes)
  - [ ] 1.2.1.3 Define resume stage (chunk) when condition met
- [ ] 1.2.2 Add gate validation in pipeline schema
- [ ] 1.2.3 Update pipeline loader to handle gate definitions

### 1.3 Stage Configuration Validation

- [ ] 1.3.1 Add validation that ingest stage has adapter binding
- [ ] 1.3.2 Ensure download stage has URL source configuration
- [ ] 1.3.3 Validate gate conditions reference valid ledger fields
- [ ] 1.3.4 Check stage dependency consistency

## 2. Stage Implementation Framework

### 2.1 Download Stage Implementation

- [ ] 2.1.1 Create `DownloadStage` class implementing stage protocol
- [ ] 2.1.1.1 Extract PDF URLs from document metadata
- [ ] 2.1.1.2 Download PDF files with retry and error handling
- [ ] 2.1.1.3 Update ledger with `pdf_downloaded=true`
- [ ] 2.1.2 Add download stage to stage factory registry
- [ ] 2.1.3 Implement download progress tracking and metrics

### 2.2 Gate Stage Implementation

- [ ] 2.2.1 Create `GateStage` class implementing stage protocol
- [ ] 2.2.1.1 Evaluate gate conditions against ledger state
- [ ] 2.2.1.2 Raise `GateConditionError` when conditions not met
- [ ] 2.2.1.3 Support timeout and retry logic for gate waiting
- [ ] 2.2.2 Add gate stage to stage factory registry
- [ ] 2.2.3 Implement gate evaluation metrics and logging

### 2.3 Enhanced Stage Registry

- [ ] 2.3.1 Update `build_default_stage_factory` to include download and gate stages
- [ ] 2.3.2 Add stage metadata for download and gate types
- [ ] 2.3.3 Ensure stage builders handle new stage configurations

## 3. Pipeline Execution Updates

### 3.1 Two-Phase Execution Model

- [ ] 3.1.1 Update Dagster job building to handle gate stages
- [ ] 3.1.1.1 Gates should not produce outputs but control flow
- [ ] 3.1.1.2 Pre-gate stages run in phase 1, post-gate in phase 2
- [ ] 3.1.1.3 Proper dependency wiring for gate-controlled execution
- [ ] 3.1.2 Add gate evaluation to pipeline execution state
- [ ] 3.1.3 Update state management for two-phase execution

### 3.2 Enhanced Pipeline Validation

- [ ] 3.2.1 Validate gate conditions reference valid ledger fields
- [ ] 3.2.2 Check stage dependencies are consistent with gate placement
- [ ] 3.2.3 Ensure required stages have proper configuration
- [ ] 3.2.4 Validate pipeline can execute without circular dependencies

## 4. Testing and Integration

### 4.1 Unit Tests for New Stages

- [ ] 4.1.1 Test `DownloadStage` with mocked PDF URLs and downloads
- [ ] 4.1.2 Test `GateStage` with various condition scenarios
- [ ] 4.1.3 Test stage factory includes new stage types
- [ ] 4.1.4 Test stage metadata and configuration validation

### 4.2 Pipeline Integration Tests

- [ ] 4.2.1 Test complete PDF pipeline configuration loads correctly
- [ ] 4.2.2 Test gate conditions are properly evaluated
- [ ] 4.2.3 Test two-phase execution model works as expected
- [ ] 4.2.4 Test error handling for malformed pipeline configurations

### 4.3 End-to-End Validation

- [ ] 4.3.1 Test PDF pipeline from ingestion to completion
- [ ] 4.3.2 Validate gate waiting and resumption behavior
- [ ] 4.3.3 Test error scenarios and recovery paths
- [ ] 4.3.4 Performance testing for pipeline execution

## 5. Documentation Updates

### 5.1 Pipeline Documentation

- [ ] 5.1.1 Update `docs/guides/pdf-two-phase-gate.md` with complete pipeline
- [ ] 5.1.2 Document download stage configuration and behavior
- [ ] 5.1.3 Document gate stage usage and condition syntax
- [ ] 5.1.4 Add troubleshooting guide for pipeline configuration errors

### 5.2 Developer Guides

- [ ] 5.2.1 Create `docs/guides/custom-pipeline-stages.md` for extending pipelines
- [ ] 5.2.2 Document stage configuration format and validation
- [ ] 5.2.3 Add examples of custom stage implementations
- [ ] 5.2.4 Update pipeline authoring guide with new stage types

## 6. Observability Enhancements

### 6.1 Pipeline Metrics

- [ ] 6.1.1 Add metrics for stage execution times including gates
- [ ] 6.1.2 Track gate evaluation success/failure rates
- [ ] 6.1.3 Monitor download stage performance and error rates
- [ ] 6.1.4 Add pipeline phase transition metrics

### 6.2 Enhanced Logging

- [ ] 6.2.1 Add structured logging for gate evaluations
- [ ] 6.2.2 Log download stage progress and results
- [ ] 6.2.3 Include pipeline phase information in logs
- [ ] 6.2.4 Add correlation IDs for two-phase execution tracing

**Total Tasks**: 45 across 6 work streams

**Risk Assessment:**

- **Medium Risk**: Changes to core pipeline execution could affect existing workflows
- **Low Risk**: Additive changes that enhance existing incomplete functionality

**Rollback Plan**: If issues arise, revert pipeline configuration changes while keeping new stage implementations for future use.
