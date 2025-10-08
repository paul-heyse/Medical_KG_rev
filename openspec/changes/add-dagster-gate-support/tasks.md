# Implementation Tasks: Dagster Gate Support

## 1. Gate Recognition and Classification

### 1.1 Gate Detection in Pipeline Building

- [x] 1.1.1 Update `_build_pipeline_job` to identify gate stages in topology
- [x] 1.1.2 Separate stages into pre-gate and post-gate phases
- [x] 1.1.3 Create dependency graph that respects gate boundaries
- [x] 1.1.4 Add validation that gates have no output dependencies

### 1.2 Gate Metadata and Configuration

- [x] 1.2.1 Extend `StageDefinition` to include gate-specific metadata
- [x] 1.2.2 Define gate condition schema (ledger field checks, operators, values)
- [x] 1.2.3 Add gate timeout and retry configuration options
- [x] 1.2.4 Create gate condition evaluator class

## 2. Two-Phase Execution Architecture

### 2.1 Phase-Based Job Construction

- [x] 2.1.1 Create separate Dagster graphs for each execution phase
- [x] 2.1.2 Implement phase transition logic with gate evaluation
- [x] 2.1.3 Add phase state tracking in job execution context
- [x] 2.1.4 Handle phase failures and rollbacks appropriately

### 2.2 Gate Execution Implementation

- [x] 2.2.1 Create `GateStage` class that evaluates conditions without producing outputs
- [x] 2.2.2 Implement ledger-based condition checking
- [x] 2.2.3 Add `GateConditionError` for failed gate evaluations
- [x] 2.2.4 Support multiple condition types (field exists, field equals, field changed)

### 2.3 Enhanced State Management

- [x] 2.3.1 Update `_apply_stage_output` to handle gate stages (no state changes)
- [x] 2.3.2 Add gate evaluation results to execution state
- [x] 2.3.3 Track gate success/failure in job metadata
- [x] 2.3.4 Implement gate timeout handling and state cleanup

## 3. Sensor Integration for Resumption

### 3.1 Resume Job Creation

- [x] 3.1.1 Modify `pdf_ir_ready_sensor` to create resume jobs correctly
- [x] 3.1.2 Implement proper phase targeting for resume execution
- [x] 3.1.3 Add resume job validation and error handling
- [x] 3.1.4 Connect resume jobs to original job context

### 3.2 Cross-Phase State Management

- [x] 3.2.1 Ensure resume jobs inherit state from original execution
- [x] 3.2.2 Handle state serialization for job persistence
- [x] 3.2.3 Implement state validation for resume operations
- [x] 3.2.4 Add state cleanup for completed or failed jobs

## 4. Pipeline Schema Enhancements

### 4.1 Gate Definition Schema

- [x] 4.1.1 Extend `PipelineTopologyConfig` to include gate definitions
- [x] 4.1.2 Define `GateDefinition` with condition, timeout, and resume_stage
- [x] 4.1.3 Add gate validation in pipeline loading
- [x] 4.1.4 Support multiple gates per pipeline

### 4.2 Enhanced Pipeline Validation

- [x] 4.2.1 Validate gate conditions reference valid ledger fields
- [x] 4.2.2 Check that resume stages exist and are post-gate
- [x] 4.2.3 Ensure gates don't have output-producing dependencies
- [x] 4.2.4 Validate timeout values are reasonable

## 5. Testing and Validation

### 5.1 Unit Tests for Gate Logic

- [x] 5.1.1 Test gate condition evaluation with various ledger states
- [x] 5.1.2 Test gate timeout and error handling
- [x] 5.1.3 Test gate stage execution (no output production)
- [x] 5.1.4 Test gate metadata validation

### 5.2 Integration Tests for Two-Phase Execution

- [ ] 5.2.1 Test complete two-phase pipeline execution
- [ ] 5.2.2 Test gate failure scenarios and error propagation
- [x] 5.2.3 Test sensor-based job resumption
- [ ] 5.2.4 Test state management across execution phases

### 5.3 Pipeline Validation Tests

- [x] 5.3.1 Test pipeline loading with gate definitions
- [x] 5.3.2 Test invalid gate configurations are rejected
- [x] 5.3.3 Test dependency validation for gated pipelines
- [x] 5.3.4 Test pipeline serialization and deserialization

## 6. Documentation and Monitoring

### 6.1 Enhanced Pipeline Documentation

- [x] 6.1.1 Update `docs/guides/dagster-orchestration.md` with gate examples
- [x] 6.1.2 Document gate condition syntax and operators
- [x] 6.1.3 Add troubleshooting guide for gate failures
- [x] 6.1.4 Document two-phase execution model

### 6.2 Monitoring and Observability

- [x] 6.2.1 Add metrics for gate evaluation success/failure rates
- [x] 6.2.2 Track execution phase transitions
- [x] 6.2.3 Monitor gate timeout occurrences
- [x] 6.2.4 Add structured logging for gate operations

### 6.3 Developer Tools

- [x] 6.3.1 Create pipeline validation CLI tool
- [x] 6.3.2 Add gate condition testing utilities
- [x] 6.3.3 Implement pipeline visualization with gate flow
- [x] 6.3.4 Add debugging tools for gate evaluation

**Total Tasks**: 45 across 6 work streams

**Risk Assessment:**

- **Medium Risk**: Changes to core execution logic could affect pipeline reliability
- **Low Risk**: Gate functionality is additive and doesn't break existing pipelines

**Rollback Plan**: If issues arise, disable gate processing and fall back to linear execution while keeping gate definitions for future use.
