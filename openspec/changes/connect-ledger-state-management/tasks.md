# Implementation Tasks: Connect Ledger State Management

## 1. State Transition Architecture

### 1.1 PDF State Machine Definition

- [ ] 1.1.1 Create `PdfProcessingState` enum with valid states:
  - [ ] 1.1.1.1 `NOT_STARTED` - Initial state for PDF documents
  - [ ] 1.1.1.2 `DOWNLOADING` - Download in progress
  - [ ] 1.1.1.3 `DOWNLOADED` - Download completed successfully
  - [ ] 1.1.1.4 `PROCESSING` - MinerU processing in progress
  - [ ] 1.1.1.5 `PROCESSED` - MinerU processing completed successfully
  - [ ] 1.1.1.6 `FAILED` - Processing failed permanently
- [ ] 1.1.2 Define valid state transitions and validation rules
- [ ] 1.1.3 Add state transition timestamps and metadata
- [ ] 1.1.4 Implement state validation and consistency checks

### 1.2 Enhanced Ledger Schema

- [ ] 1.2.1 Add PDF processing fields to `JobLedgerEntry`
- [ ] 1.2.2 Add state transition history tracking
- [ ] 1.2.3 Implement state query methods for sensor conditions
- [ ] 1.2.4 Add state validation before updates

## 2. Stage-Ledger Integration

### 2.1 Download Stage State Updates

- [ ] 2.1.1 Update `DownloadStage` to set `pdf_downloaded=true` on success
- [ ] 2.1.2 Add state transition logging and error handling
- [ ] 2.1.3 Implement download failure state management
- [ ] 2.1.4 Add retry count tracking for failed downloads

### 2.2 MinerU Stage State Updates

- [ ] 2.2.1 Update `MineruStage` to set `pdf_ir_ready=true` on success
- [ ] 2.2.2 Add processing state tracking during MinerU execution
- [ ] 2.2.3 Implement processing failure state management
- [ ] 2.2.4 Add processing time and resource usage tracking

### 2.3 State Management Service

- [ ] 2.3.1 Create `LedgerStateManager` for centralized state operations
- [ ] 2.3.2 Implement atomic state transitions with rollback
- [ ] 2.3.3 Add state validation and consistency checks
- [ ] 2.3.4 Integrate state manager with stage execution context

## 3. Sensor Integration

### 3.1 Enhanced Sensor Logic

- [ ] 3.1.1 Update `pdf_ir_ready_sensor` to use new state management
- [ ] 3.1.2 Add proper state condition evaluation
- [ ] 3.1.3 Implement sensor error handling and logging
- [ ] 3.1.4 Add sensor performance metrics

### 3.2 Resume Job State Inheritance

- [ ] 3.2.1 Ensure resume jobs inherit state from original execution
- [ ] 3.2.2 Implement state serialization for job persistence
- [ ] 3.2.3 Add state validation for resume operations
- [ ] 3.2.4 Handle state conflicts between original and resume jobs

## 4. State Validation and Consistency

### 4.1 State Transition Validation

- [ ] 4.1.1 Implement state machine validation for all transitions
- [ ] 4.1.2 Add business rule validation (e.g., can't mark processed without download)
- [ ] 4.1.3 Prevent invalid state combinations
- [ ] 4.1.4 Add state corruption detection and repair

### 4.2 Cross-Stage State Consistency

- [ ] 4.2.1 Ensure stage outputs match expected state transitions
- [ ] 4.2.2 Validate state consistency across pipeline execution
- [ ] 4.2.3 Implement state reconciliation for failed operations
- [ ] 4.2.4 Add state audit trail for debugging

## 5. Error Handling and Recovery

### 5.1 State-Based Error Recovery

- [ ] 5.1.1 Implement state-based retry logic for transient failures
- [ ] 5.1.2 Add state cleanup for permanently failed operations
- [ ] 5.1.3 Support manual state correction for operational issues
- [ ] 5.1.4 Implement state-based circuit breaker patterns

### 5.2 Failure State Management

- [ ] 5.2.1 Define comprehensive failure state taxonomy
- [ ] 5.2.2 Implement failure state escalation logic
- [ ] 5.2.3 Add failure analysis and reporting
- [ ] 5.2.4 Support failure state recovery procedures

## 6. Testing and Validation

### 6.1 State Management Unit Tests

- [ ] 6.1.1 Test state machine transitions and validation
- [ ] 6.1.2 Test ledger state update methods
- [ ] 6.1.3 Test state query functionality
- [ ] 6.1.4 Test state consistency validation

### 6.2 Integration Tests

- [ ] 6.2.1 Test stage-ledger integration with real state transitions
- [ ] 6.2.2 Test sensor triggering based on state changes
- [ ] 6.2.3 Test state management across pipeline failures
- [ ] 6.2.4 Test state recovery and cleanup procedures

### 6.3 End-to-End State Validation

- [ ] 6.3.1 Test complete PDF pipeline state transitions
- [ ] 6.3.2 Validate sensor-based resumption works correctly
- [ ] 6.3.3 Test state consistency across multi-stage operations
- [ ] 6.3.4 Performance testing for state management overhead

## 7. Monitoring and Observability

### 7.1 State Transition Metrics

- [ ] 7.1.1 Track state transition success/failure rates
- [ ] 7.1.2 Monitor state transition latency and throughput
- [ ] 7.1.3 Add state-based alerting for stuck transitions
- [ ] 7.1.4 Implement state transition trend analysis

### 7.2 Enhanced State Logging

- [ ] 7.2.1 Add structured logging for all state transitions
- [ ] 7.2.2 Include context information in state change logs
- [ ] 7.2.3 Add correlation IDs for state transition tracing
- [ ] 7.2.4 Implement state audit trail for compliance

### 7.3 State Dashboard

- [ ] 7.3.1 Create state management dashboard for operations
- [ ] 7.3.2 Add real-time state transition monitoring
- [ ] 7.3.3 Implement state-based alerting and notifications
- [ ] 7.3.4 Add state trend analysis and reporting

**Total Tasks**: 55 across 7 work streams

**Risk Assessment:**

- **Medium Risk**: State management changes could affect pipeline reliability if not carefully implemented
- **Low Risk**: Additive changes that enhance existing state tracking without breaking current functionality

**Rollback Plan**: If issues arise, revert to simpler state management while keeping enhanced logging and monitoring for future debugging.
