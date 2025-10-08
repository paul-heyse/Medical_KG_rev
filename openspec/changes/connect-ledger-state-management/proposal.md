## Why

The current system has ledger helpers for PDF state tracking (`pdf_downloaded`, `pdf_ir_ready`) but no pipeline stages actually call these methods. The `pdf_ir_ready_sensor` exists but has no real producers, making the two-phase PDF pipeline non-functional. Pipeline execution and ledger state are disconnected.

## What Changes

- **🔗 Stage-Ledger Integration**: Connect pipeline stage execution to ledger state updates
- **📊 State Transition Logic**: Implement proper state machine for PDF processing lifecycle
- **🔄 Sensor Producer Connection**: Ensure stages that should trigger sensors actually update ledger state
- **⚡ Real-time State Tracking**: Maintain accurate job state throughout pipeline execution

## Impact

- **Affected specs**: `specs/orchestration/spec.md` (state management capabilities)
- **Affected code**:
  - `src/Medical_KG_rev/orchestration/dagster/stages.py` - Add ledger updates to relevant stages
  - `src/Medical_KG_rev/orchestration/ledger.py` - Enhance state management methods
  - `src/Medical_KG_rev/orchestration/dagster/runtime.py` - Integrate state tracking into execution
- **Breaking changes**: None - enhances existing state management without changing APIs
- **Migration path**: Existing state tracking continues while adding missing state transitions

## Success Criteria

- ✅ Download stage updates `pdf_downloaded` ledger field
- ✅ MinerU stage updates `pdf_ir_ready` ledger field
- ✅ Sensor properly detects state changes and triggers resume jobs
- ✅ State transitions follow proper validation and consistency rules
- ✅ Failed operations properly reset or error state
