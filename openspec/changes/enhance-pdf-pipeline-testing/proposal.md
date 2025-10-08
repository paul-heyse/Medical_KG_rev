## Why

The current PDF "integration" test only exercises the MinerU simulator and vLLM proxy, not the actual ingestion pipeline with real downloads, state transitions, and Dagster job execution. The test validates infrastructure components but doesn't test the end-to-end PDF processing workflow that users would experience.

## What Changes

- **🧪 Comprehensive PDF Testing**: Create tests that exercise the complete PDF pipeline from ingestion to completion
- **⬇️ Real Download Testing**: Test actual PDF downloading from real URLs
- **🔄 State Transition Testing**: Validate ledger state changes throughout pipeline execution
- **🚪 Gate and Sensor Testing**: Test two-phase execution with proper gate handling and sensor resumption

## Impact

- **Affected specs**: `specs/orchestration/spec.md`, `specs/gateway/spec.md` (testing capabilities)
- **Affected code**:
  - `tests/` - New comprehensive PDF pipeline test suite
  - `tests/integration/test_pdf_e2e.py` - Enhanced to test real pipeline
  - `tests/orchestration/test_dagster_jobs.py` - Add PDF pipeline tests
- **Breaking changes**: None - adds testing capabilities without changing functionality
- **Migration path**: Existing limited tests continue while comprehensive tests are added

## Success Criteria

- ✅ End-to-end PDF pipeline test exercises real download, MinerU processing, and state transitions
- ✅ Gate-based two-phase execution is properly tested
- ✅ Sensor-based resumption works correctly in test scenarios
- ✅ Test suite provides confidence in PDF pipeline functionality
- ✅ Tests include both success and failure scenarios
