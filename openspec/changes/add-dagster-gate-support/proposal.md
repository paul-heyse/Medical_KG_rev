## Why

The current Dagster runtime ignores gate definitions in pipeline topologies, treating them as regular stages and attempting to execute them. This breaks the intended two-phase execution model where gates should control execution flow rather than produce outputs, preventing proper PDF pipeline functionality.

## What Changes

- **🚪 Gate-Aware Execution**: Modify Dagster job building to recognize and handle gate stages differently from regular stages
- **⏸️ Execution Control**: Implement gate evaluation logic that can halt or resume pipeline execution
- **🔄 Two-Phase Model**: Support proper two-phase execution where pre-gate stages run first, then post-gate stages after conditions are met
- **⚡ Sensor Integration**: Connect gate evaluation to Dagster sensors for resumable execution

## Impact

- **Affected specs**: `specs/orchestration/spec.md` (gate execution capabilities)
- **Affected code**:
  - `src/Medical_KG_rev/orchestration/dagster/runtime.py` - Add gate handling logic
  - `src/Medical_KG_rev/orchestration/dagster/stages.py` - Gate stage implementation
  - `src/Medical_KG_rev/orchestration/ledger.py` - Gate condition evaluation
- **Breaking changes**: None - enhances existing pipeline execution without changing API
- **Migration path**: Existing pipelines without gates continue to work unchanged

## Success Criteria

- ✅ Gate stages are recognized and handled differently from regular stages
- ✅ Pre-gate stages execute in phase 1, post-gate in phase 2
- ✅ Gate conditions properly evaluated against ledger state
- ✅ Failed gates prevent execution of dependent stages
- ✅ Gate timeout and error handling work correctly
