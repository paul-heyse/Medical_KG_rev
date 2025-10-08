## Why

The current Dagster orchestration runtime uses hardcoded stage type mappings that make it difficult to add new stage types without modifying core runtime code. Every new stage type requires changes to multiple functions (`build_default_stage_factory`, `_stage_state_key`, `_apply_stage_output`, `_infer_output_count`) and understanding of internal state management. This creates tight coupling and maintenance overhead when extending the orchestration system with new capabilities like `download` and `gate` stages.

## What Changes

- **🔧 Pluggable Stage Registry**: Introduce a plugin-based registry system where stage types can be registered via entry points, allowing new stages to integrate without modifying core runtime code
- **🏗️ Stage Metadata System**: Define stage metadata that includes state keys, output handling logic, and metrics collection, eliminating hardcoded mappings
- **📋 Stage Interface Standardization**: Standardize how stages declare their behavior through metadata rather than runtime introspection
- **🔌 Entry Point Registration**: Use Python entry points to allow external packages to register new stage types dynamically
- **🧪 Enhanced Testing**: Provide better testability for custom stages through standardized interfaces

## Impact

- **Affected specs**: `specs/orchestration/spec.md` (new stage registration capabilities)
- **Affected code**:
  - `src/Medical_KG_rev/orchestration/dagster/stages.py` - Replace hardcoded registry with pluggable system
  - `src/Medical_KG_rev/orchestration/dagster/runtime.py` - Update to use stage metadata instead of hardcoded mappings
  - `tests/orchestration/` - Add tests for pluggable stage registration
- **Breaking changes**: None - existing hardcoded stages will be migrated to the new system transparently
- **Migration path**: Existing stage types will be automatically registered via the new plugin system during initialization

## Success Criteria

- ✅ New stage types (`download`, `gate`) can be added without modifying core runtime files
- ✅ All existing hardcoded stages continue to work unchanged
- ✅ Stage metadata system provides clear extension points for custom logic
- ✅ Plugin registry supports both built-in and external stage registrations
- ✅ Test coverage includes examples of custom stage registration and execution
