## Why

Pipeline configuration already declares stage `type` values such as `download` and `gate`, but there are no corresponding implementations, creating a mismatch between configuration and runtime. The current system requires editing core runtime code whenever a new stage type appears in YAML configuration, making it difficult for teams to add custom stages without modifying the central codebase.

## What Changes

- **Formalize `StagePlugin` protocol**: Create a standardized interface similar to adapter Pluggy hooks that allows teams to publish stage handlers dynamically discoverable by the Dagster runtime
- **Dynamic stage discovery**: Update the runtime to discover stage implementations from plugins rather than requiring hardcoded mappings
- **Plugin-based stage handlers**: Allow teams to implement and register stage types like `download` and `gate` without modifying core code
- **Configuration-runtime alignment**: Ensure that any stage type declared in pipeline YAML has a corresponding plugin implementation
- **Plugin metadata system**: Include stage type capabilities, version requirements, and dependency information

## Impact

- **Affected specs**: `specs/orchestration/spec.md` - Stage extension API and plugin system requirements
- **Affected code**:
  - `src/Medical_KG_rev/orchestration/dagster/runtime.py` - Update to use plugin-based stage discovery
  - `src/Medical_KG_rev/orchestration/dagster/configuration.py` - Add plugin metadata to stage definitions
  - `src/Medical_KG_rev/orchestration/dagster/stages.py` - Move built-in stages to plugin implementations
- **Affected systems**: Dagster orchestration runtime, pipeline configuration loading, stage execution
