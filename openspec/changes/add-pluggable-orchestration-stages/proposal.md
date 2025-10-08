## Why

The current Dagster runtime relies on a static `StageFactory` whose registry is built once inside `build_default_stage_factory`. Adding a new stage type requires editing this central map, updating `_apply_stage_output`, `_infer_output_count`, and any callers that expect a particular key structure. This creates tight coupling between stage definitions and the runtime, making it difficult to extend the system with new stage types like `download` and `gate` without surgery on core runtime code.

**Critical Barrier**: The PDF pipeline (`pdf-two-phase.yaml`) expects `download` and `gate` stages, but the default stage factory only knows about `ingest/parse/chunk/embed/index/extract/kg`. This prevents Dagster from instantiating the PDF pipeline at runtime, blocking end-to-end testing of PDF processing workflows.

## What Changes

- **Introduce `StagePluginManager`**: Create a plugin manager similar to the adapter Pluggy integration that dynamically discovers and registers stage implementations via entry points
- **Migrate existing stages**: Move current built-in stages (`ingest`, `parse`, `chunk`, etc.) onto the plugin system while maintaining backward compatibility
- **Dynamic stage resolution**: Update `StageFactory.resolve()` to use the plugin manager for stage discovery with fallback to built-in implementations
- **Plugin interface**: Define `StagePlugin` protocol with `create_stage()` method for consistent stage instantiation
- **Registration mechanism**: Allow teams to register new stage types via entry points without modifying core runtime code
- **Plugin isolation and security**: Implement sandboxing, resource limits, and access control for plugin execution
- **Plugin dependency management**: Handle plugin interdependencies, loading order, and version compatibility
- **Plugin lifecycle management**: Support hot-reloading, health monitoring, and graceful failure recovery
- **Plugin observability**: Add comprehensive logging, metrics, and debugging capabilities for plugin behavior
- **Plugin distribution**: Support plugin packaging, versioning, and marketplace-style discovery

## Impact

- **Affected specs**: `specs/orchestration/spec.md` - Stage discovery and plugin system requirements
- **Affected code**:
  - `src/Medical_KG_rev/orchestration/dagster/stages.py` - Move stage implementations to plugins
  - `src/Medical_KG_rev/orchestration/dagster/runtime.py` - Update StageFactory to use plugin manager
  - `src/Medical_KG_rev/orchestration/dagster/configuration.py` - Update stage loading to support plugins
- **Affected systems**: Dagster orchestration runtime, pipeline configuration loading, stage execution
