## Context

The current Dagster orchestration runtime uses hardcoded mappings for stage types, making extensibility difficult. Adding new stage types like `download` and `gate` requires modifying core runtime functions, creating maintenance overhead and tight coupling.

## Goals / Non-Goals

### Goals

- ✅ Enable pluggable stage registration via entry points
- ✅ Eliminate hardcoded stage type mappings from runtime
- ✅ Maintain backward compatibility with existing pipelines
- ✅ Provide clear extension points for custom stages
- ✅ Support both built-in and external stage implementations

### Non-Goals

- ❌ Change existing pipeline YAML format (backward compatibility required)
- ❌ Break existing stage implementations (migration must be transparent)
- ❌ Add runtime performance overhead (plugin discovery should be fast)
- ❌ Require changes to stage protocol interfaces (use existing contracts)

## Decisions

### Stage Metadata Architecture

**Decision**: Introduce a `StageMetadata` dataclass that encapsulates all stage behavior, replacing scattered hardcoded mappings.

**Rationale**: Centralizes stage behavior definition in one place, making it easier to add new stages and modify existing ones without touching multiple runtime functions.

**Alternatives Considered**:

- Runtime introspection of stage classes (rejected: too fragile, hard to test)
- Configuration-driven behavior (rejected: too verbose for simple cases)
- Convention-based naming (rejected: not explicit enough)

### Plugin Registry Pattern

**Decision**: Use Python entry points with `medical_kg.orchestration.stages` group for plugin discovery.

**Rationale**: Standard Python mechanism for plugin discovery, well-supported across packaging tools, allows third-party packages to extend the system.

**Alternatives Considered**:

- Custom plugin registry class (rejected: reinventing the wheel)
- Configuration file-based registration (rejected: runtime discovery harder)
- Class-based registry pattern (rejected: less flexible than entry points)

### Migration Strategy

**Decision**: Migrate existing hardcoded stages to plugin registry during `StageFactory` initialization, maintaining full backward compatibility.

**Rationale**: Allows existing pipelines to continue working unchanged while enabling new plugin-based stages to be added incrementally.

## Risks / Trade-offs

### Risk: Plugin Loading Failures

**Risk**: Malformed plugins could break stage resolution
**Mitigation**: Comprehensive validation during plugin discovery with clear error messages and fallback to built-in stages only

### Risk: Performance Impact

**Risk**: Plugin discovery could slow down StageFactory initialization
**Mitigation**: Lazy loading of plugins, caching of metadata, benchmark performance before/after changes

### Risk: Breaking Existing Pipelines

**Risk**: Changes to core runtime could break existing pipeline execution
**Mitigation**: Maintain identical behavior for existing stages, add comprehensive integration tests

### Trade-off: Complexity vs Extensibility

**Trade-off**: Added complexity of metadata system vs ability to add stages without core changes
**Decision**: Accept complexity for extensibility - this is a foundational change that enables many future enhancements

## Migration Plan

### Phase 1: Framework Implementation

1. Implement `StageMetadata` and `StageRegistry` classes
2. Update runtime functions to use metadata instead of hardcoded mappings
3. Add plugin discovery mechanism
4. Migrate existing stages to use metadata system

### Phase 2: Plugin Examples

1. Implement `download` stage plugin with metadata
2. Implement `gate` stage plugin with metadata
3. Create example plugin package structure
4. Add comprehensive tests for plugin system

### Phase 3: Integration Testing

1. Test existing pipelines continue to work unchanged
2. Test new plugin stages integrate correctly
3. Test mixed plugin and built-in stage pipelines
4. Performance testing and benchmarking

### Phase 4: Documentation and Deprecation

1. Update developer documentation with plugin examples
2. Add migration guide for custom stage implementations
3. Deprecate direct registry access patterns
4. Plan removal of hardcoded mappings in future version

## Open Questions

1. **Plugin Naming Conflicts**: How to handle multiple plugins registering the same stage type?
   - **Decision**: First plugin loaded wins, log warning for conflicts

2. **Plugin Versioning**: How to handle plugin version compatibility?
   - **Decision**: Validate metadata schema version, fail fast on incompatible plugins

3. **Testing Plugin Stages**: How to test stages that depend on external plugins?
   - **Decision**: Provide mock plugin registry for testing, document plugin testing patterns

4. **Plugin Security**: How to prevent malicious plugins from affecting the system?
   - **Decision**: Validate plugin metadata against schema, run in isolated context if needed

## Implementation Details

### StageMetadata Structure

```python
@dataclass(frozen=True)
class StageMetadata:
    stage_type: str
    state_key: str | list[str] | None  # Single key, multiple keys, or None for no state
    output_handler: Callable[[dict, Any], dict]  # Function to apply output to state
    output_counter: Callable[[Any], int]  # Function to count outputs for metrics
    description: str
    dependencies: list[str] = field(default_factory=list)
    version: str = "1.0"
```

### Entry Point Registration

```python
# setup.py or pyproject.toml
[project.entry-points."medical_kg.orchestration.stages"]
download = "my_package.stages:register_download_stage"
gate = "my_package.stages:register_gate_stage"

# my_package/stages.py
def register_download_stage() -> StageMetadata:
    return StageMetadata(
        stage_type="download",
        state_key="downloaded_files",
        output_handler=handle_download_output,
        output_counter=count_downloaded_files,
        description="Downloads files from URLs or external sources"
    )
```

### Runtime Integration

```python
class StageFactory:
    def __init__(self, plugin_registry: StageRegistry | None = None):
        self.registry = plugin_registry or StageRegistry()
        self.registry.load_plugins()  # Discover and validate plugins
        self.registry.register_defaults()  # Add built-in stages

    def resolve(self, pipeline: str, stage: StageDefinition) -> object:
        metadata = self.registry.get_metadata(stage.stage_type)
        factory = self.registry.get_factory(stage.stage_type)
        instance = factory(stage)
        # Use metadata for state management, metrics, etc.
        return instance
```

## Testing Strategy

### Unit Tests

- Test `StageMetadata` validation and serialization
- Test `StageRegistry` plugin discovery and registration
- Test runtime integration with metadata system

### Integration Tests

- Test existing pipelines work with metadata system
- Test plugin stages integrate correctly
- Test error handling for malformed plugins

### Performance Tests

- Benchmark plugin discovery overhead
- Test stage resolution performance with many plugins
- Validate memory usage with plugin registry
