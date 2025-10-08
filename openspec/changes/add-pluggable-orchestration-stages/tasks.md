# Implementation Tasks: Pluggable Orchestration Stages

## 1. Stage Metadata System Design

### 1.1 Define Stage Metadata Model

- [x] 1.1.1 Create `StageMetadata` dataclass with fields:
  - `stage_type: str` - The stage type identifier
  - `state_key: str` - Key used in orchestration state (e.g., "payloads", "document")
  - `output_handler: Callable` - Function to apply stage output to state
  - `output_counter: Callable` - Function to count stage outputs for metrics
  - `description: str` - Human-readable description of stage purpose
  - `dependencies: list[str]` - Optional list of stage types this depends on
- [x] 1.1.2 Create `StageRegistry` class to manage stage metadata
- [x] 1.1.3 Add validation for metadata consistency (e.g., state_key should be valid Python identifier)

### 1.2 Plugin Registry Architecture

- [x] 1.2.1 Design entry point group: `medical_kg.orchestration.stages`
- [x] 1.2.2 Create `StagePlugin` protocol for stage registration functions
- [x] 1.2.3 Implement `discover_stages()` function using `importlib.metadata.entry_points()`
- [x] 1.2.4 Add error handling for malformed plugin registrations

## 2. Core Runtime Refactoring

### 2.1 Migrate Existing Stages to Metadata System

- [x] 2.1.1 Define metadata for all existing stage types:
  - `ingest` → state_key: "payloads", output_handler: set_payloads, output_counter: len
  - `parse` → state_key: "document", output_handler: set_document, output_counter: 1
  - `ir-validation` → state_key: "document", output_handler: set_document, output_counter: 1
  - `chunk` → state_key: "chunks", output_handler: set_chunks, output_counter: len
  - `embed` → state_key: "embedding_batch", output_handler: set_embedding_batch, output_counter: len(vectors)
  - `index` → state_key: "index_receipt", output_handler: set_index_receipt, output_counter: chunks_indexed
  - `extract` → state_key: ["entities", "claims"], output_handler: unpack_extraction, output_counter: len(entities)+len(claims)
  - `knowledge-graph` → state_key: "graph_receipt", output_handler: set_graph_receipt, output_counter: nodes_written
- [x] 2.1.2 Create default metadata registry with all existing stages
- [x] 2.1.3 Update `build_default_stage_factory` to use metadata registry

### 2.2 Update Runtime Functions

- [x] 2.2.1 Replace hardcoded `_stage_state_key()` with metadata lookup
- [x] 2.2.2 Replace hardcoded `_apply_stage_output()` with metadata-driven output handling
- [x] 2.2.3 Replace hardcoded `_infer_output_count()` with metadata-driven counting
- [x] 2.2.4 Update `StageFactory.resolve()` to use metadata for validation

### 2.3 Add Plugin Discovery

- [x] 2.3.1 Modify `StageFactory.__init__()` to accept optional plugin registry
- [x] 2.3.2 Add `register_stage()` method to dynamically add stages at runtime
- [x] 2.3.3 Implement `load_plugins()` class method to discover and load stage plugins
- [x] 2.3.4 Add plugin validation during discovery (ensure required metadata fields)

## 3. Example Plugin Implementations

### 3.1 Download Stage Plugin

- [x] 3.1.1 Create `download` stage metadata:
  - state_key: "downloaded_files"
  - output_handler: handles file download results
  - output_counter: counts downloaded files
- [x] 3.1.2 Implement `DownloadStage` class implementing the stage protocol
- [x] 3.1.3 Create entry point registration for download stage
- [x] 3.1.4 Add configuration for download stage (URLs, retry policies, etc.)

### 3.2 Gate Stage Plugin

- [x] 3.2.1 Create `gate` stage metadata:
  - state_key: None (gate stages don't produce outputs)
  - output_handler: no-op handler
  - output_counter: returns 0
- [x] 3.2.2 Implement `GateStage` class that checks conditions and raises `GateConditionError` if not met
- [x] 3.2.3 Create entry point registration for gate stage
- [x] 3.2.4 Add configuration for gate conditions (ledger field checks, timeout, etc.)

## 4. Pipeline Configuration Updates

### 4.1 Extend Pipeline Schema

- [ ] 4.1.1 Add plugin registration section to `PipelineTopologyConfig`
- [ ] 4.1.2 Add stage metadata override capabilities in pipeline YAML
- [ ] 4.1.3 Update pipeline validation to handle plugin-registered stages

### 4.2 Update Existing Pipelines

- [ ] 4.2.1 Update `config/orchestration/pipelines/auto.yaml` to use new plugin system
- [ ] 4.2.2 Update `config/orchestration/pipelines/pdf-two-phase.yaml` to include gate stage
- [ ] 4.2.3 Ensure backward compatibility with existing pipeline definitions

## 5. Testing and Validation

### 5.1 Unit Tests for Core System

- [x] 5.1.1 Test `StageMetadata` validation and serialization
- [x] 5.1.2 Test `StageRegistry` plugin discovery and registration
- [x] 5.1.3 Test runtime functions use metadata correctly
- [x] 5.1.4 Test error handling for unknown stage types

### 5.2 Integration Tests for New Stages

- [ ] 5.2.1 Test download stage end-to-end with mocked file downloads
- [ ] 5.2.2 Test gate stage with various condition scenarios (pass/fail/timeout)
- [ ] 5.2.3 Test plugin registration and discovery mechanisms

### 5.3 Pipeline Integration Tests

- [ ] 5.3.1 Test PDF two-phase pipeline with gate stage integration
- [ ] 5.3.2 Test mixed plugin and built-in stage pipelines
- [ ] 5.3.3 Test pipeline validation with new stage types

## 6. Documentation and Examples

### 6.1 Developer Documentation

- [ ] 6.1.1 Update `docs/guides/pipeline-authoring.md` with plugin stage examples
- [ ] 6.1.2 Create `docs/guides/custom-stages.md` with step-by-step guide
- [ ] 6.1.3 Document entry point specification for third-party plugins

### 6.2 Code Examples

- [ ] 6.2.1 Create example plugin package structure
- [ ] 6.2.2 Add example download and gate stage implementations
- [ ] 6.2.3 Create integration test examples for custom stages

## 7. Migration and Compatibility

### 7.1 Backward Compatibility

- [ ] 7.1.1 Ensure existing pipelines continue to work unchanged
- [ ] 7.1.2 Maintain API compatibility for `StageFactory` and related classes
- [ ] 7.1.3 Provide migration guide for existing custom stage implementations

### 7.2 Deprecation Strategy

- [ ] 7.2.1 Add deprecation warnings for direct registry access
- [ ] 7.2.2 Plan timeline for removing hardcoded stage mappings
- [ ] 7.2.3 Update all internal code to use new plugin system

## 8. Performance and Observability

### 8.1 Performance Validation

- [ ] 8.1.1 Benchmark stage resolution time (should be equivalent to current system)
- [ ] 8.1.2 Test plugin discovery overhead (should be minimal)
- [ ] 8.1.3 Validate memory usage with large numbers of registered stages

### 8.2 Observability Enhancements

- [ ] 8.2.1 Add metrics for plugin discovery and stage registration
- [ ] 8.2.2 Enhance error reporting for malformed plugins
- [ ] 8.2.3 Add structured logging for stage metadata operations

## 9. Security Considerations

### 9.1 Plugin Security

- [ ] 9.1.1 Add validation for plugin metadata to prevent malicious registrations
- [ ] 9.1.2 Implement plugin isolation (plugins shouldn't affect each other)
- [ ] 9.1.3 Add audit logging for plugin registration events

### 9.2 Input Validation

- [ ] 9.2.1 Validate stage metadata against schema before registration
- [ ] 9.2.2 Sanitize stage configuration parameters
- [ ] 9.2.3 Prevent plugin name collisions and conflicts

## 10. Production Readiness

### 10.1 Deployment Updates

- [ ] 10.1.1 Update Docker images to include new orchestration components
- [ ] 10.1.2 Add plugin discovery to container initialization
- [ ] 10.1.3 Update Kubernetes configurations for new dependencies

### 10.2 Monitoring Integration

- [ ] 10.2.1 Add alerts for plugin registration failures
- [ ] 10.2.2 Monitor stage resolution performance
- [ ] 10.2.3 Track usage of plugin vs built-in stages

**Total Tasks**: 45 across 10 work streams

**Risk Assessment:**

- **High Risk**: Core runtime changes could break existing pipelines
- **Medium Risk**: Plugin system could introduce security vulnerabilities
- **Low Risk**: Documentation and examples are straightforward

**Rollback Plan**: If critical issues arise, revert to previous hardcoded system via feature flag, allowing gradual migration of custom stages.
