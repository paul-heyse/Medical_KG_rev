## 1. Design & Planning

- [ ] 1.1 Analyze current StageFactory implementation and identify plugin boundaries
- [ ] 1.2 Design StagePlugin protocol interface similar to adapter plugins
- [ ] 1.3 Define plugin metadata schema (name, version, capabilities, dependencies)
- [ ] 1.4 Plan migration strategy for existing built-in stages
- [ ] 1.5 Design fallback mechanism for missing stage types
- [ ] 1.6 Analyze plugin dependency chains and loading order requirements
- [ ] 1.7 Design plugin isolation and sandboxing strategy
- [ ] 1.8 Plan plugin version compatibility and conflict resolution
- [ ] 1.9 Design plugin health monitoring and failure detection
- [ ] 1.10 Define plugin security model and access control boundaries

### Critical Library Integration Requirements

- [x] 1.11 **Integrate `pluggy>=1.6.0`**: Design plugin hook system for stage discovery and registration
- [x] 1.12 **Integrate `dagster>=1.11.13`**: Ensure plugin system works with Dagster's execution model
- [x] 1.13 **Integrate `pydantic>=2.11.10`**: Design typed plugin interfaces and configuration validation
- [x] 1.14 **Integrate `tenacity>=9.1.2`**: Add retry logic for plugin loading and execution failures
- [x] 1.15 **Integrate `structlog`**: Add structured logging for plugin lifecycle events

## 2. Plugin Infrastructure

- [x] 2.1 Create StagePluginManager class with entry point discovery
- [x] 2.2 Implement plugin loading and validation logic
- [x] 2.3 Add plugin metadata collection and capability reporting
- [ ] 2.4 Create plugin lifecycle management (load, validate, unload)
- [x] 2.5 Add plugin configuration schema validation
- [ ] 2.6 Implement plugin dependency resolution and loading order
- [ ] 2.7 Add plugin isolation mechanisms and resource limits
- [ ] 2.8 Create plugin version compatibility checking and conflict resolution
- [ ] 2.9 Implement plugin health monitoring and failure recovery
- [ ] 2.10 Add plugin security validation and access control

## 3. Stage Plugin Interface

- [ ] 3.1 Define StagePlugin abstract base class
- [ ] 3.2 Implement create_stage() method with StageDefinition parameter
- [ ] 3.3 Add plugin metadata properties (name, version, stage_types)
- [ ] 3.4 Create plugin validation interface
- [ ] 3.5 Add error handling for plugin instantiation failures
- [ ] 3.6 Define plugin lifecycle hooks (initialize, cleanup, health_check)
- [ ] 3.7 Add plugin resource management and cleanup interfaces
- [ ] 3.8 Create plugin configuration interfaces and validation
- [ ] 3.9 Implement plugin dependency declaration and resolution
- [ ] 3.10 Add plugin metrics and observability interfaces

## 4. Runtime Integration

- [x] 4.1 Update StageFactory to use StagePluginManager
- [x] 4.2 Modify resolve() method to check plugins first, then fallbacks
- [x] 4.3 Update error handling for missing stage types
- [x] 4.4 Add plugin discovery logging and metrics
- [x] 4.5 Maintain backward compatibility with existing stage loading
- [ ] 4.6 Implement plugin hot-reloading for development environments
- [x] 4.7 Add plugin performance monitoring and optimization
- [x] 4.8 Create plugin failure recovery and circuit breaker patterns
- [ ] 4.9 Implement plugin resource cleanup and garbage collection
- [ ] 4.10 Add plugin access control and security boundary enforcement

## 5. Migrate Built-in Stages

- [x] 5.1 Create plugin wrappers for existing stage implementations
- [x] 5.2 Move AdapterIngestStage to plugin architecture
- [x] 5.3 Move AdapterParseStage to plugin architecture
- [x] 5.4 Move IRValidationStage to plugin architecture
- [x] 5.5 Move chunking, embedding, and indexing stages to plugins
- [x] 5.6 Create DownloadStage plugin for PDF asset retrieval and persistence
- [x] 5.7 Create GateStage plugin for conditional pipeline progression based on ledger state
- [x] 5.8 Update stage factory to include download and gate stage implementations
- [x] 5.9 Add PDF download logic to update JobLedger.set_pdf_downloaded
- [x] 5.10 Add PDF gate logic to check JobLedger.pdf_ir_ready and control pipeline flow
- [ ] 5.11 Update stage dependency chains for plugin-based execution
- [ ] 5.12 Migrate stage configuration to plugin-based metadata
- [ ] 5.13 Update stage error handling for plugin isolation
- [ ] 5.14 Implement stage plugin performance optimizations
- [ ] 5.15 Add stage plugin monitoring and debugging interfaces

## 6. Legacy Code Decommissioning

### Phase 1: Remove Static Stage Factory (Week 1)

- [x] 6.1 **DECOMMISSION**: Remove `build_default_stage_factory()` static implementation
- [x] 6.2 **DECOMMISSION**: Delete hardcoded stage registry mapping in `stages.py:255-317`
- [ ] 6.3 **DECOMMISSION**: Remove `_apply_stage_output` and `_infer_output_count` dict manipulation
- [ ] 6.4 **DECOMMISSION**: Delete legacy stage configuration files in `config/orchestration/stages/`
- [x] 6.5 **DECOMMISSION**: Remove unused stage import statements and dependencies

### Phase 2: Clean Up Runtime Dependencies (Week 2)

- [x] 6.6 **DECOMMISSION**: Remove legacy stage execution paths in `runtime.py`
- [ ] 6.7 **DECOMMISSION**: Delete unused stage utility functions and helpers
- [ ] 6.8 **DECOMMISSION**: Remove legacy stage error handling and fallback mechanisms
- [x] 6.9 **DECOMMISSION**: Clean up unused imports and dependencies from stage modules
- [ ] 6.10 **DECOMMISSION**: Remove legacy stage configuration validation code

### Phase 3: Documentation and Cleanup (Week 3)

- [ ] 6.11 **DECOMMISSION**: Update documentation to remove references to old stage factory
- [ ] 6.12 **DECOMMISSION**: Remove legacy stage examples and configuration templates
- [ ] 6.13 **DECOMMISSION**: Clean up test fixtures and mocks for old stage system
- [ ] 6.14 **DECOMMISSION**: Remove legacy stage debugging and introspection tools
- [ ] 6.15 **DECOMMISSION**: Final cleanup of unused files and directories

## 7. Testing & Validation

- [ ] 7.1 Create unit tests for StagePluginManager
- [ ] 7.2 Test plugin discovery and loading mechanisms
- [x] 7.3 Test stage resolution with mixed plugin/built-in stages
- [ ] 7.4 Integration tests for complete pipeline execution
- [ ] 7.5 Performance tests for plugin overhead
- [ ] 7.6 Test plugin dependency resolution and loading order
- [ ] 7.7 Test plugin isolation and resource management
- [ ] 7.8 Test plugin version conflicts and compatibility
- [ ] 7.9 Test plugin failure recovery and health monitoring
- [ ] 7.10 Test plugin security boundaries and access control

## 8. Documentation & Migration

- [ ] 8.1 Update developer documentation for creating stage plugins
- [ ] 8.2 Add plugin development examples and templates
- [ ] 8.3 Update pipeline configuration documentation
- [ ] 8.4 Create migration guide for existing custom stages
- [ ] 8.5 Add plugin troubleshooting guide
- [ ] 8.6 Document plugin security model and best practices
- [ ] 8.7 Create plugin performance tuning guide
- [ ] 8.8 Add plugin debugging and monitoring documentation
- [ ] 8.9 Create plugin distribution and packaging guide
- [ ] 8.10 Add plugin migration and upgrade strategies
