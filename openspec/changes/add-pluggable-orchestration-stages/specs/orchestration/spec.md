## ADDED Requirements

### Requirement: Pluggable Stage Registration

The orchestration system SHALL support registration of custom stage types via plugin entry points, allowing new stages to integrate without modifying core runtime code.

#### Scenario: Register custom download stage

- **WHEN** a developer creates a custom download stage implementation
- **AND** registers it via the `medical_kg.orchestration.stages` entry point group
- **THEN** the stage becomes available for use in pipeline topologies
- **AND** appears in stage type validation and discovery

#### Scenario: Register custom gate stage

- **WHEN** a developer creates a conditional gate stage implementation
- **AND** provides stage metadata including state handling and output counting
- **THEN** the gate stage integrates with existing pipeline execution flow
- **AND** can be used in PDF two-phase pipelines without runtime modifications

### Requirement: Stage Metadata System

The orchestration system SHALL use a metadata-driven approach for stage behavior, eliminating hardcoded mappings for state keys, output handling, and metrics collection.

#### Scenario: Query stage metadata

- **WHEN** a pipeline references a stage type
- **THEN** the system looks up stage metadata from the plugin registry
- **AND** uses the metadata to determine state keys and output handling
- **AND** applies appropriate metrics collection for the stage type

#### Scenario: Extend stage behavior via metadata

- **WHEN** a new stage type is registered with custom metadata
- **THEN** the runtime adapts its behavior based on the metadata
- **AND** no core runtime code changes are required
- **AND** existing stages continue to work with their default metadata

### Requirement: Plugin Discovery and Loading

The orchestration system SHALL automatically discover and load stage plugins at runtime using Python entry points.

#### Scenario: Automatic plugin discovery

- **WHEN** the StageFactory is initialized
- **THEN** it scans for `medical_kg.orchestration.stages` entry points
- **AND** loads and validates all discovered stage plugins
- **AND** merges plugin stages with built-in stages

#### Scenario: Plugin validation

- **WHEN** a plugin is discovered
- **THEN** the system validates the stage metadata for required fields
- **AND** ensures no conflicts with existing stage types
- **AND** logs warnings for malformed plugin registrations

## MODIFIED Requirements

### Requirement: Stage Factory Resolution

The StageFactory SHALL resolve stage implementations using the plugin registry and stage metadata rather than hardcoded mappings.

#### Scenario: Resolve built-in stage

- **GIVEN** a pipeline defines a `parse` stage
- **WHEN** the StageFactory resolves the stage
- **THEN** it uses the `parse` stage metadata from the default registry
- **AND** creates the appropriate stage instance
- **AND** applies the metadata-defined output handling

#### Scenario: Resolve plugin stage

- **GIVEN** a pipeline defines a `download` stage from a plugin
- **WHEN** the StageFactory resolves the stage
- **THEN** it uses the plugin-provided metadata and implementation
- **AND** integrates seamlessly with the existing pipeline execution
- **AND** applies plugin-defined state management and metrics

## REMOVED Requirements

### Requirement: Hardcoded Stage Mappings

**Reason**: Replaced by pluggable system to enable extensibility
**Migration**: Existing hardcoded mappings are migrated to the plugin registry during initialization, maintaining backward compatibility while enabling new stage types to be added via plugins
