## ADDED Requirements

### Requirement: Gate-Aware Pipeline Execution

The orchestration system SHALL recognize and properly handle gate stages in pipeline topologies, supporting two-phase execution models where gates control execution flow.

#### Scenario: Gate-controlled two-phase execution

- **GIVEN** a pipeline with gate stages defined
- **WHEN** the pipeline executes
- **THEN** pre-gate stages execute in phase 1
- **AND** gate stages evaluate conditions and control phase 2 execution
- **AND** post-gate stages only execute when gate conditions are met

#### Scenario: Gate condition evaluation

- **GIVEN** a gate stage with ledger-based conditions
- **WHEN** the gate is evaluated during execution
- **THEN** it queries the job ledger for the specified conditions
- **AND** raises `GateConditionError` if conditions are not satisfied
- **AND** allows execution to proceed when conditions are met

### Requirement: Gate Definition Schema

The pipeline topology schema SHALL support gate definitions that specify conditions, timeouts, and execution control behavior.

#### Scenario: Ledger field condition gates

- **GIVEN** a gate definition with ledger field conditions
- **WHEN** the gate is processed
- **THEN** it evaluates the conditions against the current ledger state
- **AND** supports operators like `equals`, `exists`, `changed`
- **AND** handles multiple conditions with AND/OR logic

#### Scenario: Gate timeout handling

- **GIVEN** a gate with a timeout configuration
- **WHEN** the condition is not met within the timeout period
- **THEN** it raises a timeout error with appropriate logging
- **AND** cleans up any associated state
- **AND** allows configuration of timeout behavior

## MODIFIED Requirements

### Requirement: Pipeline Topology Processing

The pipeline topology processing SHALL handle gate stages differently from regular stages, recognizing them as execution control points rather than output-producing stages.

#### Scenario: Enhanced topology loading

- **GIVEN** a pipeline topology with gate definitions
- **WHEN** the topology is loaded and validated
- **THEN** gate definitions are parsed and validated
- **AND** stage dependencies respect gate boundaries
- **AND** execution planning accounts for two-phase execution

## REMOVED Requirements

### Requirement: Gate-Ignoring Pipeline Execution

**Reason**: Replaced by gate-aware execution to support proper two-phase pipeline models and resumable execution patterns
**Migration**: Existing pipelines without gates continue to execute as before, while gated pipelines now properly handle gate stages as execution control points
