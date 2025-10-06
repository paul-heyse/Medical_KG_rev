# Knowledge Graph Specification

## ADDED Requirements

### Requirement: Neo4j Graph Database

The system SHALL use Neo4j for storing entities, claims, and relationships with full provenance tracking.

#### Scenario: Node creation with provenance

- **WHEN** entity is extracted
- **THEN** Entity node MUST be created with linked ExtractionActivity node

#### Scenario: Idempotent MERGE operations

- **WHEN** same entity is written twice
- **THEN** graph MUST not create duplicate nodes

### Requirement: SHACL Validation

The system SHALL validate graph data against SHACL shapes before writing.

#### Scenario: Shape constraint enforcement

- **WHEN** writing node without required property
- **THEN** SHACL validation MUST reject with error
