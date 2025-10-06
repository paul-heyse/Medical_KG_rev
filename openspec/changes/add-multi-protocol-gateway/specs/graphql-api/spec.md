# GraphQL API Specification

## ADDED Requirements

### Requirement: GraphQL Schema Definition

The system SHALL provide a GraphQL endpoint with strongly-typed schema auto-generated from Pydantic models.

#### Scenario: Schema introspection

- **WHEN** querying `__schema { types { name } }`
- **THEN** all system types MUST be returned

#### Scenario: Type generation from Pydantic

- **WHEN** Pydantic Document model exists
- **THEN** corresponding GraphQL Document type MUST be generated

### Requirement: Query Operations

The system SHALL provide Query type with resolvers for fetching resources and relationships.

#### Scenario: Single resource query

- **WHEN** query `{ document(id: "123") { title } }`
- **THEN** document with ID 123 MUST be returned

#### Scenario: Related resource traversal

- **WHEN** query `{ document(id: "123") { organization { name } } }`
- **THEN** document and its organization MUST be returned in one request

### Requirement: Mutation Operations

The system SHALL provide Mutation type for data modification and triggering operations.

#### Scenario: Ingestion mutation

- **WHEN** mutation `{ startIngestion(source: "clinicaltrials", ids: ["NCT123"]) { jobId } }`
- **THEN** ingestion MUST be triggered and job ID returned

### Requirement: DataLoader Pattern

The system SHALL use DataLoader for efficient batching of database queries to prevent N+1 problems.

#### Scenario: Batch loading

- **WHEN** querying multiple documents with organizations
- **THEN** organizations MUST be loaded in batched queries, not one-by-one
