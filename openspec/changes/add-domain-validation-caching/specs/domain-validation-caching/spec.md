# Domain-Specific Validation & HTTP Caching Specification

## ADDED Requirements

### Requirement: UCUM Unit Validation

The system SHALL validate all medical measurements use UCUM (Unified Code for Units of Measure) standardized units.

#### Scenario: Dose with valid UCUM unit

- **WHEN** extracting a dose "20 mg/dL"
- **THEN** validator MUST accept the UCUM-compliant unit

#### Scenario: Dose with invalid unit

- **WHEN** extracting a dose "20 milligrams per deciliter"
- **THEN** validator MUST normalize to "20 mg/dL" or reject if normalization fails

#### Scenario: Unit-less numeric value

- **WHEN** extracting a numeric medical measurement without units
- **THEN** validator MUST flag as error requiring units

### Requirement: FHIR Resource Validation

The system SHALL validate medical documents against HL7 FHIR R5 resource schemas where applicable.

#### Scenario: Clinical trial as FHIR ResearchStudy

- **WHEN** ingesting a clinical trial
- **THEN** validator MUST check conformance to FHIR ResearchStudy schema

#### Scenario: Evidence resource validation

- **WHEN** creating an Evidence node in knowledge graph
- **THEN** validator MUST check required FHIR Evidence fields (status, description, outcome)

#### Scenario: Invalid FHIR resource

- **WHEN** FHIR validation fails
- **THEN** system MUST return 422 Unprocessable Entity with specific validation errors

### Requirement: PICO Extraction Schema

The system SHALL define a structured schema for PICO (Population, Intervention, Comparison, Outcome) extraction with span validation.

#### Scenario: Valid PICO extraction

- **WHEN** extracting PICO elements
- **THEN** result MUST include Population, Intervention, Comparison, Outcome with text spans

#### Scenario: Population with demographics

- **WHEN** extracting Population
- **THEN** schema MUST support age range, gender, condition, sample size, and text span

#### Scenario: Outcome with measurement

- **WHEN** extracting Outcome
- **THEN** schema MUST support outcome type, measurement, timepoint, effect size, and text span

### Requirement: Adverse Events Extraction Schema

The system SHALL define a structured schema for adverse event extraction with severity grading and causality assessment.

#### Scenario: Adverse event with severity

- **WHEN** extracting adverse event
- **THEN** schema MUST include event type, severity (mild/moderate/severe/life-threatening), frequency, and span

#### Scenario: Causality assessment

- **WHEN** extracting adverse event
- **THEN** schema MUST support causality (definite/probable/possible/unlikely/unrelated)

### Requirement: Dose Extraction Schema

The system SHALL define a structured schema for medication dosing regimen extraction with UCUM-validated units.

#### Scenario: Complete dosing regimen

- **WHEN** extracting dose
- **THEN** schema MUST include drug name, dose value, units (UCUM), route, frequency, duration, and span

#### Scenario: Dose with invalid units

- **WHEN** extracted dose has non-UCUM units
- **THEN** validator MUST reject or normalize to UCUM

### Requirement: Eligibility Criteria Extraction Schema

The system SHALL define a structured schema for clinical trial eligibility criteria with inclusion and exclusion lists.

#### Scenario: Inclusion criteria

- **WHEN** extracting eligibility
- **THEN** schema MUST support inclusion criteria list with age ranges, diagnoses, biomarkers, and spans

#### Scenario: Exclusion criteria

- **WHEN** extracting eligibility
- **THEN** schema MUST support exclusion criteria list with contraindications, prior treatments, comorbidities, and spans

### Requirement: SHACL Shape Definitions

The system SHALL provide SHACL (Shapes Constraint Language) shapes for validating knowledge graph entities.

#### Scenario: Document shape validation

- **WHEN** writing Document node to graph
- **THEN** SHACL validator MUST check required properties (id, title, source, tenant_id) exist

#### Scenario: Entity shape with ontology code

- **WHEN** writing Entity node
- **THEN** SHACL validator MUST check ontology code format matches standard (RxCUI, ICD-11, SNOMED)

#### Scenario: Claim shape with provenance

- **WHEN** writing Claim node
- **THEN** SHACL validator MUST check ExtractionActivity relationship exists

### Requirement: ETag HTTP Caching

The system SHALL implement ETag-based HTTP caching for GET endpoints to reduce bandwidth and improve performance.

#### Scenario: ETag generation

- **WHEN** returning a resource
- **THEN** response MUST include ETag header with content hash

#### Scenario: Conditional GET with If-None-Match

- **WHEN** client sends If-None-Match header matching current ETag
- **THEN** system MUST return 304 Not Modified with no body

#### Scenario: ETag mismatch

- **WHEN** client sends If-None-Match header not matching current ETag
- **THEN** system MUST return 200 OK with full resource and new ETag

### Requirement: Cache-Control Headers

The system SHALL set appropriate Cache-Control headers for different resource types.

#### Scenario: Private caching for tenant data

- **WHEN** returning tenant-scoped resource
- **THEN** response MUST include Cache-Control: private header

#### Scenario: No caching for mutations

- **WHEN** returning POST/PUT/DELETE response
- **THEN** response MUST include Cache-Control: no-store

#### Scenario: Short-term caching for search results

- **WHEN** returning search results
- **THEN** response MUST include Cache-Control: private, max-age=300

### Requirement: REST Health Check Endpoints

The system SHALL provide standardized health check endpoints for liveness and readiness probes.

#### Scenario: Liveness probe

- **WHEN** GET /health is called
- **THEN** system MUST return 200 OK if service is running

#### Scenario: Readiness probe - all dependencies healthy

- **WHEN** GET /ready is called and all dependencies (Neo4j, OpenSearch, Kafka, Redis) are reachable
- **THEN** system MUST return 200 OK with dependency status

#### Scenario: Readiness probe - dependency unhealthy

- **WHEN** GET /ready is called and a dependency is unreachable
- **THEN** system MUST return 503 Service Unavailable with failed dependency details

### Requirement: Semantic Chunking Algorithms

The system SHALL provide multiple chunking strategies that preserve semantic boundaries.

#### Scenario: Paragraph-aware chunking

- **WHEN** chunking with paragraph strategy
- **THEN** chunk boundaries MUST align with paragraph breaks (double newlines)

#### Scenario: Section-aware chunking

- **WHEN** chunking academic paper
- **THEN** chunk boundaries MUST align with section headers (Introduction, Methods, Results)

#### Scenario: Table-aware chunking

- **WHEN** document contains tables
- **THEN** tables MUST be kept intact within single chunks (not split mid-table)

#### Scenario: Token limit enforcement

- **WHEN** chunk exceeds max_tokens
- **THEN** chunker MUST split at semantic boundary closest to limit

### Requirement: Cross-Encoder Reranking

The system SHALL provide optional reranking of retrieval results using a cross-encoder model for improved relevance.

#### Scenario: Reranking top-k results

- **WHEN** reranking is enabled
- **THEN** system MUST score top-k results (e.g., 100) with cross-encoder and return top-n (e.g., 10)

#### Scenario: Reranking with batch scoring

- **WHEN** reranking multiple results
- **THEN** system MUST batch queries to cross-encoder for efficiency

#### Scenario: Reranking metrics

- **WHEN** reranking is applied
- **THEN** response MUST include reranking score and original retrieval score

### Requirement: Batch Operation Responses

The system SHALL provide structured 207 Multi-Status responses for batch operations with per-item status.

#### Scenario: Partial batch success

- **WHEN** batch operation has some successes and some failures
- **THEN** system MUST return 207 Multi-Status with array of per-item results

#### Scenario: Per-item status codes

- **WHEN** returning batch response
- **THEN** each item MUST include individual HTTP status code (200, 400, 404, etc.)

#### Scenario: Batch summary statistics

- **WHEN** returning batch response
- **THEN** response MUST include summary (total items, succeeded count, failed count)

### Requirement: Extraction Template Span Validation

The system SHALL validate that all extracted spans reference valid text positions within source documents.

#### Scenario: Valid span coordinates

- **WHEN** extraction includes span [start=100, end=150]
- **THEN** validator MUST check 0 <= start < end <= document_length

#### Scenario: Span text consistency

- **WHEN** extraction includes span with text field
- **THEN** validator MUST verify text matches document[start:end]

#### Scenario: Overlapping spans detection

- **WHEN** multiple spans are extracted
- **THEN** validator MAY flag overlapping spans for review
