# Implementation Tasks: Domain-Specific Validation & HTTP Caching

## 1. UCUM Unit Validation

- [x] 1.1 Create UCUM validator using pint library
- [x] 1.2 Define allowed units per medical context (dose, lab values, vitals)
- [x] 1.3 Implement unit normalization (convert to standard form)
- [x] 1.4 Add validation for numeric ranges with units
- [x] 1.5 Write UCUM validation tests

## 2. FHIR Resource Validation

- [x] 2.1 Integrate FHIR R5 JSON schemas
- [x] 2.2 Create FHIR validator for Evidence resources
- [x] 2.3 Create FHIR validator for ResearchStudy resources
- [x] 2.4 Create FHIR validator for MedicationStatement resources
- [x] 2.5 Add FHIR terminology validation (CodeableConcept, Coding)
- [x] 2.6 Write FHIR validation tests

## 3. Extraction Template Schemas

- [x] 3.1 Define PICO extraction schema (Population, Intervention, Comparison, Outcome)
- [x] 3.2 Define effects extraction schema (outcome measures, effect sizes, confidence intervals)
- [x] 3.3 Define adverse events schema (event type, severity, causality, frequency)
- [x] 3.4 Define dose extraction schema (drug, dose, route, frequency, duration, units)
- [x] 3.5 Define eligibility criteria schema (inclusion/exclusion, age, gender, conditions)
- [x] 3.6 Add span validation for all templates (text offsets must be valid)
- [x] 3.7 Create template validation utilities
- [x] 3.8 Write extraction template tests

## 4. SHACL Shape Definitions

- [x] 4.1 Create Document shape (required properties, cardinality)
- [x] 4.2 Create Entity shape (ontology code validation)
- [x] 4.3 Create Claim shape (subject-predicate-object validation)
- [x] 4.4 Create ExtractionActivity shape (provenance requirements)
- [x] 4.5 Create relationship shapes (TREATS, CAUSES, etc.)
- [x] 4.6 Integrate pyshacl for validation
- [x] 4.7 Add SHACL validation to KG write path
- [x] 4.8 Write SHACL validation tests

## 5. HTTP Caching

- [x] 5.1 Implement ETag generation middleware (content hash)
- [x] 5.2 Add If-None-Match request header handling
- [x] 5.3 Return 304 Not Modified for cache hits
- [x] 5.4 Add Cache-Control headers (private, max-age)
- [x] 5.5 Implement Vary header for content negotiation
- [x] 5.6 Add Last-Modified header support
- [x] 5.7 Create caching configuration (per-endpoint TTLs)
- [x] 5.8 Write HTTP caching tests

## 6. REST Health Checks

- [x] 6.1 Implement GET /health endpoint (basic alive check)
- [x] 6.2 Implement GET /ready endpoint (dependencies ready)
- [x] 6.3 Check Neo4j connectivity in readiness
- [x] 6.4 Check OpenSearch connectivity in readiness
- [x] 6.5 Check Kafka connectivity in readiness
- [x] 6.6 Check Redis connectivity in readiness
- [x] 6.7 Add version and uptime to health response
- [x] 6.8 Write health check tests

## 7. Semantic Chunking Algorithms

- [x] 7.1 Implement paragraph-aware chunker (preserve paragraph boundaries)
- [x] 7.2 Implement section-aware chunker (use section headers as boundaries)
- [x] 7.3 Implement table-aware chunker (keep tables intact)
- [x] 7.4 Implement sliding window chunker (with overlap)
- [x] 7.5 Add token counting with tiktoken
- [x] 7.6 Implement max_tokens enforcement per chunk
- [x] 7.7 Add metadata preservation (section title, chunk position)
- [x] 7.8 Write chunking algorithm tests

## 8. Cross-Encoder Reranking

- [x] 8.1 Integrate cross-encoder model (ms-marco-MiniLM or BGE-reranker)
- [x] 8.2 Implement reranking service
- [x] 8.3 Add batch scoring for efficiency
- [x] 8.4 Apply reranking to top-k results (e.g., top 100 → rerank → top 10)
- [x] 8.5 Add reranking metrics to response
- [x] 8.6 Implement reranking toggle (optional per query)
- [x] 8.7 Write reranking tests

## 9. Batch Operation Schemas

- [x] 9.1 Define BatchResponse schema with per-item status
- [x] 9.2 Define BatchError schema with item-level errors
- [x] 9.3 Implement 207 Multi-Status response builder
- [x] 9.4 Add partial success handling (some succeed, some fail)
- [x] 9.5 Include individual HTTP status codes per item
- [x] 9.6 Add summary statistics (total, succeeded, failed)
- [x] 9.7 Write batch operation tests

## 10. Integration & Documentation

- [x] 10.1 Update OpenAPI spec with new validators
- [x] 10.2 Update OpenAPI spec with caching headers
- [x] 10.3 Update OpenAPI spec with health endpoints
- [x] 10.4 Document UCUM validation rules
- [x] 10.5 Document FHIR validation requirements
- [x] 10.6 Document extraction template schemas
- [x] 10.7 Create SHACL shapes documentation
- [x] 10.8 Write comprehensive integration tests
