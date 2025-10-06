# Implementation Tasks: Domain-Specific Validation & HTTP Caching

## 1. UCUM Unit Validation

- [ ] 1.1 Create UCUM validator using pint library
- [ ] 1.2 Define allowed units per medical context (dose, lab values, vitals)
- [ ] 1.3 Implement unit normalization (convert to standard form)
- [ ] 1.4 Add validation for numeric ranges with units
- [ ] 1.5 Write UCUM validation tests

## 2. FHIR Resource Validation

- [ ] 2.1 Integrate FHIR R5 JSON schemas
- [ ] 2.2 Create FHIR validator for Evidence resources
- [ ] 2.3 Create FHIR validator for ResearchStudy resources
- [ ] 2.4 Create FHIR validator for MedicationStatement resources
- [ ] 2.5 Add FHIR terminology validation (CodeableConcept, Coding)
- [ ] 2.6 Write FHIR validation tests

## 3. Extraction Template Schemas

- [ ] 3.1 Define PICO extraction schema (Population, Intervention, Comparison, Outcome)
- [ ] 3.2 Define effects extraction schema (outcome measures, effect sizes, confidence intervals)
- [ ] 3.3 Define adverse events schema (event type, severity, causality, frequency)
- [ ] 3.4 Define dose extraction schema (drug, dose, route, frequency, duration, units)
- [ ] 3.5 Define eligibility criteria schema (inclusion/exclusion, age, gender, conditions)
- [ ] 3.6 Add span validation for all templates (text offsets must be valid)
- [ ] 3.7 Create template validation utilities
- [ ] 3.8 Write extraction template tests

## 4. SHACL Shape Definitions

- [ ] 4.1 Create Document shape (required properties, cardinality)
- [ ] 4.2 Create Entity shape (ontology code validation)
- [ ] 4.3 Create Claim shape (subject-predicate-object validation)
- [ ] 4.4 Create ExtractionActivity shape (provenance requirements)
- [ ] 4.5 Create relationship shapes (TREATS, CAUSES, etc.)
- [ ] 4.6 Integrate pyshacl for validation
- [ ] 4.7 Add SHACL validation to KG write path
- [ ] 4.8 Write SHACL validation tests

## 5. HTTP Caching

- [ ] 5.1 Implement ETag generation middleware (content hash)
- [ ] 5.2 Add If-None-Match request header handling
- [ ] 5.3 Return 304 Not Modified for cache hits
- [ ] 5.4 Add Cache-Control headers (private, max-age)
- [ ] 5.5 Implement Vary header for content negotiation
- [ ] 5.6 Add Last-Modified header support
- [ ] 5.7 Create caching configuration (per-endpoint TTLs)
- [ ] 5.8 Write HTTP caching tests

## 6. REST Health Checks

- [ ] 6.1 Implement GET /health endpoint (basic alive check)
- [ ] 6.2 Implement GET /ready endpoint (dependencies ready)
- [ ] 6.3 Check Neo4j connectivity in readiness
- [ ] 6.4 Check OpenSearch connectivity in readiness
- [ ] 6.5 Check Kafka connectivity in readiness
- [ ] 6.6 Check Redis connectivity in readiness
- [ ] 6.7 Add version and uptime to health response
- [ ] 6.8 Write health check tests

## 7. Semantic Chunking Algorithms

- [ ] 7.1 Implement paragraph-aware chunker (preserve paragraph boundaries)
- [ ] 7.2 Implement section-aware chunker (use section headers as boundaries)
- [ ] 7.3 Implement table-aware chunker (keep tables intact)
- [ ] 7.4 Implement sliding window chunker (with overlap)
- [ ] 7.5 Add token counting with tiktoken
- [ ] 7.6 Implement max_tokens enforcement per chunk
- [ ] 7.7 Add metadata preservation (section title, chunk position)
- [ ] 7.8 Write chunking algorithm tests

## 8. Cross-Encoder Reranking

- [ ] 8.1 Integrate cross-encoder model (ms-marco-MiniLM or BGE-reranker)
- [ ] 8.2 Implement reranking service
- [ ] 8.3 Add batch scoring for efficiency
- [ ] 8.4 Apply reranking to top-k results (e.g., top 100 → rerank → top 10)
- [ ] 8.5 Add reranking metrics to response
- [ ] 8.6 Implement reranking toggle (optional per query)
- [ ] 8.7 Write reranking tests

## 9. Batch Operation Schemas

- [ ] 9.1 Define BatchResponse schema with per-item status
- [ ] 9.2 Define BatchError schema with item-level errors
- [ ] 9.3 Implement 207 Multi-Status response builder
- [ ] 9.4 Add partial success handling (some succeed, some fail)
- [ ] 9.5 Include individual HTTP status codes per item
- [ ] 9.6 Add summary statistics (total, succeeded, failed)
- [ ] 9.7 Write batch operation tests

## 10. Integration & Documentation

- [ ] 10.1 Update OpenAPI spec with new validators
- [ ] 10.2 Update OpenAPI spec with caching headers
- [ ] 10.3 Update OpenAPI spec with health endpoints
- [ ] 10.4 Document UCUM validation rules
- [ ] 10.5 Document FHIR validation requirements
- [ ] 10.6 Document extraction template schemas
- [ ] 10.7 Create SHACL shapes documentation
- [ ] 10.8 Write comprehensive integration tests
