# Change Proposal: Domain-Specific Validation & HTTP Caching

## Why

Fill critical gaps in data quality and performance: domain-specific validation rules (UCUM units, FHIR resources, extraction template schemas), HTTP caching with ETags for performance, and detailed chunking/reranking algorithms. These are essential for production quality but missing from core proposals.

## What Changes

- **UCUM Unit Validation**: Enforce standardized medical units (mg/dL, mmHg, etc.)
- **FHIR Resource Validation**: Validate medical documents against FHIR R5 schemas
- **Extraction Template Schemas**: Define schemas for PICO, effects, adverse events, dose, eligibility
- **SHACL Shape Definitions**: Create validation shapes for knowledge graph entities
- **HTTP Caching**: Implement ETag generation, conditional requests, Cache-Control headers
- **REST Health Checks**: Add `/health` and `/ready` endpoints
- **Semantic Chunking Algorithms**: Detail paragraph-aware, section-aware, table-aware strategies
- **Cross-Encoder Reranking**: Specify algorithm and model integration
- **Batch Operation Schemas**: Complete 207 Multi-Status response specifications

## Impact

- **Affected specs**: NEW capability `domain-validation-caching`, ENHANCEMENTS to `rest-api`, `knowledge-graph`, `retrieval-system`
- **Affected code**:
  - `src/Medical_KG_rev/validation/` - Domain-specific validators
  - `src/Medical_KG_rev/validation/ucum.py` - UCUM unit validator
  - `src/Medical_KG_rev/validation/fhir.py` - FHIR resource validator
  - `src/Medical_KG_rev/extraction/templates/` - Template schemas (PICO, effects, AE, dose, eligibility)
  - `src/Medical_KG_rev/kg/shapes/` - SHACL shape definitions
  - `src/Medical_KG_rev/gateway/caching.py` - ETag middleware
  - `src/Medical_KG_rev/gateway/health.py` - Health check endpoints
  - `src/Medical_KG_rev/chunking/strategies/` - Chunking algorithms
  - `src/Medical_KG_rev/retrieval/reranking.py` - Cross-encoder reranker
  - `tests/validation/` - Validation tests
