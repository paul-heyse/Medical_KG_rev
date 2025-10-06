# Design Document: Domain-Specific Validation & HTTP Caching

## Context

The core proposals established the infrastructure but left domain-specific validation rules, caching strategies, and detailed extraction schemas unspecified. This proposal fills those gaps to ensure production-quality data validation and performance optimization.

## Goals / Non-Goals

### Goals

- Medical domain validation (UCUM units, FHIR conformance)
- Structured extraction template schemas for consistency
- HTTP caching for reduced bandwidth and improved latency
- Health check endpoints for orchestration (Kubernetes, load balancers)
- Detailed chunking and reranking algorithms

### Non-Goals

- Not implementing domain overlays for finance/legal (defer to future)
- Not building custom ontology servers (use existing FHIR/RxNorm APIs)
- Not implementing distributed caching (Redis caching is separate)

## Decisions

### Decision 1: UCUM Library (Pint)

**What**: Use Python's pint library for UCUM unit validation and normalization
**Why**: Battle-tested, supports UCUM, can parse and convert units
**Example**: `pint.Quantity("20 mg/dL").to("g/L")` → automatic conversion
**Alternative**: Build custom parser (too complex, error-prone)

### Decision 2: FHIR JSON Schema Validation

**What**: Use official HL7 FHIR R5 JSON schemas with jsonschema library
**Why**: Canonical validation, keeps us aligned with FHIR spec updates
**Implementation**: Download FHIR schemas, validate medical documents against ResearchStudy/Evidence schemas
**Alternative**: Custom validation rules (harder to maintain, drift from standard)

### Decision 3: Extraction Template as Pydantic Models

**What**: Define PICO, effects, AE, dose, eligibility as strict Pydantic models
**Why**: Type safety, automatic JSON serialization, validation
**Example**:

```python
class PICOExtraction(BaseModel):
    population: Population  # Nested model
    interventions: list[Intervention]
    comparison: Optional[Comparison]
    outcomes: list[Outcome]
    confidence: float = Field(ge=0, le=1)

class Population(BaseModel):
    text_span: Span
    age_range: Optional[str]
    sample_size: Optional[int]
    condition: Optional[str]
```

### Decision 4: ETag as Content Hash

**What**: Generate ETag as SHA256 hash of response content
**Why**: Deterministic, changes when content changes, no state needed
**Implementation**: Middleware computes hash, stores in ETag header
**Alternative**: Use version numbers (requires tracking versions in database)

### Decision 5: pyshacl for Graph Validation

**What**: Use pyshacl library for SHACL shape validation
**Why**: Python-native, integrates with Neo4j, supports SHACL-SPARQL
**When**: Validate before writing to graph, not on read (performance)

### Decision 6: ms-marco-MiniLM Cross-Encoder

**What**: Use Microsoft's ms-marco-MiniLM-L-12-v2 for reranking
**Why**: Small (120MB), fast (~10ms per query-doc pair), good accuracy
**Alternative**: BGE-reranker-large (better but 1.3GB, slower)

## Risks / Trade-offs

### Risk 1: UCUM Validation Performance

**Impact**: Validating every numeric value could slow ingestion
**Mitigation**: Cache normalized units, only validate on first encounter, batch validate

### Risk 2: FHIR Schema Complexity

**Impact**: Full FHIR validation is expensive (deep nested schemas)
**Mitigation**: Validate only critical fields, not full conformance

### Risk 3: ETag Storage

**Impact**: Computing hashes on every request adds latency
**Mitigation**: Cache ETags in Redis with short TTL, only compute if missing

### Risk 4: Reranking Latency

**Impact**: Cross-encoder adds ~100-500ms for 100 results
**Mitigation**: Make optional (query parameter), only rerank top-k not all results

## Migration Plan

New capability, no migration. Existing data can be retroactively validated/normalized if needed.

## Open Questions

1. **Q**: Validate on write or on read?
   **A**: Validate on write (fail-fast), store validated data

2. **Q**: Support custom SHACL shapes per tenant?
   **A**: Not in v1, all tenants use same shapes

3. **Q**: Cache ETags indefinitely?
   **A**: No, TTL of 1 hour (data may change)
