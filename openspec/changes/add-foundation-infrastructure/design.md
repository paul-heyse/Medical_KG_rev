# Design Document: Foundation Infrastructure

## Context

The Medical_KG_rev system requires a solid foundation that all other components build upon. This includes a federated data model that can represent diverse biomedical content (clinical trials, literature, drug labels) in a unified way while allowing domain-specific extensions. The system must support multi-tenancy, provenance tracking, and be adaptable to multiple knowledge domains beyond medicine.

## Goals / Non-Goals

### Goals

- Unified Intermediate Representation (IR) for all ingested content
- Type-safe models with comprehensive validation
- Pluggable adapter architecture for easy integration of new data sources
- Configuration-driven behavior to support multiple domains and environments
- Comprehensive observability from the foundation layer
- Fail-fast validation to prevent bad data propagation

### Non-Goals

- Not implementing specific data source adapters (those come in subsequent changes)
- Not implementing API endpoints (gateway comes later)
- Not implementing storage backends (just abstractions)
- Not implementing domain-specific business logic (just models)

## Decisions

### Decision 1: Pydantic for All Data Models

**What**: Use Pydantic v2 for all data models with strict validation
**Why**: Type safety, automatic JSON serialization, OpenAPI schema generation, runtime validation
**Alternatives**:

- Dataclasses: Lacks validation and serialization
- attrs: Less ecosystem integration
- Plain dicts: No type safety
**Trade-off**: Slight performance overhead vs. massive safety and DX benefits

### Decision 2: Federated Data Model with Overlay Pattern

**What**: Core entities (Document, Block, Entity) with domain-specific overlays via Pydantic discriminated unions
**Why**: Allows medical, finance, legal domains to coexist while sharing infrastructure
**Example**:

```python
class Document(BaseModel):
    id: str
    type: Literal["Document"] = "Document"
    content: str
    meta: dict  # Domain-specific metadata goes here

class MedicalDocument(Document):
    """Medical domain overlay with FHIR alignment"""
    meta: MedicalMetadata  # Strong typing for medical fields

class FinancialDocument(Document):
    """Financial domain overlay with XBRL alignment"""
    meta: FinancialMetadata
```

### Decision 3: Adapter SDK with YAML Configuration

**What**: Base adapter class + YAML-driven configuration (inspired by Singer/Airbyte)
**Why**: Makes adding new data sources declarative rather than requiring full code implementations
**Example**:

```yaml
# adapters/clinicaltrials.yaml
source: "ClinicalTrialsAPI"
auth:
  type: "none"
requests:
  - url: "https://clinicaltrials.gov/api/v2/studies/{id}"
    method: GET
    params:
      format: "JSON"
mapping:
  document:
    id: "$.protocolSection.identificationModule.nctId"
    title: "$.protocolSection.identificationModule.briefTitle"
```

### Decision 4: Fail-Fast Validation Strategy

**What**: Validate all inputs at ingestion boundaries; reject invalid data immediately
**Why**: Prevents cascading errors, makes debugging easier, ensures data quality
**Examples**:

- NCT ID format validation before API calls
- DOI format validation
- Span coordinates must be within document bounds
- Required fields enforced by Pydantic

### Decision 5: OpenTelemetry for Observability

**What**: Structured logging + traces from foundation layer
**Why**: Distributed tracing across microservices, standardized telemetry
**Implementation**: Context propagation in HTTP client, span creation in key functions

### Decision 6: Storage Abstraction Layer

**What**: Abstract interfaces for object store, cache, ledger
**Why**: Allows swapping implementations (S3 vs MinIO, Redis vs in-memory)
**Pattern**: Repository pattern with async/await support

### Decision 7: ID Generation Strategy

**What**: Composite IDs with format `{source}:{source_id}#{version}:{hash12}`
**Why**: Globally unique, includes provenance, supports versioning, deterministic
**Example**: `clinicaltrials:NCT04267848#v1:a3f2c1b4e5d6`

## Risks / Trade-offs

### Risk 1: Pydantic Performance for Large Documents

**Mitigation**: Use `model_validate_json()` for parsing, streaming for very large docs, lazy validation where appropriate

### Risk 2: YAML Adapter Configs May Be Insufficient

**Mitigation**: Fallback to Python code for complex adapters, provide escape hatches

### Risk 3: Federation Complexity

**Mitigation**: Start with medical domain only, add others incrementally, keep core models stable

### Risk 4: Over-Abstraction

**Mitigation**: Follow YAGNI (You Ain't Gonna Need It), only abstract when 2+ use cases exist

## Migration Plan

This is the initial implementation (no migration needed). Future changes will extend but not break these foundations.

**Version compatibility**: All models include explicit version fields for future migrations.

## Open Questions

1. **Q**: Should we use SQLAlchemy models alongside Pydantic for database persistence?
   **A**: No, keep Pydantic for API/IR layer, use separate graph/index schemas for persistence

2. **Q**: How to handle backward compatibility when medical overlay evolves?
   **A**: Version the overlay models, support multiple versions in parallel during transitions

3. **Q**: Should configuration support hot reloading?
   **A**: Not in v1; require restart for config changes to ensure consistency
