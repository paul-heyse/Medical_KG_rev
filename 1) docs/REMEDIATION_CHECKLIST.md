# Quick Action Checklist: Apply Remediations to Remaining Proposals

**Target**: `add-universal-embedding-system`, `add-vector-storage-retrieval`, `add-reranking-fusion-system`, `add-retrieval-pipeline-orchestration`

---

## Step-by-Step Remediation Template

For each remaining proposal, add these sections to the `spec.md` file:

---

### 1. Add Security & Multi-Tenant Integration (CRITICAL)

**Location**: After `## ADDED Requirements` header, before first existing requirement

```markdown
### Requirement: Security & Multi-Tenant Integration

The {SERVICE_NAME} SHALL integrate with the existing OAuth 2.0 authentication and multi-tenant isolation infrastructure.

#### Scenario: Tenant isolation in {SERVICE} metadata

- **WHEN** {SERVICE} operations execute
- **THEN** all outputs SHALL include tenant_id extracted from authenticated context
- **AND** tenant_id SHALL be validated against input resource tenant_id
- **AND** tenant_id SHALL be immutable after creation

#### Scenario: Scope-based access control

- **WHEN** {SERVICE} is invoked
- **THEN** the system SHALL verify `{REQUIRED_SCOPE}` scope in JWT token
- **AND** return 403 Forbidden if scope is missing
- **AND** log unauthorized access attempts

#### Scenario: Audit logging for {SERVICE} operations

- **WHEN** {SERVICE} operations complete
- **THEN** the system SHALL log: user_id, tenant_id, operation, duration, status
- **AND** include correlation_id for distributed tracing
- **AND** scrub PII from log messages

---
```

**Service-Specific Scopes**:

- Embedding: `embed:write`
- Vector Storage: `index:write` (for upsert), `index:read` (for KNN)
- Reranking: `retrieve:read`
- Orchestration: `ingest:write`, `retrieve:read` (depending on operation)

---

### 2. Add Error Handling & Status Codes (CRITICAL)

```markdown
### Requirement: Error Handling & Status Codes

The {SERVICE_NAME} SHALL provide comprehensive error handling with standardized HTTP status codes and RFC 7807 Problem Details.

#### Scenario: Invalid input error

- **WHEN** {SERVICE} receives malformed input
- **THEN** the system SHALL return 400 Bad Request
- **AND** include RFC 7807 Problem Details with type, title, detail, instance
- **AND** log error with correlation_id

#### Scenario: Configuration error

- **WHEN** invalid configuration specified
- **THEN** the system SHALL return 422 Unprocessable Entity
- **AND** include validation errors in Problem Details
- **AND** suggest valid alternatives

#### Scenario: Resource exhaustion error

- **WHEN** operation exceeds memory or time limits
- **THEN** the system SHALL return 503 Service Unavailable
- **AND** include Retry-After header
- **AND** trigger circuit breaker after repeated failures

#### Scenario: GPU unavailable for {SERVICE}

- **WHEN** {SERVICE} requires GPU but none available
- **THEN** the system SHALL return 503 Service Unavailable
- **AND** include clear error message about GPU requirement
- **AND** fail-fast without CPU fallback (as per design)

---
```

---

### 3. Add Versioning & Backward Compatibility (HIGH)

```markdown
### Requirement: Versioning & Backward Compatibility

The {SERVICE_NAME} SHALL support versioning for implementations and output schemas.

#### Scenario: Version tracking

- **WHEN** {SERVICE} produces output
- **THEN** output SHALL include {version_field} (e.g., "model_version: bge-large:v1.5")
- **AND** version SHALL be immutable after creation
- **AND** enable querying by version

#### Scenario: Schema evolution

- **WHEN** output schema changes (new fields added)
- **THEN** new fields SHALL be optional with defaults
- **AND** existing data SHALL remain queryable
- **AND** migration scripts SHALL handle schema updates

#### Scenario: Deprecated {SERVICE} migration

- **WHEN** implementation is deprecated
- **THEN** deprecation warning SHALL be logged
- **AND** migration path SHALL be documented
- **AND** deprecated version SHALL remain functional for 2 major versions

---
```

**Service-Specific Version Fields**:

- Embedding: `model_version` (e.g., "bge-large-en-v1.5:v1.0")
- Vector Storage: `index_version` (e.g., "hnsw:v1.0")
- Reranking: `reranker_version` (e.g., "bge-reranker-m3:v2.0")
- Orchestration: `pipeline_version` (e.g., "hybrid-rrfs:v1.0")

---

### 4. Add Performance SLOs & Circuit Breakers (HIGH)

```markdown
### Requirement: Performance SLOs & Circuit Breakers

The {SERVICE_NAME} SHALL enforce performance SLOs and implement circuit breakers for failing operations.

#### Scenario: {SERVICE} latency SLO

- **WHEN** {SERVICE} operation executes
- **THEN** P95 latency SHALL be <{P95_TARGET}ms for {INPUT_SIZE}
- **AND** P95 latency SHALL be <{P95_LARGE}ms for {LARGE_INPUT_SIZE}
- **AND** operations exceeding 5× SLO SHALL trigger alerts

#### Scenario: Circuit breaker on repeated failures

- **WHEN** {SERVICE} fails 5 consecutive times
- **THEN** circuit breaker SHALL open
- **AND** subsequent requests SHALL fail-fast with 503
- **AND** circuit SHALL attempt recovery after exponential backoff

#### Scenario: Resource monitoring

- **WHEN** {SERVICE} operations execute
- **THEN** the system SHALL monitor memory usage per operation
- **AND** reject operations exceeding {MEMORY_LIMIT}GB memory limit
- **AND** emit metrics for memory usage, CPU time, GPU utilization

---
```

**Service-Specific SLOs**:

| Service | P95 Target | Input Size | Large P95 | Large Input | Memory Limit |
|---------|------------|------------|-----------|-------------|--------------|
| Embedding (dense) | 150ms | <100 texts | 500ms | <1000 texts | 4GB |
| Embedding (sparse) | 100ms | <100 texts | 300ms | <1000 texts | 2GB |
| Vector KNN | 50ms | top_k=10 | 200ms | top_k=1000 | 2GB |
| Reranking | 50ms | 100 pairs | 200ms | 1000 pairs | 4GB |
| Orchestration | 100ms | single query | 500ms | batch queries | 1GB |

---

### 5. Add Comprehensive Testing Requirements (MEDIUM)

```markdown
### Requirement: Comprehensive Testing Requirements

The {SERVICE_NAME} SHALL include comprehensive test coverage with contract, performance, and integration tests.

#### Scenario: Contract tests for {SERVICE} interface

- **WHEN** new {SERVICE} implementation is added
- **THEN** contract tests SHALL verify protocol compliance
- **AND** validate output schema
- **AND** test parameter handling

#### Scenario: Performance regression tests

- **WHEN** {SERVICE} implementation changes
- **THEN** performance tests SHALL verify latency within SLO
- **AND** measure throughput (operations/second)
- **AND** compare against baseline to detect regressions

#### Scenario: Integration tests with downstream services

- **WHEN** {SERVICE} completes
- **THEN** integration tests SHALL verify outputs are consumable by downstream
- **AND** test end-to-end pipeline integration
- **AND** verify error propagation

---
```

---

### 6. Add Implementation Notes Section (at end of spec.md)

```markdown
---

## Implementation Notes

### Monitoring & Alerting Thresholds

**Prometheus Metrics** (all labeled by {relevant_labels}):
- `{service}_operations_total` (counter) - Total operations
- `{service}_operations_duration_seconds` (histogram) - Operation latency with buckets: [0.01, 0.05, 0.1, 0.5, 1, 2, 5]
- `{service}_errors_total` (counter) - Errors by error_type
- `{service}_memory_bytes` (gauge) - Memory usage per operation
- `{service}_gpu_utilization_percent` (gauge) - GPU utilization
- `{service}_circuit_breaker_state` (gauge) - Circuit breaker states (0=closed, 1=open, 2=half-open)

**Alert Rules**:
- `{Service}HighLatency`: P95 > {SLO * 2} for 5 minutes → Page on-call
- `{Service}HighErrorRate`: Error rate > 5% for 5 minutes → Page on-call
- `{Service}CircuitBreakerOpen`: Circuit breaker open > 1 minute → Notify team
- `{Service}MemoryHigh`: Memory usage > {LIMIT * 0.8} → Warning
- `{Service}GPUUnavailable`: GPU required but unavailable > 2 minutes → Page on-call

### Data Validation Rules

**{Service} Output Validation**:
- `{id_field}` format: `^{REGEX_PATTERN}$`
- `{text_field}` length: {MIN} ≤ len ≤ {MAX} characters
- `{numeric_field}`: {MIN} ≤ value ≤ {MAX}

**Configuration Validation**:
- `{param1}`: {MIN} ≤ value ≤ {MAX}
- `{param2}`: {MIN} ≤ value ≤ {MAX}

### API Versioning

**{Service} API Endpoints**:
- `/v1/{service}` - Current stable API
- `/v2/{service}` - Future breaking changes (reserved)

**Version Headers**:
- Request: `Accept: application/vnd.medkg.{service}.v1+json`
- Response: `Content-Type: application/vnd.medkg.{service}.v1+json`
- Response: `X-API-Version: 1.0`

**Breaking Change Policy**:
- Breaking changes require new major version
- Old version supported for 12 months after new version release
- Deprecation warnings logged 6 months before sunset
- Migration guide published with new version

### Security Considerations

**Input Validation**:
- Reject inputs > {SIZE_LIMIT}MB
- Sanitize all text content (remove control characters, validate UTF-8)
- Validate all IDs against format regex before processing

**Rate Limiting**:
- Per-tenant: {TENANT_LIMIT} operations/minute
- Per-user: {USER_LIMIT} operations/minute
- Burst: {BURST}
- Return 429 with Retry-After header when exceeded

**Secrets Management**:
- Service endpoints: Environment variables or Vault
- Model paths: Configuration files (not hardcoded)
- API keys: Rotate every 90 days

### Dependencies

- **Upstream**: {UPSTREAM_SERVICES}
- **Downstream**: {DOWNSTREAM_SERVICES}
- **Python packages**: {PACKAGES}
- **Models**: {MODELS}
```

**Service-Specific Values**:

#### Embedding

- Labels: `model_id`, `kind`, `namespace`, `batch_size`
- Rate limits: 500 ops/min (tenant), 200 ops/min (user), burst 50
- Size limit: 10MB (batch input)
- Upstream: `add-modular-chunking-system`
- Downstream: `add-vector-storage-retrieval`
- Packages: `torch`, `transformers`, `sentence-transformers`, `colbert-ai`, `pyserini`
- GPU builds: install `torch==2.8.0+cu128` (and matching `torchvision`, `torchaudio`) via `https://download.pytorch.org/whl/cu128`

#### Vector Storage

- Labels: `namespace`, `operation`, `backend`
- Rate limits: 1000 ops/min (tenant), 500 ops/min (user), burst 100
- Size limit: 50MB (batch upsert)
- Upstream: `add-universal-embedding-system`
- Downstream: `add-reranking-fusion-system`, `add-retrieval-pipeline-orchestration`
- Packages: `qdrant-client`, `faiss-cpu` or `faiss-gpu`, `pymilvus`, `opensearch-py`

#### Reranking

- Labels: `reranker_id`, `batch_size`
- Rate limits: 500 ops/min (tenant), 200 ops/min (user), burst 50
- Size limit: 5MB (batch input)
- Upstream: `add-vector-storage-retrieval`
- Downstream: `add-retrieval-pipeline-orchestration`
- Packages: `torch`, `transformers`, `sentence-transformers`

#### Orchestration

- Labels: `strategy`, `fusion_method`
- Rate limits: 1000 queries/min (tenant), 500 queries/min (user), burst 100
- Size limit: 1MB (query input)
- Upstream: ALL (chunking, embedding, vector-storage, reranking)
- Downstream: REST/GraphQL API gateways
- Packages: None (orchestration only)

---

## Validation Checklist (per proposal)

After applying all remediations, verify:

- [ ] Security: OAuth scopes, tenant isolation, audit logging specified
- [ ] Errors: RFC 7807 compliance, all status codes defined (400, 401, 403, 422, 429, 503)
- [ ] Versioning: API versions, model versions, deprecation policy
- [ ] Performance: SLOs defined, circuit breakers configured, resource limits set
- [ ] Monitoring: All metrics defined with labels, alert rules with thresholds
- [ ] Validation: Regex patterns for IDs, ranges for numeric params
- [ ] Testing: Contract, performance, integration tests specified
- [ ] Rate Limiting: Per-tenant/per-user quotas, burst allowances
- [ ] Implementation Notes: Complete section at end
- [ ] Dependencies: All upstream/downstream, packages, models listed
- [ ] **Validate with**: `openspec validate {CHANGE_ID} --strict`

---

## Expected Results

After completing all remediations:

- ✅ All 5 proposals pass `openspec validate --strict`
- ✅ Each spec.md ~700-800 lines (was ~400-500)
- ✅ ~7 new requirements per proposal (security, errors, versioning, SLOs, testing)
- ✅ Comprehensive Implementation Notes section
- ✅ Production-ready specifications

**Total effort**: ~2-3 hours for all 4 remaining proposals

---

## Quick Start

```bash
# 1. Apply remediations to embedding system
vim openspec/changes/add-universal-embedding-system/specs/embedding-core/spec.md
# (Follow template above)

# 2. Validate
openspec validate add-universal-embedding-system --strict

# 3. Repeat for vector-storage
vim openspec/changes/add-vector-storage-retrieval/specs/vector-storage/spec.md
openspec validate add-vector-storage-retrieval --strict

# 4. Repeat for reranking
vim openspec/changes/add-reranking-fusion-system/specs/reranking/spec.md
openspec validate add-reranking-fusion-system --strict

# 5. Repeat for orchestration
vim openspec/changes/add-retrieval-pipeline-orchestration/specs/orchestration/spec.md
openspec validate add-retrieval-pipeline-orchestration --strict

# 6. Final validation of all
openspec validate --strict
```

---

**Status**: ✅ Template ready. Ready to apply to remaining 4 proposals.
