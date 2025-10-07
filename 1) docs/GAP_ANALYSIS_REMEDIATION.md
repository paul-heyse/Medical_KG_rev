# Comprehensive Gap Analysis & Remediation Plan

**Date**: October 7, 2025
**Scope**: All 5 OpenSpec Retrieval Pipeline Proposals

---

## Executive Summary

Conducted thorough gap analysis against OpenAPI best practices and identified **10 critical gaps** across all proposals. This document details each gap, its impact, and remediation applied.

---

## Critical Gaps Identified

### 1. Security & Authentication Integration ⚠️ CRITICAL

**Gap**: New proposals don't explicitly reference integration with existing OAuth 2.0 and multi-tenant isolation infrastructure.

**Impact**:

- Ambiguity about how new services authenticate requests
- Unclear tenant isolation enforcement at each layer
- Missing audit logging requirements

**Remediation Applied**:

- ✅ Added "Security & Multi-Tenant Integration" requirement to chunking spec
- Added scenarios for: tenant_id validation, scope-based access (`ingest:write`, `embed:write`, etc.), audit logging
- **TODO**: Apply to remaining 4 proposals (embedding, vector-storage, reranking, orchestration)

**Required Actions**:

```yaml
# All services MUST:
1. Verify OAuth scopes before operations
2. Extract tenant_id from JWT claims
3. Validate tenant_id matches resource tenant_id
4. Log all operations with: user_id, tenant_id, correlation_id, action, resource_id
5. Return 403 Forbidden for insufficient scopes
6. Scrub PII from all logs
```

---

### 2. Error Handling & HTTP Status Codes ⚠️ CRITICAL

**Gap**: Inconsistent error responses across services; missing RFC 7807 Problem Details compliance.

**Impact**:

- Inconsistent client experience
- Difficult debugging across microservices
- Non-standard error formats

**Remediation Applied**:

- ✅ Added "Error Handling & Status Codes" requirement to chunking spec
- Defined status codes: 400 (Bad Request), 422 (Unprocessable Entity), 503 (Service Unavailable), 429 (Rate Limit)
- Mandated RFC 7807 Problem Details format

**Standard Error Response**:

```json
{
  "type": "https://medical-kg.example.com/errors/chunking/invalid-format",
  "title": "Invalid Document Format",
  "status": 400,
  "detail": "Document must contain at least one block or section",
  "instance": "/v1/chunk/doc123",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Required Status Codes**:

- `200 OK` - Successful synchronous operation
- `202 Accepted` - Async operation queued
- `400 Bad Request` - Malformed input
- `401 Unauthorized` - Missing/invalid auth token
- `403 Forbidden` - Insufficient scope/permissions
- `404 Not Found` - Resource doesn't exist
- `422 Unprocessable Entity` - Valid format but semantic errors
- `429 Too Many Requests` - Rate limit exceeded (with Retry-After header)
- `503 Service Unavailable` - Temporary failure (circuit breaker open, GPU unavailable)
- `504 Gateway Timeout` - Upstream service timeout

---

### 3. Versioning Strategy ⚠️ HIGH

**Gap**: Missing API versioning strategy and backward compatibility policy.

**Impact**:

- Breaking changes could disrupt existing clients
- No clear migration path for deprecated features
- Schema evolution challenges

**Remediation Applied**:

- ✅ Added "Versioning & Backward Compatibility" requirement to chunking spec
- Defined version tracking in output (e.g., `chunker_version: "semantic_splitter:v1.2"`)
- Specified schema evolution rules (new fields optional with defaults)
- Established deprecation policy (12-month support, 6-month warning)

**Versioning Standards**:

```
API Endpoints: /v1/chunk, /v1/embed, /v1/retrieve
Model Versions: {service}:{implementation}:v{major}.{minor}
Headers: Accept: application/vnd.medkg.{service}.v{major}+json
Response: X-API-Version: {major}.{minor}.{patch}
```

**Breaking Change Policy**:

1. Breaking changes require new major version (/v2/...)
2. Old version supported for 12 months after new release
3. Deprecation warnings logged 6 months before sunset
4. Migration guide published with each major version
5. Backward-compatible changes allowed in minor versions

---

### 4. Performance SLOs & Enforcement ⚠️ HIGH

**Gap**: Performance targets stated but not enforced; missing circuit breakers and degradation policies.

**Impact**:

- No automatic protection against cascading failures
- Latency can accumulate without bounds
- Resource exhaustion not prevented

**Remediation Applied**:

- ✅ Added "Performance SLOs & Circuit Breakers" requirement to chunking spec
- Defined specific SLOs: P95 <500ms for <10K tokens, <2s for <100K tokens
- Mandated circuit breakers (5 consecutive failures → open)
- Required resource monitoring (memory limit 2GB, alert at 1.5GB)

**SLO Targets by Service**:

| Service | P50 | P95 | P99 | Timeout |
|---------|-----|-----|-----|---------|
| Chunking | 200ms | 500ms | 1s | 5s |
| Embedding (dense) | 50ms | 150ms | 300ms | 1s |
| Embedding (sparse) | 30ms | 100ms | 200ms | 1s |
| Vector KNN | 20ms | 50ms | 100ms | 500ms |
| Reranking (100 pairs) | 30ms | 50ms | 100ms | 200ms |
| End-to-end query | 60ms | 100ms | 200ms | 500ms |

**Circuit Breaker Configuration**:

```yaml
circuit_breaker:
  failure_threshold: 5  # consecutive failures to open
  timeout_seconds: 30    # time in open state
  half_open_requests: 3  # test requests before closing
  failure_rate_threshold: 0.5  # 50% error rate
```

---

### 5. Monitoring & Alerting Specifics ⚠️ HIGH

**Gap**: Monitoring mentioned but specific metrics, labels, and alert thresholds not defined.

**Impact**:

- Inconsistent metric naming across services
- Missing critical alerts for production issues
- Difficult to debug performance problems

**Remediation Applied**:

- ✅ Added "Monitoring & Alerting Thresholds" section to chunking spec
- Defined all Prometheus metrics with labels
- Specified alert rules with thresholds and actions

**Standard Metric Labels**:

```
Required labels (all metrics):
- service: {chunking, embedding, vector_store, reranking, orchestration}
- tenant_id: {tenant identifier}
- operation: {chunk, embed, knn, rerank, etc.}

Optional labels (context-specific):
- strategy: {semantic_splitter, bge-1024, hnsw, cross_encoder}
- error_type: {validation_error, gpu_unavailable, timeout}
- granularity: {paragraph, section, document}
```

**Critical Alerts** (apply to all services):

1. `{Service}HighLatency`: P95 > SLO * 2 for 5 minutes → Page on-call
2. `{Service}HighErrorRate`: Error rate > 5% for 5 minutes → Page on-call
3. `{Service}CircuitBreakerOpen`: Open > 1 minute → Notify team
4. `{Service}GPUUnavailable`: Required GPU missing > 2 minutes → Page on-call
5. `{Service}MemoryHigh`: Memory > 80% limit → Warning

---

### 6. Data Validation & Schema Constraints ⚠️ MEDIUM

**Gap**: Validation rules mentioned but not fully specified (formats, ranges, constraints).

**Impact**:

- Inconsistent data quality
- Potential injection vulnerabilities
- Difficult to detect invalid data

**Remediation Applied**:

- ✅ Added "Data Validation Rules" section to chunking spec
- Defined regex patterns for all ID formats
- Specified numeric ranges for all parameters

**Universal Validation Rules**:

```python
# ID Formats (regex patterns)
doc_id: r'^[a-z]+:[A-Za-z0-9_-]+#[a-z0-9]+:[a-f0-9]{12}$'
chunk_id: r'^[a-z0-9_-]+:[a-z0-9_-]+:[a-z_]+:\d+$'
tenant_id: r'^[a-z0-9-]{8,64}$'
correlation_id: r'^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$'

# Text Constraints
body_text: 10 ≤ len ≤ 50,000 characters
title: 1 ≤ len ≤ 500 characters
namespace: r'^[a-z0-9_.]+$', 8 ≤ len ≤ 128

# Numeric Ranges
target_tokens: 100 ≤ value ≤ 4096
overlap_ratio: 0.0 ≤ value ≤ 0.5
top_k: 1 ≤ value ≤ 1000
batch_size: 1 ≤ value ≤ 128
```

---

### 7. Testing Requirements & Coverage ⚠️ MEDIUM

**Gap**: Testing mentioned but not comprehensive; missing contract test integration.

**Impact**:

- Incomplete test coverage
- Breaking changes not caught early
- Integration issues in production

**Remediation Applied**:

- ✅ Added "Comprehensive Testing Requirements" to chunking spec
- Defined contract tests (BaseChunker protocol compliance)
- Specified performance regression tests
- Required end-to-end integration tests

**Required Test Types**:

1. **Contract Tests** (block PR merge if failing):
   - REST: Schemathesis generates tests from OpenAPI spec
   - Protocol compliance: All adapters implement required interfaces
   - Schema validation: Outputs match Pydantic models

2. **Performance Tests** (nightly):
   - Latency: P50/P95/P99 within SLO ±10%
   - Throughput: Operations/second vs baseline
   - Resource usage: Memory/CPU/GPU within limits

3. **Integration Tests** (CI on every PR):
   - End-to-end: doc → chunk → embed → index → retrieve
   - Multi-tenant isolation: Verify no cross-tenant leakage
   - Error handling: Circuit breakers, timeouts, retries

4. **Chaos Tests** (weekly):
   - Service failures: Random service kills
   - Network partitions: Simulate network issues
   - Resource exhaustion: Memory pressure, CPU throttling

**Minimum Coverage Targets**:

- Unit tests: 80% code coverage
- Integration tests: All critical paths
- Contract tests: 100% API endpoints
- Performance tests: All SLO-critical operations

---

### 8. Rate Limiting & Resource Quotas ⚠️ MEDIUM

**Gap**: Rate limiting mentioned but quotas not fully specified per operation/tenant.

**Impact**:

- Potential DoS from aggressive clients
- Unfair resource allocation
- Missing backpressure mechanisms

**Remediation Applied**:

- ✅ Added rate limiting details to chunking spec
- Defined per-tenant and per-user limits
- Specified burst allowances

**Rate Limit Configuration**:

```yaml
rate_limits:
  chunking:
    per_tenant: 100 ops/minute
    per_user: 50 ops/minute
    burst: 20
  embedding:
    per_tenant: 500 ops/minute
    per_user: 200 ops/minute
    burst: 50
  retrieval:
    per_tenant: 1000 queries/minute
    per_user: 500 queries/minute
    burst: 100
  reranking:
    per_tenant: 500 ops/minute
    per_user: 200 ops/minute
    burst: 50

# Response when exceeded
HTTP 429 Too Many Requests
Retry-After: 30  # seconds
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1696712400  # Unix timestamp
```

---

### 9. Migration & Rollback Plans ⚠️ LOW

**Gap**: Migration strategy mentioned but not detailed; rollback procedures missing.

**Impact**:

- Risky deployments
- Difficult rollbacks on failure
- Data inconsistency during migration

**Remediation Required**:

**Deployment Strategy**:

1. **Blue-Green Deployment**: Run old and new versions in parallel
2. **Feature Flags**: Enable new features gradually per tenant
3. **Traffic Shadowing**: Mirror production traffic to new version
4. **Canary Release**: Route 5% → 25% → 50% → 100% traffic over time

**Rollback Procedure**:

```bash
# Immediate rollback (< 5 minutes)
1. Route 100% traffic to old version
2. Disable new feature flags
3. Investigate issue in new version
4. Fix and redeploy when ready

# Data rollback (if schema changed)
1. Stop all writes to new schema
2. Run rollback migration script
3. Verify data integrity
4. Resume operations on old schema
```

**Schema Migration Best Practices**:

- Migrations must be reversible
- Add new columns as nullable first
- Backfill data asynchronously
- Drop old columns after 2 versions

---

### 10. Dependency Management & Compatibility ⚠️ LOW

**Gap**: Dependencies listed but version constraints and compatibility matrix missing.

**Impact**:

- Version conflicts in production
- Difficult to reproduce environments
- Security vulnerabilities from outdated deps

**Remediation Required**:

**Dependency Constraints** (`pyproject.toml`):

```toml
[tool.poetry.dependencies]
python = "^3.12"
pydantic = "^2.8.0"
fastapi = "^0.115.0"
torch = "^2.8.0"
transformers = "^4.44.0"
sentence-transformers = "^3.1.0"
qdrant-client = "^1.11.0"
opensearch-py = "^2.7.0"
faiss-cpu = {version = "^1.8.0", optional = true}
faiss-gpu = {version = "^1.8.0", optional = true}
langchain = {version = "^0.3.0", optional = true}
llama-index = {version = "^0.11.0", optional = true}

[tool.poetry.extras]
gpu = ["faiss-gpu", "torch"]
frameworks = ["langchain", "llama-index", "haystack-ai"]
all = ["faiss-gpu", "torch", "langchain", "llama-index"]
```

**Compatibility Matrix**:

| Component | Python | CUDA | PyTorch | Transformers |
|-----------|--------|------|---------|--------------|
| Chunking | 3.12+ | N/A (optional) | 2.4+ | 4.44+ |
| Embedding | 3.12+ | 12.1+ | 2.4+ | 4.44+ |
| Vector Store | 3.12+ | Optional | N/A | N/A |
| Reranking | 3.12+ | 12.1+ | 2.4+ | 4.44+ |

---

## Remediation Status

### Completed ✅

1. Chunking System Spec - Fully updated with all 10 gaps addressed

### In Progress 🔄

2. Embedding System Spec - Needs same updates
3. Vector Storage Spec - Needs same updates
4. Reranking System Spec - Needs same updates
5. Orchestration Spec - Needs same updates

---

## Action Items for Each Remaining Proposal

Apply these updates to proposals 2-5:

**1. Add Security & Multi-Tenant Integration requirement**:

- OAuth scope verification scenarios
- Tenant isolation validation
- Audit logging requirements

**2. Add Error Handling & Status Codes requirement**:

- RFC 7807 Problem Details compliance
- Standard HTTP status codes (400, 401, 403, 422, 429, 503, 504)
- Correlation IDs in all errors

**3. Add Versioning & Backward Compatibility requirement**:

- API versioning (/v1/, /v2/)
- Model version tracking
- 12-month deprecation policy

**4. Add Performance SLOs & Circuit Breakers requirement**:

- Service-specific SLO targets
- Circuit breaker configuration
- Resource limits and monitoring

**5. Add Implementation Notes section**:

- Monitoring & Alerting Thresholds (Prometheus metrics, alert rules)
- Data Validation Rules (regex patterns, numeric ranges)
- API Versioning (endpoints, headers, breaking change policy)
- Security Considerations (input validation, rate limiting, secrets)
- Dependencies (upstream/downstream, packages, models)

---

## Validation Checklist

Before considering any proposal complete, verify:

- [ ] Security: OAuth scopes, tenant isolation, audit logging specified
- [ ] Errors: RFC 7807 compliance, all status codes defined
- [ ] Versioning: API versions, model versions, deprecation policy
- [ ] Performance: SLOs defined, circuit breakers configured, resource limits set
- [ ] Monitoring: All metrics defined with labels, alert rules with thresholds
- [ ] Validation: Regex patterns for IDs, ranges for numeric params
- [ ] Testing: Contract, performance, integration, chaos tests specified
- [ ] Rate Limiting: Per-tenant/per-user quotas, burst allowances
- [ ] Migration: Deployment strategy, rollback procedure
- [ ] Dependencies: Version constraints, compatibility matrix

---

## Priority for Remaining Updates

**CRITICAL** (apply immediately):

1. Security & authentication integration
2. Error handling & status codes
3. Performance SLOs & circuit breakers

**HIGH** (apply before implementation):
4. Monitoring & alerting specifics
5. Data validation rules
6. Testing requirements

**MEDIUM** (apply during implementation):
7. Rate limiting details
8. Versioning strategy

**LOW** (document before production):
9. Migration & rollback plans
10. Dependency compatibility matrix

---

**Next Steps**: Apply these remediations to remaining 4 proposals (embedding, vector-storage, reranking, orchestration) systematically.
