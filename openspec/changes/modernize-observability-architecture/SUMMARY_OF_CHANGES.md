# Summary of gRPC-First Architectural Clarification

## What Changed

Based on user feedback that **all internal Docker service communication must use gRPC (not HTTP)**, we have clarified and updated the change proposal documentation.

## Key Clarifications

### 1. Protocol Usage Matrix

| Communication Type | Protocol | Registry |
|-------------------|----------|----------|
| **Internal Docker services** | ✅ gRPC | gRPCMetricRegistry |
| **GPU microservices** | ✅ gRPC | gRPCMetricRegistry |
| **External client APIs** | ✅ HTTP/REST/GraphQL/SOAP | ExternalAPIMetricRegistry |
| **Adapter → External APIs** | ✅ HTTP | ExternalAPIMetricRegistry |
| **External databases** | ✅ HTTP (if required) | ExternalAPIMetricRegistry |

### 2. Updated Metric Registry Architecture

**FROM** (original): 5 registries

- GPUMetricRegistry
- HTTPMetricRegistry ❌ (ambiguous name)
- PipelineMetricRegistry
- CacheMetricRegistry
- RerankingMetricRegistry

**TO** (clarified): 6 registries

- **GPUMetricRegistry** - GPU hardware metrics only (memory, utilization, temperature)
- **gRPCMetricRegistry** - Internal service-to-service gRPC communication (NEW)
- **ExternalAPIMetricRegistry** - External HTTP traffic and adapter clients (RENAMED from HTTPMetricRegistry)
- **PipelineMetricRegistry** - Orchestration pipeline state
- **CacheMetricRegistry** - Caching layer performance
- **RerankingMetricRegistry** - Search reranking operations

## Validation: No Code Changes Required

✅ **Audited codebase** - All internal services already use gRPC correctly
✅ **Proto contracts exist** - `embedding_service.proto`, `gpu_service.proto` already defined
✅ **No internal HTTP found** - Architecture is already compliant

**This change is documentation-only to clarify metric collection scope.**

## Files Updated

### 1. New Files Created

- `ARCHITECTURE_CLARIFICATION.md` - Comprehensive clarification document (700+ lines)
- `SUMMARY_OF_CHANGES.md` - This file

### 2. Updated Files

- `proposal.md` - Updated registry list from 5 to 6, clarified gRPC-first architecture
- `specs/observability/spec.md` - Added gRPC registry requirements, renamed HTTP → External API

### 3. Files Pending Update (To Be Done)

- `design.md` - Add gRPC registry decision, update counts
- `tasks.md` - Rename HTTPMetricRegistry → ExternalAPIMetricRegistry, add gRPCMetricRegistry
- `DETAILED_TASKS.md` - Add Task 1.1.7 (gRPCMetricRegistry), update Task 1.1.3
- `README.md` - Update registry count and list

## Impact Analysis

### Scope Increase

- **Registries**: 5 → 6 (+1)
- **Implementation tasks**: ~5-7 additional tasks for gRPC registry
- **Timeline impact**: +2-3 days in Phase 1

### Benefits

- **Clearer boundaries**: gRPC vs External API vs GPU hardware
- **Architectural compliance**: Explicitly enforces gRPC-first internal communication
- **Better observability**: Separate metrics for internal RPC calls vs external HTTP
- **Reduced confusion**: "HTTPMetricRegistry" was ambiguous, "ExternalAPIMetricRegistry" is explicit

### No Risks Added

- ✅ No protocol changes required (already using gRPC)
- ✅ No breaking changes (metrics are additive)
- ✅ No performance impact (just metric collection reorganization)

## Implementation Priorities

### Must Do (High Priority)

1. ✅ Update `proposal.md` - DONE
2. ✅ Update `specs/observability/spec.md` - DONE
3. ✅ Create `ARCHITECTURE_CLARIFICATION.md` - DONE
4. ⏳ Update `design.md` - TO DO
5. ⏳ Update `tasks.md` - TO DO
6. ⏳ Update `DETAILED_TASKS.md` - TO DO

### Should Do (Medium Priority)

7. ⏳ Update `README.md` - TO DO
8. ⏳ Update `GAP_ASSESSMENT.md` references - TO DO

### Nice to Have (Low Priority)

9. Add gRPC interceptor task for automatic instrumentation
10. Add validation tests for protocol compliance

## Validation Status

✅ **openspec validate modernize-observability-architecture --strict** - PASSED

All spec deltas properly formatted with updated requirements.

## Next Steps

1. **Complete remaining documentation updates** (items 4-8 above)
2. **Add gRPCMetricRegistry implementation task** to DETAILED_TASKS.md
3. **Update gap assessment** to reflect 6 registries instead of 5
4. **Optional**: Add gRPC interceptor implementation guide

## Summary

The change proposal now explicitly enforces the architectural constraint that:

**All internal Docker service communication uses gRPC. HTTP is ONLY for external clients and external databases.**

This clarification adds one registry (gRPCMetricRegistry), renames one registry (HTTPMetricRegistry → ExternalAPIMetricRegistry), and better aligns metric collection with the system's gRPC-first internal architecture.

**Total impact**: +2-3 days in Phase 1, better architectural clarity, no code protocol changes required.
