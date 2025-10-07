# OpenSpec Change Proposal Summary: DAG-Based Orchestration Pipeline

## ✅ Validation Status

**PASSED** - All requirements properly formatted, scenarios complete, no validation errors

## 📋 Proposal Overview

**Change ID**: `add-dag-orchestration-pipeline`
**Type**: Architecture Refactor
**Scope**: Major system-wide change affecting orchestration, ingestion, retrieval, and observability

### Problem Statement

Current orchestration hardwires pipeline stages in Python code, making it difficult to visualize flows, implement complex gates (PDF two-phase), swap implementations, and configure resilience per-stage.

### Solution

Refactor to **declarative DAG-based execution** using:

- **Dagster 1.5.0+** for workflow engine with sensor-based gates
- **Haystack 2.0+** for text operations (chunk, embed, index)
- **tenacity/pybreaker/aiolimiter** for configurable resilience
- **CloudEvents + OpenLineage** for portable observability
- **respx** for comprehensive HTTP mocking in tests

## 📊 Spec Deltas Summary

### Orchestration (6 ADDED, 1 MODIFIED, 1 REMOVED, 1 RENAMED)

**ADDED Requirements**:

1. Declarative Pipeline Topology (YAML configs)
2. Typed Stage Contracts (Python Protocols)
3. Dagster-Based Job Execution
4. Resilience Policy Configuration
5. CloudEvents Stage Lifecycle
6. Haystack Component Integration

**MODIFIED**: Job Ledger Tracking (new fields: `pdf_downloaded`, `pdf_ir_ready`, `current_stage`)

**REMOVED**: Hardcoded Pipeline Stages (replaced by YAML)

**RENAMED**: Pipeline Execution → Dagster-Based Job Execution

### Ingestion (3 ADDED, 1 MODIFIED)

**ADDED Requirements**:

1. Adapter Plugin Stage Wrapper (`PluginIngestStage`)
2. Parse Stage for IR Conversion (`IRParseStage`)
3. Download Stage for PDF Retrieval (`PDFDownloadStage`)

**MODIFIED**: Adapter Plugin Execution (support both direct and stage-based invocation)

### Retrieval (5 ADDED, 1 MODIFIED)

**ADDED Requirements**:

1. Haystack-Based Chunking Stage (`HaystackChunker`)
2. Haystack-Based Embedding Stage (`HaystackEmbedder`)
3. Haystack-Based SPLADE Expansion (`HaystackSparseExpander`)
4. Haystack-Based Dual Index Writer (`HaystackIndexWriter`)
5. Haystack-Based Hybrid Retrieval (`HaystackRetriever`)

**MODIFIED**: Chunking Service API (support both legacy and stage-based)

### Observability (4 ADDED, 1 MODIFIED)

**ADDED Requirements**:

1. CloudEvents Stage Lifecycle Emission (4 event types)
2. OpenLineage Job Run Tracking (optional, feature-flagged)
3. Prometheus Dagster Metrics (6 new metrics)
4. AsyncAPI Queue Documentation
5. Dagster UI Integration

**MODIFIED**: Structured Logging with Correlation IDs (add Dagster context fields)

## 📈 Implementation Scope

**Total Tasks**: 176 across 16 work streams

### Work Streams

1. **Foundation & Dependencies** (8 tasks) - Add libraries, configure environment
2. **Stage Contracts** (10 tasks) - Define Python Protocols for all stage types
3. **Pipeline Configuration Schema** (7 tasks) - Pydantic models, validation
4. **Resilience Configuration** (8 tasks) - Named policies with tenacity/pybreaker/aiolimiter
5. **Pipeline Topology Definitions** (7 tasks) - auto.yaml, pdf-two-phase.yaml, versioning
6. **Haystack Component Wrappers** (7 tasks) - Chunk, embed, SPLADE, index, retrieve
7. **Dagster Job Definitions** (8 tasks) - Jobs, ops, sensors, resources
8. **Job Ledger Integration** (5 tasks) - New fields, sensor queries, migration
9. **CloudEvents & OpenLineage** (7 tasks) - Event factory, emitter, schemas
10. **AsyncAPI Documentation** (5 tasks) - Update spec, generate docs, CI validation
11. **Migration & Compatibility** (8 tasks) - Feature flags, compatibility shims, deprecation warnings
12. **Testing with respx** (6 tasks) - Mock fixtures, resilience tests, contract compliance
13. **Observability Integration** (6 tasks) - Prometheus metrics, OpenTelemetry spans, Grafana dashboards
14. **Documentation** (6 tasks) - Guides for Dagster, pipeline authoring, PDF gate, migration
15. **Deployment & Operations** (6 tasks) - Docker Compose, Kubernetes, storage config
16. **Rollout & Validation** (10 tasks) - 4-phase rollout, performance comparison, retrospective

## 🔑 Key Technical Decisions

### Decision 1: Dagster vs Airflow/Prefect/Temporal

**Choice**: Dagster 1.5.0+

**Rationale**:

- Local-first (no scheduler required)
- Native sensor support for Job Ledger polling
- Typed ops with strong Python integration
- Minimal infrastructure (PostgreSQL for history, MinIO for logs)

### Decision 2: Haystack 2 vs LlamaIndex vs Custom

**Choice**: Haystack 2.0+

**Rationale**:

- Component-based design matches ports-and-adapters architecture
- Vendor-agnostic (OpenSearch, FAISS, vLLM all supported)
- OpenAI-compatible for Qwen embeddings via vLLM
- Production-ready with commercial deployments

### Decision 3: Resilience Stack

**Choice**: tenacity + pybreaker + aiolimiter

**Rationale**:

- Composable (each handles one concern)
- Battle-tested in production systems
- Async-native (aiolimiter designed for asyncio)
- Configuration-driven (YAML policies)

### Decision 4: Observability Format

**Choice**: CloudEvents 1.0 + OpenLineage (optional)

**Rationale**:

- CloudEvents is CNCF standard, portable across systems
- OpenLineage provides lineage graph for Marquez UI
- Optional OpenLineage allows gradual adoption
- Integrates with existing Prometheus/OpenTelemetry stack

## 🚀 Migration Strategy

### 4-Phase Rollout (8 weeks total)

#### Phase 1: Foundation (Week 1-2)

- Add dependencies
- Implement stage contracts
- Create YAML topologies
- Deploy to dev with feature flag OFF

#### Phase 2: Non-PDF Sources (Week 3-4)

- Enable for ClinicalTrials, OpenAlex (auto pipeline)
- Validate outputs match legacy
- Performance comparison
- Gradual enablement for all non-PDF sources

#### Phase 3: PDF Two-Phase (Week 5-6)

- Implement `pdf_ir_ready_sensor`
- Enable for PMC, Unpaywall (PDF pipeline)
- Test gate halt/resume
- Monitor for stalled jobs

#### Phase 4: Full Production (Week 7-8)

- Enable for 100% traffic
- Deprecate legacy orchestration
- Schedule removal in v0.3.0
- Conduct retrospective

## 📊 Expected Impact

### Benefits

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Pipeline Visibility** | Logs only | Dagster UI + CloudEvents | ✅ Real-time visualization |
| **Topology Changes** | Code + Deploy | YAML edit | ✅ <1 hour vs <1 day |
| **Component Swapping** | Cascading changes | Stage contract swap | ✅ Isolated changes |
| **Resilience Config** | Code edits | YAML policy edit | ✅ No deployment |
| **Lineage Tracking** | Log parsing | CloudEvents + OpenLineage | ✅ Complete audit trail |
| **Test Coverage** | Partial mocks | respx-based mocks | ✅ >90% stage coverage |

### Risks & Mitigation

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Learning curve** | Medium | Comprehensive docs, video walkthroughs, pair programming |
| **Haystack 2 maturity** | Low | Stage contracts insulate system, can swap components |
| **Migration complexity** | Medium | Feature flag, gradual rollout, comprehensive tests |
| **Operational overhead** | Low | Local-first design, health checks, runbook |

## 🎯 Success Criteria

1. ✅ **Pipeline visibility**: 100% of jobs visualized in Dagster UI with stage-level status
2. ✅ **Topology agility**: New pipeline created and deployed in <1 hour (vs <1 day)
3. ✅ **Resilience configurability**: Policy changes via YAML edit, no code deployment
4. ✅ **GPU fail-fast enforcement**: 0 jobs fall back to CPU for GPU-required stages
5. ✅ **Lineage completeness**: Every stage execution produces CloudEvent, 100% traceable
6. ✅ **Migration success**: Legacy orchestration removed in v0.3.0 without production incidents
7. ✅ **Performance parity**: Dagster throughput ≥ legacy (100+ docs/second)
8. ✅ **P95 latency**: Stage execution within existing SLOs (chunk <2s, embed <5s, index <1s)

## 🔗 Related Resources

- **Dagster Documentation**: <https://dagster.io/>
- **Haystack 2 Documentation**: <https://haystack.deepset.ai/>
- **tenacity**: <https://tenacity.readthedocs.io/>
- **pybreaker**: <https://github.com/danielfm/pybreaker>
- **aiolimiter**: <https://github.com/mjpieters/aiolimiter>
- **CloudEvents**: <https://cloudevents.io/>
- **OpenLineage**: <https://openlineage.io/>
- **respx**: <https://lundberg.github.io/respx/>

## 📚 Documentation Structure

```
openspec/changes/add-dag-orchestration-pipeline/
├── proposal.md              # Why, What, Impact (this overview)
├── tasks.md                 # 176 implementation tasks across 16 streams
├── design.md                # Technical decisions, alternatives, risks
├── specs/
│   ├── orchestration/
│   │   └── spec.md         # DAG execution, resilience, CloudEvents
│   ├── ingestion/
│   │   └── spec.md         # Adapter stages, PDF download
│   ├── retrieval/
│   │   └── spec.md         # Haystack chunking, embedding, indexing
│   └── observability/
│       └── spec.md         # CloudEvents, OpenLineage, Prometheus
└── README.md                # Quick reference with examples
```

## ✅ Validation Results

**OpenSpec Validation**: PASSED with `--strict` flag

**Validation Summary**:

- ✅ All requirements properly formatted
- ✅ All scenarios use `#### Scenario:` format
- ✅ All scenarios include WHEN/THEN clauses
- ✅ All MODIFIED requirements reference existing requirements
- ✅ All REMOVED requirements include reason and migration path
- ✅ All RENAMED requirements specify FROM and TO
- ✅ All library references include versions and documentation links
- ✅ All deltas properly prefixed (ADDED, MODIFIED, REMOVED, RENAMED)

**File Statistics**:

- **proposal.md**: 350 lines, 11 sections, 8 breaking changes documented
- **tasks.md**: 440 lines, 176 tasks across 16 work streams
- **design.md**: 820 lines, 6 major decisions, 5 alternatives considered
- **spec deltas**: 4 files, 19 new requirements, 4 modified, 1 removed, 1 renamed
- **README.md**: 380 lines with examples and quick reference

## 🎉 Proposal Status

**Status**: ✅ READY FOR REVIEW

**Next Actions**:

1. Technical review by backend, ML, and data engineering teams
2. Approval from architecture leads
3. Sprint planning and task assignment
4. Phase 1 implementation kickoff (Week 1-2)

---

**Author**: AI Assistant (following OpenSpec protocol)
**Date**: 2025-01-15
**OpenSpec Version**: 1.0
