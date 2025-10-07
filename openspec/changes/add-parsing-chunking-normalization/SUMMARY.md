# Summary: Clinical-Aware Parsing, Chunking & Normalization

## Overview

This proposal replaces 10 files and 975 lines of fragmented, bespoke chunking/parsing code with a unified, library-based architecture (9 files, 480 lines) that:

1. **Respects Clinical Structure**: Profile-based chunking for IMRaD, Registry, SPL, Guideline domains
2. **Delegates to Proven Libraries**: LangChain, LlamaIndex, scispaCy, syntok, unstructured (51% code reduction)
3. **Enforces GPU-Only Policy**: MinerU two-phase gate with explicit `postpdf-start` trigger
4. **Produces Span-Grounded Chunks**: Complete provenance (char offsets, section labels, intent hints, page/bbox)
5. **Preserves Table Fidelity**: HTML preservation when rectangularization uncertain

## Key Decisions

### Decision 1: ChunkerPort Protocol

**Choice**: Unified `ChunkerPort` interface with runtime registry

**Why**: Eliminates 8 custom chunker implementations, enables pluggability, supports profile-based configuration

**Impact**: Single entry point for all chunking, testable, extensible without modification

---

### Decision 2: Profile-Based Clinical Domain Awareness

**Choice**: Declarative YAML profiles (IMRaD, Registry, SPL, Guideline)

**Why**: Clinical document structure varies by domain; profiles enable domain-aware chunking without hardcoded per-source logic

**Impact**:

- IMRaD: Heading-aware boundaries, preserve figure captions
- Registry: Atomic outcomes/eligibility/AEs, no overlap
- SPL: LOINC-coded sections for RxNorm mapping
- Guideline: Recommendation units with evidence tables

---

### Decision 3: Library Delegation Strategy

**Choice**: Replace all custom code with library wrappers

| Custom Code | Library | Lines Saved |
|-------------|---------|-------------|
| 8 custom chunkers | LangChain, LlamaIndex | 420 |
| 3 custom parsers | unstructured | 415 |
| 3 sentence splitters | scispaCy, syntok | 140 |
| **Total** | | **975 lines removed** |

**Why**: Community-maintained, proven quality, continuous improvements, 51% code reduction

**Impact**: Maintenance burden reduced, leverages industry standards

---

### Decision 4: MinerU Two-Phase Gate

**Choice**: Explicit manual gate after MinerU processing

**Flow**:

```
PDF Download → MinerU (GPU) → pdf_ir_ready → HALT
              ↓ (manual or auto-sensor)
         postpdf-start → Chunking → Embedding
```

**Why**: Enables quality inspection, enforces GPU-only policy, prevents silent CPU fallbacks, maintains audit trail

**Impact**: Requires workflow change (explicit trigger), but provides control and compliance

---

### Decision 5: Docling Scope Limitation

**Choice**: Docling for HTML/XML/text only, NOT for PDF OCR

**Why**: MinerU is sole PDF path (GPU-only), Docling lacks GPU semantics, prevents accidental CPU fallbacks

**Impact**: Clear separation (MinerU for PDFs, Docling for non-OCR), policy adherence

---

### Decision 6: Complete Chunk Provenance

**Choice**: Every chunk has `doc_id`, `char_offsets`, `section_label`, `intent_hint`, `page_bbox`, `metadata`

**Why**: Enables span-grounded extraction, clinical routing, SHACL validation, reproducibility, A/B testing

**Impact**: Richer retrieval, better extraction quality, audit trail for compliance

---

## Breaking Changes

1. **ChunkingService API**: `chunk_document(document, profile)` replaces `chunk_document(document)`
2. **Chunk Schema**: `section_label`, `intent_hint`, `char_offsets` now required (previously optional or absent)
3. **PDF Gate**: `postpdf-start` trigger required (no automatic resume after MinerU)
4. **Table Handling**: HTML preservation by default (rectangularize opt-in only when confidence ≥0.8)

## Requirements Summary

### Added Requirements

| Capability | Added |
|------------|-------|
| Chunking | 6 (ChunkerPort, Profiles, Library Delegation, Provenance, Filters, Token Budget) |
| Parsing | 4 (MinerU Gate, Docling Limitation, Unstructured Wrapper, MinerU Output Format) |
| **Total** | **10** |

### Modified Requirements

| Capability | Modified |
|------------|----------|
| Chunking | 2 (Chunking Service API, Error Handling) |
| Parsing | 1 (PDF Parsing API) |
| Orchestration | 2 (PDF Two-Phase Pipeline, Job Ledger Schema) |
| Storage | 2 (Chunk Storage Schema, Chunk Indexing) |
| **Total** | **7** |

### Removed Requirements

| Capability | Removed |
|------------|---------|
| Chunking | 3 (Custom Chunkers, Source-Specific Logic, Approximate Token Counting) |
| Parsing | 2 (Custom XML Parsers, Bespoke PDF Logic) |
| **Total** | **5** |

---

## Implementation Scope

### Tasks: 240+ across 14 work streams

1. **Legacy Decommissioning** (56 tasks): Audit, dependency analysis, delegation validation, atomic deletions
2. **Foundation** (10 tasks): Install dependencies, setup scispaCy model
3. **ChunkerPort Interface** (5 tasks): Protocol definition, registry, validation
4. **Profiles** (20 tasks): IMRaD, Registry, SPL, Guideline profiles with tests
5. **Library Wrappers** (30 tasks): LangChain, LlamaIndex, scispaCy, syntok, unstructured
6. **Filters** (12 tasks): Boilerplate, references, deduplication, table HTML
7. **MinerU Gate** (12 tasks): Explicit postpdf-start, ledger schema, Dagster sensor
8. **I/O & Provenance** (15 tasks): Chunk schema, validation, failure semantics
9. **Orchestration Integration** (8 tasks): Dagster stages, gateway API
10. **Testing** (71 tasks): Unit, integration, quality validation, regression
11. **Performance** (5 tasks): Batching, caching, parallelization, benchmarking
12. **Monitoring** (10 tasks): Prometheus metrics, CloudEvents, Grafana dashboards
13. **Documentation** (10 tasks): Update comprehensive docs, create guides, runbooks
14. **Production Deployment** (6 tasks): Deploy, validate, monitor, rollback readiness

### Timeline: 6 weeks

- **Week 1-2**: Build new architecture (foundation, ChunkerPort, profiles, wrappers, atomic deletions)
- **Week 3-4**: Integration testing (all 4 profiles, MinerU gate, quality validation)
- **Week 5-6**: Production deployment (deploy, monitor, stabilize, document lessons learned)

---

## Dependencies

### New Libraries (7)

```txt
langchain-text-splitters>=0.2.0
llama-index-core>=0.10.0
scispacy>=0.5.4
syntok>=1.4.4
unstructured[local-inference]>=0.12.0
tiktoken>=0.6.0
transformers>=4.38.0
```

### Removed Dependencies

- All custom chunking/parsing code (no external dependencies removed, only internal code)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Profile tuning complexity | Start with conservative defaults, iterate based on metrics |
| scispaCy performance overhead (10x slower) | Use syntok for batches, scispaCy only when biomedical accuracy critical |
| MinerU gate workflow change | Document runbook, add Dagster UI shortcuts, implement auto-sensor |
| Library version management | Pin exact versions, test upgrades in staging, maintain golden output tests |

---

## Success Criteria

### Code Quality

- ✅ Codebase reduction: 51% (975 → 480 lines)
- ✅ Test coverage: ≥90% for new chunking code
- ✅ No legacy imports remain
- ✅ Lint clean: 0 ruff/mypy errors

### Functionality

- ✅ All 4 profiles (IMRaD, Registry, SPL, Guideline) produce expected chunk structure
- ✅ MinerU gate enforces explicit `postpdf-start` (no auto-resume)
- ✅ Docling validation guard rejects PDFs
- ✅ Chunk provenance complete: char offsets, section labels, intent hints, page/bbox

### Performance

- ✅ Chunking throughput: ≥100 docs/sec for non-PDF sources
- ✅ scispaCy vs syntok ratio: ~1:10 (syntok 10x faster)
- ✅ Downstream retrieval unchanged: Recall@10 stable, P95 latency <500ms

### Observability

- ✅ Prometheus metrics for chunking duration, chunks per document, failures
- ✅ CloudEvents for chunking lifecycle (started, completed, failed)
- ✅ CloudEvents for MinerU gate (waiting, postpdf-start triggered)
- ✅ Grafana dashboard: chunk quality, MinerU gate latency

---

## Validation Summary

- **Spec Deltas**: 4 capabilities (chunking, parsing, orchestration, storage)
- **Requirements**: 10 ADDED, 7 MODIFIED, 5 REMOVED
- **Scenarios**: 35+ detailed scenarios with GIVEN/WHEN/THEN
- **OpenSpec Validation**: `openspec validate add-parsing-chunking-normalization --strict` ✅ PASS

---

## Next Steps

1. Review proposal with team (1 week)
2. Approve for implementation
3. Create feature branch: `git checkout -b add-parsing-chunking-normalization main`
4. Execute Phase 1: Build new architecture (Week 1-2)
5. Execute Phase 2: Integration testing (Week 3-4)
6. Execute Phase 3: Production deployment (Week 5-6)

---

**Status**: Ready for review and approval
**Created**: 2025-10-07
**Change ID**: `add-parsing-chunking-normalization`
