# Clinical-Aware Parsing, Chunking & Normalization - Change Proposal

**Change ID**: `add-parsing-chunking-normalization`
**Status**: Ready for Review
**Created**: 2025-10-08
**Validation**: ✅ PASS (`openspec validate --strict`)

---

## Quick Reference

| Metric | Value |
|--------|-------|
| **Files Created** | 8 (ChunkerPort, profiles/, wrappers/) |
| **Files Deleted** | 6 (custom chunkers, parsers) |
| **Lines Added** | ~480 |
| **Lines Removed** | ~850 |
| **Net Reduction** | -370 lines (-43%) |
| **Tasks** | 240+ across 14 work streams |
| **Timeline** | 6 weeks |
| **Breaking Changes** | 4 |

---

## Overview

```yaml
# config/chunking/profiles/pmc-imrad.yaml
name: pmc-imrad
domain: literature
chunker_type: langchain_recursive
target_tokens: 450
overlap_tokens: 50
respect_boundaries:
  - heading  # Never split across IMRaD sections
  - figure_caption
  - table
sentence_splitter: huggingface  # Biomedical-aware tokenizer configured via MEDICAL_KG_SENTENCE_MODEL
preserve_tables_as_html: true
filters:
  - drop_boilerplate
  - exclude_references
  - deduplicate_page_furniture
metadata:
  section_label_source: imrad_heading
  intent_hints:
    Introduction: narrative
    Methods: narrative
    Results: outcome
    Discussion: narrative
```

### Key Innovations

1. **ChunkerPort Interface** - Single abstraction for all chunking strategies
2. **Profile-Based Chunking** - Declarative domain rules (IMRaD, Registry, SPL, Guideline)
3. **Library Delegation** - Replace 8 custom chunkers with LangChain/LlamaIndex/scispaCy
4. **MinerU Two-Phase Gate** - Explicit PDF workflow with no CPU fallbacks
5. **Span Provenance** - Every chunk has doc_id, char offsets, section labels, intent hints

---

## Problem Statement

Current codebase has **fragmented, source-specific parsing/chunking logic** causing:

- **Maintainability Crisis**: 8+ custom chunkers with overlapping logic
- **Quality Issues**: Mid-sentence splits, fractured tables, misaligned sections
- **Provenance Gaps**: Inconsistent offsets, missing section labels
- **GPU Policy Violations**: PDF processing lacks clear MinerU gate, silent CPU fallbacks

---

## Solution Architecture

```yaml
# config/chunking/profiles/spl-label.yaml
name: spl-label
domain: label
chunker_type: langchain_recursive
target_tokens: 400
overlap_tokens: 30
respect_boundaries:
  - loinc_section  # LOINC-coded sections
  - table
sentence_splitter: huggingface
preserve_tables_as_html: true
filters:
  - drop_boilerplate
  - exclude_references
metadata:
  section_label_source: loinc_code
  intent_hints:
    LOINC:34089-3: indications
    LOINC:34068-7: dosage
    LOINC:43685-7: warnings
    LOINC:34084-4: adverse_reactions
```

**Usage**:

```python
class ChunkerPort(Protocol):
    """Unified interface for all chunking strategies."""

    def chunk(self, document: Document, profile: str) -> list[Chunk]:
        """Chunk document according to named profile."""
        ...
```

**Runtime Registry** - Discovers chunker implementations by profile name

### Profile-Based Chunking

```yaml
# config/chunking/profiles/pmc-imrad.yaml
name: pmc-imrad
domain: literature
chunker_type: langchain_recursive
target_tokens: 450
overlap_tokens: 50
respect_boundaries: [heading, figure_caption, table]
sentence_splitter: scispacy
preserve_tables_as_html: true
```

**4 Profiles**:

- **IMRaD** (PMC JATS): Heading-aware, preserve figures
- **Registry** (CT.gov): Atomic outcomes, eligibility, AEs
- **SPL** (DailyMed): LOINC-coded sections
- **Guideline**: Isolate recommendation units

### Library Integration

| Functionality | Library | Purpose |
|---------------|---------|---------|
| Text Splitting | langchain-text-splitters | Structure-aware segmentation |
| Coherence | LlamaIndex node parsers | Sentence/semantic windows |
| Biomedical Sentences | scispaCy + syntok | Domain-aware segmentation |
| Tokenization | transformers / tiktoken | Qwen3 alignment |
| XML/HTML Parsing | unstructured | Safety net for oddball docs |

### MinerU Two-Phase Gate

```
PDF Download → MinerU (GPU) → pdf_ir_ready → HALT
                                            ↓
                              postpdf-start (manual trigger)
                                            ↓
                             Chunking → Embedding → Index
```

**No CPU Fallback** - MinerU fails fast if GPU unavailable

---

## Chunk Schema

```python
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    char_offsets: tuple[int, int]  # Start, end positions
    section_label: Optional[str]   # IMRaD section, LOINC code
    intent_hint: Optional[str]     # eligibility, outcome, ae, dose
    page_bbox: Optional[tuple[int, float, float, float, float]]  # page, x, y, w, h
    table_html: Optional[str]      # Preserved HTML if table
    is_unparsed_table: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
```

---

## Breaking Changes

1. **ChunkingService.chunk_document()** - Now requires `profile: str` parameter
2. **Chunk Schema** - `section_label`, `intent_hint`, `char_offsets` now required
3. **PDF Processing** - Requires explicit `postpdf-start` call (no auto-resume)
4. **Table Chunks** - Preserve HTML by default, rectangularize only when confident

---

## Migration Path (Hard Cutover)

### Week 1-2: Implementation

- Build ChunkerPort + profiles
- Integrate libraries (LangChain, LlamaIndex, scispaCy)
- **Delete legacy chunkers in same commits**

### Week 3-4: Testing

- Validate all 4 profiles (IMRaD, Registry, SPL, Guideline)
- Test MinerU two-phase gate
- Verify chunk quality (offsets, section labels, table fidelity)

### Week 5-6: Production

- Deploy with complete replacement (no legacy code)
- Monitor chunk quality metrics
- Emergency rollback: revert entire feature branch

---

## Performance Targets

### Chunking Latency (P95)

| Profile | Target | Throughput |
|---------|--------|------------|
| pmc-imrad | <2s | 30 docs/min |
| ctgov-registry | <1s | 60 docs/min |
| spl-loinc | <1.5s | 40 docs/min |
| guideline | <1s | 60 docs/min |

### MinerU Performance

- **Latency**: P95 <30s per PDF (20-30 pages)
- **Throughput**: 2-3 PDFs/second (GPU)
- **Success Rate**: >95%
- **GPU Utilization**: 60-80%

### Quality Metrics

- **Token Overflow Rate**: <1% across all profiles
- **Section Label Coverage**: >90% of chunks
- **Table Preservation Rate**: 100% when uncertainty high

---

## API Changes

### REST

```http
POST /v1/ingest/clinicaltrials
Content-Type: application/vnd.api+json

{
  "data": {
    "type": "IngestionRequest",
    "attributes": {
      "identifiers": ["NCT04267848"],
      "chunking_profile": "ctgov-registry",
      "options": {
        "preserve_tables_html": true,
        "sentence_splitter": "scispacy"
      }
    }
  }
}
```

### GraphQL

```graphql
mutation IngestClinicalTrial($input: IngestionInput!) {
  startIngestion(input: $input) {
    jobId
    chunkingProfile
    estimatedChunks
  }
}
```

---

## Monitoring & Observability

### Prometheus Metrics

```python
CHUNKING_DURATION = Histogram(
    "medicalkg_chunking_duration_seconds",
    "Chunking duration per profile",
    ["profile", "source"]
)

TOKEN_OVERFLOW_RATE = Counter(
    "medicalkg_chunk_token_overflow_total",
    "Token budget overflows",
    ["profile"]
)

MINERU_FAILURES = Counter(
    "medicalkg_mineru_failures_total",
    "MinerU processing failures",
    ["error_type"]
)
```

### CloudEvents

```json
{
  "type": "com.medical-kg.chunking.completed",
  "data": {
    "profile": "pmc-imrad",
    "chunk_count": 45,
    "duration_seconds": 1.2,
    "token_overflows": 0
  }
}
```

### Grafana Dashboards

- Chunking Latency by Profile (P50, P95, P99)
- Token Overflow Rate (% of chunks)
- Table Preservation Rate (%)
- MinerU Success Rate (%)
- Profile Usage Distribution
- Section Label Coverage (%)

---

## Testing Strategy

### Test Coverage

- **Unit Tests**: 50+ (ChunkerPort, profiles, libraries, filters)
- **Integration Tests**: 21 (end-to-end per profile, PDF two-phase)
- **Performance Tests**: Latency benchmarks, load tests, soak tests
- **Contract Tests**: REST/GraphQL/gRPC API compatibility
- **Table Tests**: HTML preservation, rectangularization decisions

### Quality Validation

- Chunk offsets accuracy (manual inspection of 100 chunks)
- Section labels match expected structure (IMRaD/LOINC/registry)
- No mid-sentence splits (sample 100 chunks)
- Table HTML preserved when uncertainty high

---

## Dependencies Added

```txt
# Parsing & Chunking
langchain-text-splitters>=0.2.0
llama-index-core>=0.10.0
syntok>=1.4.4
unstructured[local-inference]>=0.12.0
tiktoken>=0.6.0
transformers>=4.38.0
pydantic>=2.6.0
```

---

## Rollback Procedures

### Trigger Conditions

**Automated**:

- Token overflow rate >10% for >15 minutes
- Chunking latency P95 >5s for >10 minutes
- Section label coverage <80% for >15 minutes

**Manual**:

- Critical quality issues (mid-sentence splits, corrupted tables)
- Downstream extraction failures

### Rollback Steps

```bash
# 1. Revert feature branch
git revert <feature-branch-commit-sha>

# 2. Redeploy previous version
kubectl rollout undo deployment/chunking-service

# 3. Validate restoration (5 minutes)
# Check metrics return to baseline

# RTO: 5 minutes (target)
```

---

## Benefits

✅ **Maintainability**: Single ChunkerPort interface, library delegation
✅ **Clinical Fidelity**: Profile-based chunking respects domain structure
✅ **Provenance**: Complete tracking (doc_id, offsets, section labels, intent)
✅ **GPU Policy**: Explicit MinerU gate, no CPU fallbacks
✅ **Span-Grounded**: Precise offsets enable downstream extraction
✅ **Code Reduction**: 43% reduction in parsing/chunking code

---

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Profile tuning complexity | Start conservative, tune based on metrics |
| scispaCy overhead | Use syntok for high-volume, scispaCy when needed |
| MinerU gate training | Document workflow in runbook, add Dagster shortcuts |
| Token overflow | Monitor metrics, tune profile token budgets |

---

## Files Modified

### Created

- `src/Medical_KG_rev/services/chunking/port.py` (ChunkerPort interface)
- `src/Medical_KG_rev/services/chunking/profiles/` (4 profile YAML files)
- `src/Medical_KG_rev/services/chunking/wrappers/` (library wrappers)
- `tests/chunking/test_profiles.py` (profile tests)

### Deleted

- `src/Medical_KG_rev/services/chunking/custom_splitters.py` (420 lines)
- `src/Medical_KG_rev/services/parsing/pdf_parser.py` (180 lines)
- `src/Medical_KG_rev/services/parsing/xml_parser.py` (95 lines)
- `src/Medical_KG_rev/services/parsing/sentence_splitters.py` (140 lines)

### Modified

- `src/Medical_KG_rev/gateway/rest/routes/ingestion.py` (add chunking_profile param)
- `src/Medical_KG_rev/orchestration/orchestrator.py` (PDF two-phase gate)
- All adapter `split_document()` calls → delegate to ChunkerPort

---

## Next Steps

1. ✅ **Review & Approval** - Stakeholder review of this proposal
2. ⏳ **Implementation** - 6 weeks (Week 1-2: build, Week 3-4: test, Week 5-6: deploy)
3. ⏳ **Validation** - Monitor metrics for 2 weeks post-deployment
4. ⏳ **Iteration** - Tune profiles based on downstream quality

---

## Document Index

- **proposal.md** - Why, what changes, impact, benefits
- **tasks.md** - 240+ implementation tasks across 14 work streams
- **design.md** - Technical decisions, alternatives, architecture
- **specs/chunking/spec.md** - 6 ADDED, 2 MODIFIED, 3 REMOVED requirements
- **specs/parsing/spec.md** - 4 ADDED, 1 MODIFIED, 2 REMOVED requirements
- **specs/orchestration/spec.md** - 2 MODIFIED requirements (PDF gate)
- **specs/storage/spec.md** - 2 MODIFIED requirements (chunk schema)

---

**Status**: ✅ Ready for stakeholder review and approval
