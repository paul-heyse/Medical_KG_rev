# Proposal: Clinical-Aware Parsing, Chunking & Normalization

## Why

The current codebase contains **fragmented, source-specific parsing and chunking logic** scattered across multiple services, with inconsistent handling of clinical document structure. This creates:

1. **Maintainability Crisis**: 8+ custom chunkers with overlapping logic, no clear boundaries
2. **Quality Issues**: Chunks split mid-sentence, tables fractured, clinical sections misaligned
3. **Provenance Gaps**: Inconsistent offset tracking, missing section labels, lost table context
4. **GPU Policy Violations**: PDF processing lacks clear MinerU gate, silent CPU fallbacks exist
5. **Library Duplication**: Bespoke implementations of functionality available in LangChain, LlamaIndex, HuggingFace

This proposal **replaces fragmented parsing/chunking with a unified, library-based architecture** that:

- Respects clinical document structure (IMRaD, LOINC sections, registry outcomes, guideline recommendations)
- Preserves the **manual PDF two-phase gate** (MinerU → `pdf_ir_ready` → explicit `postpdf-start`)
- Delegates to proven libraries (langchain-text-splitters, LlamaIndex, HuggingFace, unstructured)
- Enforces **GPU-only policy** for MinerU (no CPU fallbacks)
- Produces **span-grounded chunks** with precise offsets, section labels, and table fidelity

## What Changes

### Core Architecture

**NEW: ChunkerPort Interface** - Single abstraction for all chunking strategies

- Replaces 8+ bespoke chunkers with pluggable implementations
- Input: IR Document (blocks, tables, sections) + named profile (e.g., "pmc-imrad", "ctgov-registry")
- Output: Chunks with offsets, section labels, intent hints (eligibility vs outcome vs AE)

**NEW: Profile-Based Chunking** - Declarative domain rules

- **IMRaD Profile** (PMC JATS): Heading-aware, preserve figure captions, mid-sized narrative chunks
- **Registry Profile** (CT.gov): Atomic outcomes, eligibility, AEs; keep effect pairs together
- **SPL Profile** (DailyMed): LOINC-coded sections (Indications, Dosage, Warnings, AEs)
- **Guideline Profile**: Isolate recommendation units (statement, strength, grade)

**NEW: Filter Chain System** - Composable normalization without evidence loss

- Drop boilerplate (headers/footers), exclude "References" sections
- De-duplicate repeated page furniture
- **Preserve table chunks verbatim** (including HTML) when rectangularization uncertain

**NEW: Library Integration Layer**

- **langchain-text-splitters**: Default for structure-aware segmentation (recursive character/token)
- **LlamaIndex node parsers**: Sentence/semantic-window chunking for coherence
- **HuggingFace models + syntok**: Biomedical-aware sentence segmentation using transformer models
- **tokenizers (HF) / tiktoken**: Budget enforcement aligned with Qwen3 embedding model
- **unstructured**: Safety net for non-PDF HTML and oddball documents

### MinerU Two-Phase Gate (Hardened)

**MODIFIED: PDF Processing Flow** - Explicit gate enforcement

- MinerU runs on GPU, produces Markdown/JSON + structured IR with page/bbox maps
- On success: Ledger updated to `pdf_ir_ready`, pipeline **HALTS**
- Manual resume: Explicit `postpdf-start` trigger resumes chunking/embedding
- On failure: Ledger marked `mineru_failed`, **NO CPU FALLBACK**, pipeline aborts

**REMOVED: Docling from PDF Path** - Keep for non-OCR contexts only

- Docling remains available for HTML/XML/text normalization (non-GPU)
- Docling **NOT wired** into GPU-mandatory PDF path (honors fail-fast policy)

### Legacy Code Decommissioning

**DELETE**:

- `src/Medical_KG_rev/services/chunking/custom_splitters.py` (8 custom chunkers, 420 lines)
- `src/Medical_KG_rev/services/parsing/pdf_parser.py` (bespoke PDF logic, replaced by MinerU)
- `src/Medical_KG_rev/services/parsing/xml_parser.py` (replaced by unstructured)
- All `*.split_document()` methods scattered across adapters (15 occurrences)
- Custom sentence splitters (3 implementations, replaced by HuggingFace models/syntok)

**REPLACE**:

- Custom recursive splitters → `langchain_text_splitters.RecursiveCharacterTextSplitter`
- Custom sentence splitters → Hugging Face tokenizer-backed segmenter or `syntok.segment`
- Custom tokenizers → `transformers.AutoTokenizer` (aligned with Qwen3)
- Bespoke XML parsing → `unstructured.partition_xml`

### Breaking Changes

- **BREAKING**: `ChunkingService.chunk_document()` signature changes to accept `profile: str` parameter
- **BREAKING**: Chunks now require `section_label`, `intent_hint`, and `char_offsets` (non-optional)
- **BREAKING**: PDF processing via `Orchestrator.submit_pdf_job()` now requires explicit `postpdf-start` call (no auto-resume)
- **BREAKING**: Table chunks preserve HTML by default; `rectangularize=True` opt-in only when confidence high

## Impact

### Affected Capabilities

- **chunking** (spec delta: 6 ADDED, 2 MODIFIED, 3 REMOVED requirements)
- **parsing** (spec delta: 4 ADDED, 1 MODIFIED, 2 REMOVED requirements)
- **orchestration** (spec delta: 2 MODIFIED requirements for PDF gate)
- **storage** (spec delta: 1 MODIFIED requirement for chunk schema)

### Affected Code

- **Gateway**: `/v1/ingest/{source}` endpoint (add `chunking_profile` parameter)
- **Orchestration**: `Orchestrator.submit_pdf_job()`, PDF two-phase gate logic
- **Services**: Complete rewrite of chunking service, delete custom parsers
- **Adapters**: Remove `.split_document()` methods, delegate to `ChunkerPort`
- **Tests**: Delete 8 custom chunker tests, add 12 profile-based chunker tests

### Codebase Impact

- **Lines Removed**: ~850 lines (custom chunkers, parsers, sentence splitters)
- **Lines Added**: ~480 lines (ChunkerPort, profiles, library wrappers)
- **Net Reduction**: 370 lines (43% reduction in parsing/chunking code)
- **Files Deleted**: 6 (custom_splitters.py, pdf_parser.py, xml_parser.py, + 3 test files)
- **Files Added**: 8 (chunker_port.py, profiles/, library wrappers/, + 5 test files)

### Dependencies Added

```txt
# Parsing & Chunking
langchain-text-splitters>=0.2.0
llama-index-core>=0.12.0,<0.12.1
transformers>=4.38.0  # Hugging Face tokenizers for segmentation and Qwen3 alignment
syntok>=1.4.4  # Fast sentence splitter
unstructured[local-inference]>=0.12.0
tiktoken>=0.6.0  # OpenAI tokenizer
```

### Migration Path

**Hard Cutover** - No legacy compatibility:

1. **Week 1-2**: Implement new architecture
   - Build ChunkerPort interface + profiles
   - Integrate langchain-text-splitters, LlamaIndex, HuggingFace models
   - Delete custom chunkers in same commits

2. **Week 3-4**: Integration testing
   - Validate all 4 profiles (IMRaD, Registry, SPL, Guideline)
   - Test MinerU two-phase gate with explicit `postpdf-start`
   - Verify chunk quality (offsets, section labels, table fidelity)

3. **Week 5-6**: Production deployment
   - Deploy with complete replacement (no legacy code remains)
   - Monitor chunk quality metrics, section alignment
   - Emergency rollback: revert entire feature branch

### Benefits

- **Maintainability**: Single ChunkerPort interface, pluggable strategies, library delegation
- **Clinical Fidelity**: Profile-based chunking respects IMRaD, LOINC, registry, guideline structure
- **Provenance**: Every chunk carries doc_id, char offsets, page/bbox, section labels, intent hints
- **GPU Policy**: Explicit MinerU gate, no CPU fallbacks, fail-fast semantics
- **Span-Grounded**: Precise offsets enable downstream extraction, SHACL validation, graph writes
- **Library Standards**: Delegation to LangChain, LlamaIndex, HuggingFace reduces bespoke code by 43%

### Risks

- **Profile Tuning**: Initial profiles may need iteration based on downstream extraction quality
- **HuggingFace Model Overhead**: Transformer-based sentence segmentation heavier than syntok; profile selection affects batch throughput
- **MinerU Gate**: Requires team training on explicit `postpdf-start` workflow (no auto-resume)

### Mitigation

- Start with conservative profile defaults, tune based on retrieval/extraction metrics
- Use syntok for high-volume batches, HuggingFace models only when biomedical sentence boundaries critical
- Document `postpdf-start` workflow in runbook, add Dagster UI shortcuts for common cases

---

## Observability & Monitoring

### Prometheus Metrics

```python
# Chunking metrics
CHUNKING_DURATION = Histogram(
    "medicalkg_chunking_duration_seconds",
    "Chunking duration per profile",
    ["profile", "source"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

CHUNK_COUNT = Histogram(
    "medicalkg_chunks_per_document",
    "Number of chunks produced per document",
    ["profile", "source"],
    buckets=[5, 10, 20, 50, 100, 200]
)

TOKEN_OVERFLOW_RATE = Counter(
    "medicalkg_chunk_token_overflow_total",
    "Token budget overflows per profile",
    ["profile"]
)

TABLE_PRESERVATION_RATE = Gauge(
    "medicalkg_table_preservation_rate",
    "% of tables preserved as HTML",
    ["profile"]
)

# Parsing metrics
MINERU_DURATION = Histogram(
    "medicalkg_mineru_duration_seconds",
    "MinerU PDF processing duration",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0]
)

MINERU_FAILURES = Counter(
    "medicalkg_mineru_failures_total",
    "MinerU processing failures",
    ["error_type"]  # gpu_unavailable, oom, timeout, parse_error
)
```

### CloudEvents

```json
{
  "specversion": "1.0",
  "type": "com.medical-kg.chunking.completed",
  "source": "/chunking-service",
  "id": "chunk-job-abc123",
  "time": "2025-10-08T14:30:00Z",
  "data": {
    "job_id": "job-abc123",
    "doc_id": "PMC123:chunk_5",
    "profile": "pmc-imrad",
    "chunk_count": 45,
    "duration_seconds": 1.2,
    "token_overflows": 0,
    "tables_preserved": 3
  }
}
```

### Grafana Dashboard Panels

1. **Chunking Latency by Profile**: Line chart (P50, P95, P99) per profile
2. **Token Overflow Rate**: Gauge showing % of chunks exceeding token budget
3. **Table Preservation Rate**: Gauge showing % of tables kept as HTML
4. **MinerU Success Rate**: Gauge showing PDF processing success rate
5. **Profile Usage Distribution**: Pie chart of profile selection frequency
6. **Section Label Coverage**: Gauge showing % of chunks with section labels

---

## Performance Targets

### Chunking Performance

| Profile | Target Latency (P95) | Throughput | Token Overflow Rate |
|---------|---------------------|------------|---------------------|
| pmc-imrad | <2s per document | 30 docs/min | <1% |
| ctgov-registry | <1s per document | 60 docs/min | <0.5% |
| spl-loinc | <1.5s per document | 40 docs/min | <1% |
| guideline | <1s per document | 60 docs/min | <1% |

### MinerU Performance

- **Latency**: P95 <30s per PDF (20-30 pages)
- **Throughput**: 2-3 PDFs/second (GPU-accelerated)
- **Success Rate**: >95% (excluding malformed PDFs)
- **GPU Utilization**: 60-80% during processing

---

## API Integration

### REST API Changes

```http
POST /v1/ingest/{source}
Content-Type: application/vnd.api+json

{
  "data": {
    "type": "IngestionRequest",
    "attributes": {
      "identifiers": ["NCT04267848"],
      "chunking_profile": "ctgov-registry",
      "options": {
        "preserve_tables_html": true,
        "sentence_splitter": "huggingface"
      }
    }
  }
}
```

### GraphQL API Changes

```graphql
mutation IngestClinicalTrial($input: IngestionInput!) {
  startIngestion(input: $input) {
    jobId
    chunkingProfile
    estimatedChunks
  }
}

input IngestionInput {
  source: String!
  identifiers: [String!]!
  chunkingProfile: String = "default"
  options: ChunkingOptions
}

input ChunkingOptions {
  preserveTablesHtml: Boolean = true
  sentenceSplitter: String = "syntok"
  customTokenBudget: Int
}
```

---

## Rollback Procedures

### Rollback Trigger Conditions

**Automated Rollback**:

- Token overflow rate >10% for >15 minutes
- Chunking latency P95 >5s for >10 minutes
- Section label coverage <80% for >15 minutes
- MinerU failure rate >20% for >10 minutes

**Manual Rollback**:

- Critical quality issues (chunks split mid-sentence, tables corrupted)
- Downstream extraction failures caused by chunk quality
- Team decision based on user feedback

### Rollback Steps

```bash
# Step 1: Immediate traffic shift (if gradual rollout was used)
# Revert feature branch commit
git revert <feature-branch-commit-sha>

# Step 2: Redeploy previous version
kubectl rollout undo deployment/chunking-service

# Step 3: Validate baseline restoration (5 minutes)
# Check metrics return to baseline
# - Chunking latency P95 <1s (baseline)
# - Token overflow rate <2%
# - Section label coverage >90%

# Step 4: Post-incident analysis (1 hour)
# Gather logs, metrics, example chunks
# Identify root cause
# Create incident report

# Step 5: Fix and re-deploy (1-2 days)
# Fix identified issues
# Re-test in staging
# Schedule new production deployment
```

### Recovery Time Objective (RTO)

**Target RTO**: 5 minutes (revert commit + redeploy)
**Maximum RTO**: 15 minutes (if requires config changes)
