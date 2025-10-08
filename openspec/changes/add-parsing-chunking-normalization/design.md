# Design: Clinical-Aware Parsing, Chunking & Normalization

## Context

The existing chunking/parsing architecture has evolved organically, resulting in:

- **8 custom chunking implementations** with overlapping logic
- **3 separate sentence splitters** (syntactic, semantic, biomedical-aware)
- **PDF parsing ambiguity** (MinerU vs Docling vs bespoke logic)
- **Inconsistent provenance** (offsets, section labels, table handling vary by source)
- **No clear clinical structure awareness** (IMRaD, LOINC, registry outcomes not explicitly modeled)

This creates technical debt and quality issues: chunks split mid-sentence, tables fractured, clinical boundaries ignored, provenance incomplete. The system needs a **unified architecture** that:

1. Respects clinical document structure via **profiles**
2. Delegates to proven libraries (LangChain, LlamaIndex, HuggingFace)
3. Enforces **MinerU two-phase gate** for PDFs (no CPU fallbacks)
4. Produces **span-grounded chunks** with complete provenance

## Goals / Non-Goals

### Goals

- **Single Chunking Interface**: `ChunkerPort` protocol for all strategies
- **Profile-Based Chunking**: Declarative clinical domain rules (IMRaD, Registry, SPL, Guideline)
- **Library Delegation**: Replace 8 custom chunkers with LangChain/LlamaIndex/HuggingFace wrappers
- **MinerU Gate Enforcement**: Explicit two-phase PDF workflow (`pdf_downloaded` → `pdf_ir_ready` → `postpdf-start`)
- **Span Provenance**: Every chunk has `doc_id`, `char_offsets`, `section_label`, `intent_hint`, `page_bbox`
- **Table Fidelity**: Preserve HTML when rectangularization uncertain
- **Codebase Reduction**: ≥40% reduction in chunking/parsing code

### Non-Goals

- **Not** adding new embedding or retrieval stages (separate proposals)
- **Not** changing IR schema (Document/Block/Section structure remains)
- **Not** supporting Docling for PDF OCR (MinerU is the only prod path)
- **Not** preserving legacy chunking APIs (hard cutover, no compatibility shims)

## Decisions

### Decision 1: ChunkerPort Protocol + Runtime Registry

**What**: Single interface for all chunking strategies, discovered at runtime

```python
# src/Medical_KG_rev/services/chunking/port.py
from typing import Protocol

class ChunkerPort(Protocol):
    """Unified interface for all chunking strategies."""

    def chunk(self, document: Document, profile: str) -> list[Chunk]:
        """
        Chunk a document according to a named profile.

        Args:
            document: IR Document with blocks, sections, tables
            profile: Named profile (e.g., "pmc-imrad", "ctgov-registry")

        Returns:
            List of chunks with complete provenance
        """
        ...

# Runtime registry
CHUNKER_REGISTRY: dict[str, Type[ChunkerPort]] = {}

def register_chunker(name: str, implementation: Type[ChunkerPort]):
    """Register a chunker implementation."""
    CHUNKER_REGISTRY[name] = implementation

def get_chunker(profile: str) -> ChunkerPort:
    """Get chunker for a profile."""
    profile_config = load_profile(profile)
    chunker_class = CHUNKER_REGISTRY[profile_config.chunker_type]
    return chunker_class(profile_config)
```

**Why**:

- **Pluggability**: Add new strategies without modifying existing code
- **Testability**: Mock `ChunkerPort` for unit tests, swap implementations easily
- **Discovery**: Runtime registry allows dynamic configuration per profile
- **Type Safety**: Protocol ensures all implementations have correct signature

**Alternatives Considered**:

- **Abstract Base Class**: More rigid, harder to mock, less idiomatic for protocols
- **Function-based**: Would work but lose stateful benefits (cached tokenizers, etc.)

**Trade-offs**:

- **+** Clean separation of concerns, easy to test
- **+** Profiles can mix and match chunker implementations
- **-** Runtime registry requires initialization order discipline

---

### Decision 2: Profile-Based Clinical Domain Awareness

**What**: Declarative YAML profiles that encode domain-specific chunking rules

```yaml
# config/chunking/profiles/pmc-imrad.yaml
name: pmc-imrad
domain: literature
chunker_type: langchain_recursive  # or "llamaindex_sentence_window"
target_tokens: 450
overlap_tokens: 50
respect_boundaries:
  - heading  # Never split across IMRaD section boundaries
  - figure_caption
  - table
sentence_splitter: huggingface  # or "syntok"
preserve_tables_as_html: true
filters:
  - drop_boilerplate
  - exclude_references
  - deduplicate_page_furniture
metadata:
  section_label_source: "imrad_heading"  # Where to get section labels
  intent_hints:
    Introduction: narrative
    Methods: narrative
    Results: outcome
    Discussion: narrative
```

**Why**:

- **Clinical Structure**: Profiles explicitly model IMRaD, LOINC sections, registry outcomes, guideline recommendations
- **Reproducibility**: Same profile → same chunking behavior across runs
- **Experimentation**: Tune profiles without code changes
- **Multi-Source**: Single document can be chunked with different profiles for different use cases

**Alternatives Considered**:

- **Hardcoded per-source logic**: Less flexible, harder to tune, couples chunking to adapters
- **Single global config**: Doesn't respect domain-specific structure (PMC vs CT.gov very different)

**Trade-offs**:

- **+** Declarative, easy to understand and tune
- **+** Profiles can be versioned and tested independently
- **-** Requires profile management (YAML files, validation)
- **-** Initial profiles may need iteration based on downstream quality

---

### Decision 3: Library Delegation Strategy

**What**: Replace all custom chunkers with thin wrappers around proven libraries

| Library | Use Case | Replaces |
|---------|----------|----------|
| **langchain-text-splitters** | Default recursive character/token splitting | `CustomSplitter`, `RecursiveSplitter` |
| **LlamaIndex node parsers** | Sentence/semantic-window chunking for coherence | `SemanticSplitter`, `CoherenceSplitter` |
| **HuggingFace** | Biomedical-aware sentence segmentation | `BiomedicalSentenceSplitter` |
| **syntok** | Fast, robust sentence splitting (non-biomedical) | `SimpleSentenceSplitter` |
| **transformers / tiktoken** | Token counting aligned with Qwen3 | Custom tokenizers |
| **unstructured** | XML/HTML parsing (non-PDF) | `XMLParser`, `HTMLParser` |

**Implementation Example** (LangChain wrapper):

```python
# src/Medical_KG_rev/services/chunking/wrappers/langchain_splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

class LangChainChunker:
    def __init__(self, profile: Profile):
        self.profile = profile
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=profile.target_tokens * 4,  # Approximate tokens→chars
            chunk_overlap=profile.overlap_tokens * 4,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens using Qwen tokenizer."""
        return len(self.tokenizer.encode(text))

    def chunk(self, document: Document, profile: str) -> list[Chunk]:
        """Chunk document respecting profile boundaries."""
        # 1. Apply filters
        filtered_blocks = self._apply_filters(document.blocks)

        # 2. Group by respect_boundaries (e.g., heading, table)
        boundary_groups = self._group_by_boundaries(filtered_blocks)

        # 3. Chunk each group independently
        chunks = []
        for group in boundary_groups:
            group_text = "\n\n".join(block.text for block in group)
            split_texts = self.splitter.split_text(group_text)

            # 4. Map back to char offsets
            for split_text in split_texts:
                chunk = self._create_chunk(split_text, group)
                chunks.append(chunk)

        return chunks
```

**Why**:

- **Proven Quality**: LangChain/LlamaIndex battle-tested on millions of documents
- **Community Support**: Active maintenance, bug fixes, feature additions
- **Reduced Maintenance**: We don't maintain splitting logic, only wrappers
- **Consistency**: Industry-standard behavior for splitting, not bespoke rules

**Alternatives Considered**:

- **Keep custom chunkers, add tests**: Still high maintenance burden, no community support
- **Single library (e.g., only LangChain)**: Doesn't cover all use cases (sentence windows need LlamaIndex)

**Trade-offs**:

- **+** Massive reduction in code (43% fewer lines)
- **+** Leverage community improvements without effort
- **-** Dependency on external libraries (version management)
- **-** Some profile-specific logic still needed (boundary detection)

---

### Decision 4: MinerU Two-Phase Gate Enforcement

**What**: Explicit, manual gate for PDF processing with no automatic resume

```python
# PDF processing flow
def submit_pdf_job(job_id: str, pdf_url: str):
    # 1. Download PDF
    pdf_content = download_pdf(pdf_url)
    ledger.update(job_id, pdf_downloaded=True)

    # 2. Submit to MinerU (GPU-only)
    try:
        mineru_result = mineru_service.process_pdf(pdf_content)
        ir_document = convert_mineru_to_ir(mineru_result)
        store_ir(job_id, ir_document)
        ledger.update(job_id, pdf_ir_ready=True)
    except GpuNotAvailableError:
        ledger.update(job_id, status="mineru_failed", error="GPU unavailable")
        raise  # FAIL FAST, no CPU fallback
    except MineruProcessingError as e:
        ledger.update(job_id, status="mineru_failed", error=str(e))
        raise

    # 3. HALT - wait for explicit postpdf-start trigger
    # (No automatic progression to chunking/embedding)

def trigger_postpdf_start(job_id: str):
    """Explicitly resume PDF processing after MinerU."""
    # Validate ledger state
    entry = ledger.get(job_id)
    if not entry.pdf_ir_ready:
        raise ValueError(f"Job {job_id} not ready for postpdf-start (state: {entry.status})")

    # Mark postpdf started
    ledger.update(job_id, postpdf_start_triggered=True)

    # Resume pipeline: chunking → embedding → indexing
    resume_pipeline(job_id, start_stage="chunk")
```

**Gateway API**:

```http
POST /v1/jobs/{job_id}/postpdf-start
Authorization: Bearer {token}

Response:
{
  "data": {
    "type": "Job",
    "id": "job-abc123",
    "attributes": {
      "status": "chunking",
      "postpdf_start_triggered_at": "2025-10-07T14:30:00Z"
    }
  }
}
```

**Why**:

- **Manual Control**: Team can inspect MinerU output before proceeding (quality check)
- **GPU Policy**: Explicit fail-fast on GPU unavailability, no silent CPU fallbacks
- **Audit Trail**: Ledger records who/when triggered `postpdf-start`
- **Failure Isolation**: MinerU failures don't cascade to chunking/embedding

**Alternatives Considered**:

- **Automatic resume**: Simpler but loses control, no quality check opportunity
- **Approval workflow**: Could add but adds complexity; explicit trigger sufficient

**Trade-offs**:

- **+** Full control over PDF processing progression
- **+** Prevents silent quality degradation (bad OCR → bad chunks)
- **-** Requires manual step (or scheduled automation)
- **-** Potential for jobs to stall if `postpdf-start` forgotten

**Mitigation**:

- Add Dagster sensor that auto-triggers `postpdf-start` after configurable delay (e.g., 5 minutes)
- Add Dagster UI shortcuts for bulk `postpdf-start` operations
- Alert on jobs stuck in `pdf_ir_ready` for >1 hour

---

### Decision 5: Docling Scope Limitation (Non-PDF Only)

**What**: Keep Docling for HTML/XML/text normalization, **exclude from PDF OCR path**

**Docling Allowed Use Cases**:

1. HTML guideline normalization (non-OCR)
2. XML parsing as alternative to `unstructured` (developer choice)
3. Local experiments in non-GPU environments
4. Text document cleaning (non-PDF)

**Docling Forbidden Use Cases**:

1. ❌ PDF OCR (MinerU is the only prod path)
2. ❌ CPU fallback for MinerU failures
3. ❌ Any GPU-mandatory processing stage

**Implementation** (validation guard):

```python
# src/Medical_KG_rev/services/parsing/docling_parser.py
class DoclingParser:
    ALLOWED_FORMATS = ["html", "xml", "txt"]

    def parse(self, content: bytes, format: str) -> Document:
        if format == "pdf":
            raise ValueError(
                "Docling cannot be used for PDF parsing in production. "
                "Use MinerU for PDF OCR (GPU-only policy)."
            )

        if format not in self.ALLOWED_FORMATS:
            raise ValueError(f"Docling only supports {self.ALLOWED_FORMATS}, got {format}")

        # Proceed with HTML/XML/text parsing
        ...
```

**Why**:

- **GPU Policy Adherence**: Prevents accidental CPU fallbacks for PDFs
- **Clear Boundaries**: Docling for non-OCR, MinerU for OCR (no ambiguity)
- **Flexibility**: Docling still useful for HTML/XML, just not PDFs

**Alternatives Considered**:

- **Remove Docling entirely**: Too restrictive, loses value for HTML/XML use cases
- **Allow Docling for PDFs in dev**: Creates confusion, risk of prod leakage

**Trade-offs**:

- **+** Clear separation: MinerU (PDFs) vs Docling (HTML/XML/text)
- **+** Prevents policy violations
- **-** Requires documentation to clarify Docling scope

---

### Decision 6: Chunk Schema with Complete Provenance

**What**: Every chunk carries complete metadata for span-grounded extraction

```python
# src/Medical_KG_rev/services/chunking/models.py
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    """A chunk of text from a document with complete provenance."""

    # Identity
    chunk_id: str = Field(description="Unique chunk identifier")
    doc_id: str = Field(description="Source document ID")

    # Content
    text: str = Field(description="Chunk text content")
    char_offsets: tuple[int, int] = Field(description="Start/end char offsets in source document")

    # Clinical Structure
    section_label: str = Field(description="Clinical section (e.g., 'Methods', 'LOINC:34089-3')")
    intent_hint: str = Field(description="Chunk purpose: 'narrative', 'eligibility', 'outcome', 'ae', 'dose'")

    # PDF Provenance (optional)
    page_bbox: dict | None = Field(default=None, description="Page number + bounding box for PDFs")

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata: source_system, chunking_profile, chunker_version, created_at"
    )

    # Table Handling
    is_unparsed_table: bool = Field(default=False, description="True if table HTML preserved due to uncertainty")
    table_html: str | None = Field(default=None, description="Original table HTML if unparsed")
```

**Required Metadata** (always present):

- `source_system`: "pmc", "ctgov", "dailymed", "guideline"
- `chunking_profile`: "pmc-imrad", "ctgov-registry", etc.
- `chunker_version`: "langchain-v0.2.0", "llamaindex-v0.10.0"
- `created_at`: ISO 8601 timestamp

**Why**:

- **Span-Grounded Extraction**: Char offsets enable linking extractions to source text
- **Clinical Routing**: `section_label` and `intent_hint` guide retrieval and extraction
- **Reproducibility**: `chunking_profile` and `chunker_version` enable A/B testing
- **Audit Trail**: Complete provenance for compliance and debugging

**Alternatives Considered**:

- **Minimal schema** (just text + doc_id): Loses provenance, breaks span grounding
- **Separate metadata store**: Adds complexity, risks orphaned records

**Trade-offs**:

- **+** Complete provenance enables advanced features (span highlighting, SHACL validation)
- **+** Clinical structure (section_label, intent_hint) improves retrieval relevance
- **-** Larger storage footprint (metadata adds ~200 bytes per chunk)

---

## Risks / Trade-offs

### Risk 1: Profile Tuning Complexity

**Risk**: Initial profiles may not produce optimal chunk quality for downstream tasks

**Mitigation**:

- Start with conservative defaults (450 tokens, 50 overlap, respect major boundaries)
- Monitor downstream metrics: retrieval Recall@10, extraction F1, KG completeness
- Iterate profiles based on quantitative feedback
- Version profiles in Git (e.g., `pmc-imrad-v2.yaml`) for reproducibility

### Risk 2: HuggingFace Model Performance Overhead

**Risk**: Transformer-based sentence segmentation is 5-10x slower than syntok, may bottleneck high-volume ingestion

**Mitigation**:

- Use syntok for high-throughput batches (e.g., bulk PMC ingestion)
- Use HuggingFace models only when biomedical sentence boundaries are critical (e.g., Methods→Results transitions)
- Profile-level flag: `sentence_splitter: syntok` vs `sentence_splitter: huggingface`
- Benchmark both on representative corpus, choose based on quality/speed trade-off

### Risk 3: MinerU Gate Requires Workflow Changes

**Risk**: Team must learn explicit `postpdf-start` workflow (not automatic)

**Mitigation**:

- Document workflow in runbook: `docs/runbooks/mineru-two-phase-gate.md`
- Add Dagster UI shortcuts for common operations
- Implement auto-trigger sensor (configurable delay, e.g., 5 minutes after `pdf_ir_ready`)
- Alert on jobs stuck in `pdf_ir_ready` state for >1 hour

### Risk 4: Library Version Management

**Risk**: LangChain/LlamaIndex/HuggingFace may introduce breaking changes in future versions

**Mitigation**:

- Pin exact versions in `requirements.txt` (no `^` or `~`)
- Test upgrades in staging before production
- Maintain test suite with golden outputs (detect regressions immediately)
- Version chunker implementations alongside profiles (e.g., `langchain-v0.2.0`)

---

## Migration Plan

### Phase 1: Build New Architecture (Week 1-2)

**Day 1-3**: Foundation

- [ ] Install dependencies (langchain-text-splitters, llama-index-core, sentence-transformers, syntok, unstructured)
- [ ] Create `ChunkerPort` interface + runtime registry
- [ ] Define `Chunk` Pydantic model with provenance fields

**Day 4-7**: Library Wrappers

- [ ] Implement `LangChainChunker` (recursive character/token splitting)
- [ ] Implement `LlamaIndexChunker` (sentence window)
- [ ] Implement `HuggingFaceSentenceSegmenter`
- [ ] Implement `SyntokSentenceSegmenter`
- [ ] Implement `UnstructuredParser` (XML/HTML)

**Day 8-10**: Profiles

- [ ] Create 4 profile YAML files (IMRaD, Registry, SPL, Guideline)
- [ ] Implement profile loader with Pydantic validation
- [ ] Wire profiles to chunker registry

**Day 11-14**: Atomic Deletions

- [ ] **Commit 1**: Add ChunkerPort + delete `custom_splitters.py` (8 custom chunkers)
- [ ] **Commit 2**: Add LangChain wrapper + delete `semantic_splitter.py`, `sliding_window.py`
- [ ] **Commit 3**: Add HuggingFace/syntok wrappers + delete `sentence_splitters.py`
- [ ] **Commit 4**: Add unstructured wrapper + delete `xml_parser.py`
- [ ] **Commit 5**: Harden MinerU gate + delete `pdf_parser.py`
- [ ] **Commit 6**: Add profiles + delete adapter `.split_document()` methods
- [ ] **Commit 7**: Update all imports, delete legacy tests

### Phase 2: Integration Testing (Week 3-4)

**Day 15-18**: Profile Validation

- [ ] Test PMC articles (10 samples) with `pmc-imrad` profile
- [ ] Test CT.gov studies (10 samples) with `ctgov-registry` profile
- [ ] Test SPL labels (10 samples) with `spl-label` profile
- [ ] Test guidelines (5 samples) with `guideline` profile
- [ ] Validate: char offsets accurate, section labels correct, tables preserved

**Day 19-21**: MinerU Two-Phase Gate

- [ ] Test 3 PDFs end-to-end: download → MinerU → `pdf_ir_ready` → `postpdf-start` → chunking
- [ ] Test MinerU failure handling (GPU unavailable, processing error)
- [ ] Verify no CPU fallbacks
- [ ] Test Dagster sensor for `postpdf-start` auto-trigger

**Day 22-28**: Quality & Performance

- [ ] Benchmark chunking throughput (should be ≥100 docs/sec for non-PDF)
- [ ] Compare chunk quality before/after (manual inspection of 50 chunks)
- [ ] Validate downstream retrieval unchanged (Recall@10, P95 latency)
- [ ] Measure codebase reduction (≥40% target)

### Phase 3: Production Deployment (Week 5-6)

**Day 29-32**: Deployment

- [ ] Merge feature branch to main (no legacy code remains)
- [ ] Deploy to production
- [ ] Monitor chunk quality metrics for 48 hours
- [ ] Monitor MinerU gate behavior (`pdf_ir_ready` → `postpdf-start` latency)

**Day 33-35**: Validation & Rollback Readiness

- [ ] Verify all 4 profiles producing expected chunk structure
- [ ] Verify span grounding: sample 100 chunks, validate char offsets
- [ ] Verify table fidelity: sample 50 table chunks, check HTML preservation
- [ ] Emergency rollback plan: revert entire feature branch if critical issues

**Day 36-42**: Stabilization

- [ ] Address any profile tuning needs based on downstream feedback
- [ ] Document lessons learned
- [ ] Conduct retrospective
- [ ] Finalize `CODEBASE_REDUCTION_REPORT.md`

---

## Configuration Management

### Profile YAML Schema

```yaml
# Profile configuration schema
name: str  # Unique profile identifier
domain: str  # "literature", "registry", "label", "guideline"
chunker_type: str  # "langchain_recursive", "llamaindex_sentence_window"
target_tokens: int  # Target chunk size in tokens (e.g., 450)
overlap_tokens: int  # Overlap between chunks (e.g., 50)
respect_boundaries: list[str]  # ["heading", "table", "figure_caption", "section"]
sentence_splitter: str  # "huggingface" or "syntok"
preserve_tables_as_html: bool  # true to keep HTML when uncertain
filters: list[str]  # ["drop_boilerplate", "exclude_references", "deduplicate_page_furniture"]
metadata:
  section_label_source: str  # Where to extract section labels
  intent_hints: dict[str, str]  # Map section names to intent hints
```

### Profile Loading

```python
# src/Medical_KG_rev/services/chunking/profiles/loader.py
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, validator

class ProfileConfig(BaseModel):
    name: str
    domain: str
    chunker_type: str
    target_tokens: int = 512
    overlap_tokens: int = 50
    respect_boundaries: list[str] = Field(default_factory=list)
    sentence_splitter: str = "syntok"
    preserve_tables_as_html: bool = True
    filters: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @validator("chunker_type")
    def validate_chunker_type(cls, v):
        allowed = ["langchain_recursive", "llamaindex_sentence_window"]
        if v not in allowed:
            raise ValueError(f"chunker_type must be one of {allowed}")
        return v

def load_profile(name: str) -> ProfileConfig:
    """Load and validate a profile from YAML."""
    profile_path = Path(f"config/chunking/profiles/{name}.yaml")
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {name}")

    with open(profile_path) as f:
        data = yaml.safe_load(f)

    return ProfileConfig(**data)
```

---

## Observability & Monitoring

### Metrics

```python
# src/Medical_KG_rev/services/chunking/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Chunking metrics
CHUNKING_DOCUMENTS_TOTAL = Counter(
    "chunking_documents_total",
    "Total documents chunked",
    ["profile", "source_system"]
)

CHUNKING_DURATION_SECONDS = Histogram(
    "chunking_duration_seconds",
    "Chunking duration",
    ["profile", "source_system"]
)

CHUNKING_CHUNKS_PER_DOCUMENT = Histogram(
    "chunking_chunks_per_document",
    "Number of chunks produced per document",
    ["profile", "source_system"]
)

CHUNKING_FAILURES_TOTAL = Counter(
    "chunking_failures_total",
    "Chunking failures",
    ["profile", "error_type"]
)

# MinerU gate metrics
MINERU_GATE_WAITING_TOTAL = Gauge(
    "mineru_gate_waiting_total",
    "Jobs waiting at MinerU gate (pdf_ir_ready)"
)

POSTPDF_START_TRIGGERED_TOTAL = Counter(
    "postpdf_start_triggered_total",
    "postpdf-start triggers",
    ["trigger_source"]  # "manual", "auto_sensor", "api"
)
```

### CloudEvents

```python
# Chunking lifecycle events
{
  "specversion": "1.0",
  "type": "medical_kg.chunking.started",
  "source": "/chunking/pmc-imrad",
  "id": "chunk-job-abc123",
  "time": "2025-10-07T14:30:00Z",
  "data": {
    "job_id": "job-abc123",
    "doc_id": "pmc:PMC8675309",
    "profile": "pmc-imrad",
    "source_system": "pmc"
  }
}

{
  "type": "medical_kg.chunking.completed",
  "data": {
    "job_id": "job-abc123",
    "doc_id": "pmc:PMC8675309",
    "profile": "pmc-imrad",
    "chunks_produced": 42,
    "duration_seconds": 2.5,
    "avg_chunk_tokens": 438
  }
}

{
  "type": "medical_kg.mineru.gate.waiting",
  "data": {
    "job_id": "job-def456",
    "doc_id": "pmc:PMC1234567",
    "pdf_ir_ready_at": "2025-10-07T14:28:00Z",
    "waiting_duration_seconds": 120
  }
}

{
  "type": "medical_kg.postpdf.start.triggered",
  "data": {
    "job_id": "job-def456",
    "trigger_source": "auto_sensor",
    "triggered_by": "dagster-sensor",
    "triggered_at": "2025-10-07T14:30:00Z"
  }
}
```

---

## Testing Strategy

### Unit Tests (50 tests, ≥90% coverage)

**ChunkerPort** (5 tests):

- Protocol compliance check
- Registry registration/retrieval
- Profile not found error
- Invalid profile validation
- Chunk schema validation

**Profiles** (12 tests):

- IMRaD profile (3 tests: heading respect, figure caption preservation, token budget)
- Registry profile (3 tests: atomic outcomes, eligibility, AE tables)
- SPL profile (3 tests: LOINC sections, dosage, warnings)
- Guideline profile (3 tests: recommendation units, evidence tables, grades)

**Library Wrappers** (15 tests):

- LangChain recursive splitter (4 tests: boundary respect, overlap, offset accuracy, token budget)
- LlamaIndex sentence window (3 tests: window size, coherence, offset accuracy)
- HuggingFace segmenter (3 tests: biomedical abbreviations, sentence boundaries, offset accuracy)
- syntok segmenter (2 tests: fast throughput, messy punctuation)
- Unstructured parser (3 tests: JATS XML, SPL XML, HTML)

**Filters** (8 tests):

- drop_boilerplate (2 tests)
- exclude_references (2 tests)
- deduplicate_page_furniture (2 tests)
- preserve_tables_html (2 tests)

**MinerU Gate** (6 tests):

- Success flow: `pdf_downloaded` → `pdf_ir_ready` → HALT
- Explicit `postpdf-start` trigger
- GPU unavailable error (fail-fast)
- MinerU processing error (no CPU fallback)
- Ledger state validation
- Docling validation guard (reject PDFs)

**Chunk Schema** (4 tests):

- Required fields validation
- Provenance metadata completeness
- Table HTML preservation
- Offset accuracy

### Integration Tests (21 tests)

**End-to-End Profiles** (18 tests):

- PMC articles with `pmc-imrad` (5 articles)
- CT.gov studies with `ctgov-registry` (5 studies)
- SPL labels with `spl-label` (5 labels)
- Guidelines with `guideline` (3 guidelines)

**PDF Two-Phase Gate** (3 tests):

- PMC full-text PDF (download → MinerU → `pdf_ir_ready` → `postpdf-start` → chunking)
- Unpaywall PDF
- CORE PDF

### Quality Validation Tests

**Chunk Quality**:

- Sample 100 chunks, manually inspect:
  - Char offsets point to correct text in source
  - Section labels match expected structure (IMRaD, LOINC, registry outcomes)
  - No mid-sentence splits (except at boundaries)
  - Tables preserved as HTML when uncertainty > 0.2

**Performance Benchmarks**:

- Chunking throughput ≥100 docs/sec for non-PDF sources
- HuggingFace vs syntok throughput ratio ~1:10 (syntok 10x faster)
- Memory usage <500MB for batches of 100 documents

**Regression Tests**:

- Compare chunk quality before/after for 10 PMC articles
- Compare chunk quality before/after for 10 CT.gov studies
- Verify downstream retrieval quality unchanged (Recall@10, P95 latency <500ms)

---

## Rollback Procedures

### Rollback Trigger Conditions

Execute rollback if any occur **within 48 hours** of production deployment:

1. **Chunk Quality Degradation**: >10% of chunks with incorrect offsets or missing section labels
2. **Downstream Failures**: Retrieval Recall@10 drops >5% or extraction F1 drops >10%
3. **Performance Regression**: Chunking throughput <70 docs/sec or P95 latency >2x baseline
4. **MinerU Gate Issues**: >50% of PDF jobs stalled in `pdf_ir_ready` for >1 hour
5. **Critical Bug**: Data loss, incorrect provenance, or system instability

### Rollback Procedure

```bash
# Step 1: Identify feature branch commit
FEATURE_COMMIT=$(git log --oneline --grep="feat: Add ChunkerPort interface" -n 1 | awk '{print $1}')

# Step 2: Create rollback branch
git checkout -b rollback-chunking-normalization main

# Step 3: Revert feature branch
git revert -m 1 $FEATURE_COMMIT

# Step 4: Test rollback locally
pytest tests/chunking/ tests/parsing/ -v

# Step 5: Deploy rollback
git push origin rollback-chunking-normalization

# Step 6: Monitor for 24 hours
# - Verify chunk quality restored
# - Verify downstream retrieval/extraction restored
```

### Post-Rollback Actions

1. Preserve logs (CloudEvents, Prometheus metrics, failed chunks)
2. Root cause analysis (why did new architecture fail?)
3. Fix forward plan (remediate issues, re-attempt deployment)

---

## Open Questions

1. **Profile Tuning**: What's the optimal `target_tokens` for IMRaD vs Registry profiles?
   - **Answer**: Start with 450 for literature, 300 for registry; tune based on retrieval Recall@10

2. **HuggingFace vs syntok**: When is biomedical-aware sentence segmentation worth the 10x performance cost?
   - **Answer**: Use HuggingFace models for Methods→Results transitions, syntok for bulk ingestion

3. **Auto `postpdf-start`**: Should Dagster sensor auto-trigger after fixed delay or require manual approval?
   - **Answer**: Configurable delay (default 5 min), with manual override for quality-critical sources

4. **Table Rectangularization**: At what confidence threshold do we preserve HTML vs rectangularize?
   - **Answer**: Preserve HTML if MinerU confidence <0.8, rectangularize if ≥0.8

5. **Profile Versioning**: How do we handle profile changes (e.g., `pmc-imrad-v2.yaml`)?
   - **Answer**: Git version profiles, store `chunking_profile` + `profile_version` in chunk metadata

---

## Summary

This design replaces fragmented, bespoke chunking/parsing with a **unified, library-based architecture** that:

- **Respects clinical structure** via profiles (IMRaD, LOINC, registry, guideline)
- **Delegates to proven libraries** (LangChain, LlamaIndex, HuggingFace) reducing code by 43%
- **Enforces GPU-only policy** for MinerU with explicit two-phase gate
- **Produces span-grounded chunks** with complete provenance (offsets, section labels, intent hints)
- **Preserves table fidelity** (HTML when uncertainty high)
- **Hard cutover** (no legacy compatibility, complete replacement)

**Status**: Ready for implementation (Week 1-6 timeline)
