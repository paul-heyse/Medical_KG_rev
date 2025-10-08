# Implementation Tasks: Clinical-Aware Parsing, Chunking & Normalization

## CRITICAL: Hard Cutover Strategy

**No Legacy Compatibility** - Complete replacement, not migration:

- ❌ No feature flags or compatibility shims
- ❌ No gradual rollout or dual-path testing
- ✅ Delete legacy chunkers/parsers in same commits as new implementations
- ✅ Single feature branch with full replacement
- ✅ Rollback = revert entire branch (not toggle back to legacy)

---

## 1. Legacy Code Decommissioning Plan

### 1.1 Legacy Code Inventory (Audit Phase)

- [ ] 1.1.1 Identify all files in `src/Medical_KG_rev/services/chunking/` to delete:
  - [ ] `custom_splitters.py` (8 custom chunkers, 420 lines) - Replace with ChunkerPort + library wrappers
  - [ ] `semantic_splitter.py` (75 lines) - Replace with LlamaIndex node parsers
  - [ ] `sliding_window.py` (62 lines) - Replace with langchain RecursiveCharacterTextSplitter
  - [ ] `section_aware_splitter.py` (110 lines) - Replace with profile-based chunking
- [ ] 1.1.2 Identify all files in `src/Medical_KG_rev/services/parsing/` to delete:
  - [ ] `pdf_parser.py` (bespoke PDF logic, 180 lines) - MinerU is the only PDF path
  - [ ] `xml_parser.py` (custom XML parsing, 95 lines) - Replace with unstructured
  - [ ] `sentence_splitters.py` (3 implementations, 140 lines) - Replace with scispaCy/syntok
- [ ] 1.1.3 Identify adapter methods to delete:
  - [ ] Search for `def split_document\|\.chunk\(` in adapters/ (15 occurrences)
  - [ ] All adapter-specific chunking logic → delegate to ChunkerPort
- [ ] 1.1.4 Create deletion checklist: `LEGACY_DECOMMISSION_CHECKLIST.md`

### 1.2 Dependency Analysis (Pre-Delete Validation)

- [ ] 1.2.1 Find all imports of custom chunkers:

  ```bash
  grep -r "from.*custom_splitters import\|SemanticSplitter\|SlidingWindow" src/
  ```

- [ ] 1.2.2 Find all imports of custom parsers:

  ```bash
  grep -r "from.*pdf_parser import\|from.*xml_parser import" src/
  ```

- [ ] 1.2.3 Find all `.split_document()` calls:

  ```bash
  grep -r "\.split_document\(\|\.chunk\(" src/Medical_KG_rev/adapters/
  ```

- [ ] 1.2.4 Document all dependencies in `LEGACY_DEPENDENCIES.md` with replacement plan

### 1.3 Delegation to Open-Source Libraries (Validation)

- [ ] 1.3.1 **Chunking**: Verify delegation to langchain-text-splitters + LlamaIndex
  - [ ] Audit: Which custom splitters are semantic vs recursive vs sliding window?
  - [ ] Decision: Map to `RecursiveCharacterTextSplitter` (default) or `SentenceWindowNodeParser` (coherence-sensitive)
  - [ ] Delete: All 8 custom chunker implementations
  - [ ] Verify: Chunk quality (offsets, section labels) preserved
- [ ] 1.3.2 **Sentence Segmentation**: Verify delegation to scispaCy/syntok
  - [ ] Audit: Where are custom sentence splitters used?
  - [ ] Decision: scispaCy for biomedical-aware splitting, syntok for speed
  - [ ] Delete: All 3 custom sentence splitter implementations
  - [ ] Verify: Sentence boundaries match expected behavior
- [ ] 1.3.3 **Tokenization**: Verify delegation to transformers/tiktoken
  - [ ] Audit: Where are custom tokenizers used?
  - [ ] Decision: Use `transformers.AutoTokenizer` for Qwen3 alignment
  - [ ] Delete: All custom tokenizer logic
  - [ ] Verify: Token budgets honored before embedding
- [ ] 1.3.4 **XML Parsing**: Verify delegation to unstructured
  - [ ] Audit: What XML parsing logic exists?
  - [ ] Decision: Replace with `unstructured.partition_xml`
  - [ ] Delete: Custom XML parsing implementations
  - [ ] Verify: JATS XML, SPL XML parsed correctly
- [ ] 1.3.5 **PDF Parsing**: Verify MinerU-only path (no Docling in prod)
  - [ ] Audit: Where is Docling used?
  - [ ] Decision: Keep Docling for non-OCR contexts (HTML/text), not PDFs
  - [ ] Verify: MinerU is the **only** PDF path, with explicit GPU gate

### 1.4 Atomic Deletion (Commit Strategy)

- [ ] 1.4.1 Create commit plan with atomic deletions:
  - [ ] Commit 1: Add ChunkerPort interface + delete `custom_splitters.py`
  - [ ] Commit 2: Add langchain/LlamaIndex wrappers + delete `semantic_splitter.py`, `sliding_window.py`
  - [ ] Commit 3: Add scispaCy/syntok wrappers + delete `sentence_splitters.py`
  - [ ] Commit 4: Add unstructured wrapper + delete `xml_parser.py`
  - [ ] Commit 5: Harden MinerU gate + delete `pdf_parser.py`
  - [ ] Commit 6: Add profile system + delete adapter `.split_document()` methods
  - [ ] Commit 7: Update all imports
  - [ ] Commit 8: Delete legacy tests, add new ChunkerPort tests
- [ ] 1.4.2 Run full test suite after each commit
- [ ] 1.4.3 Document deleted code statistics in commit messages

### 1.5 Import Cleanup (Post-Delete)

- [ ] 1.5.1 Update `src/Medical_KG_rev/services/chunking/__init__.py`:
  - [ ] Remove: `from .custom_splitters import SemanticSplitter, SlidingWindow, ...`
  - [ ] Add: `from .port import ChunkerPort, chunk_document`
- [ ] 1.5.2 Update adapter imports:
  - [ ] Remove: `from ..services.chunking.custom_splitters import ...`
  - [ ] Add: `from ..services.chunking.port import chunk_document`
- [ ] 1.5.3 Run `ruff check --select F401` to find unused imports
- [ ] 1.5.4 Run `mypy src/` to verify no type errors

### 1.6 Test Migration (Delete and Replace)

- [ ] 1.6.1 Delete legacy chunking tests:
  - [ ] `tests/chunking/test_custom_splitters.py` (8 chunker tests)
  - [ ] `tests/chunking/test_semantic_splitter.py`
  - [ ] `tests/chunking/test_sliding_window.py`
- [ ] 1.6.2 Create new ChunkerPort tests:
  - [ ] `tests/chunking/test_chunker_port.py` (interface compliance)
  - [ ] `tests/chunking/test_profiles.py` (IMRaD, Registry, SPL, Guideline)
  - [ ] `tests/chunking/test_library_wrappers.py` (langchain, LlamaIndex, scispaCy)
- [ ] 1.6.3 Verify test coverage ≥90% for new chunking code
- [ ] 1.6.4 Delete all references to custom chunkers in test fixtures

### 1.7 Documentation Updates

- [ ] 1.7.1 Update `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md`:
  - [ ] Remove: Section on "Custom Chunking Strategies"
  - [ ] Add: Section on "ChunkerPort Interface & Profiles"
  - [ ] Add: Table of profile configurations (IMRaD, Registry, SPL, Guideline)
- [ ] 1.7.2 Update `docs/guides/chunking.md`:
  - [ ] Remove: Legacy examples with custom chunkers
  - [ ] Add: Profile-based chunking examples
  - [ ] Add: Library delegation guide (langchain, LlamaIndex, scispaCy)
- [ ] 1.7.3 Create `DELETED_CODE.md` for chunking/parsing

### 1.8 Codebase Size Validation

- [ ] 1.8.1 Measure codebase before changes:

  ```bash
  cloc src/Medical_KG_rev/services/chunking/ src/Medical_KG_rev/services/parsing/
  ```

- [ ] 1.8.2 Measure after changes
- [ ] 1.8.3 Validate codebase shrinkage:
  - [ ] Assert: ≥40% reduction in chunking/parsing code
  - [ ] Document: `CODEBASE_REDUCTION_REPORT.md`

---

## 2. Foundation & Dependencies

- [x] 2.1 Add **langchain-text-splitters>=0.2.0** to requirements.txt
- [x] 2.2 Add **llama-index-core>=0.10.0** for node parsers
- [x] 2.3 Add **scispacy>=0.5.4** + **en-core-sci-sm** model
- [x] 2.4 Add **syntok>=1.4.4** for fast sentence splitting
- [x] 2.5 Add **unstructured[local-inference]>=0.12.0** for XML/HTML
- [x] 2.6 Add **tiktoken>=0.6.0** and **transformers>=4.38.0** for tokenization
- [x] 2.7 Pin exact versions in requirements.txt (no `^` or `~`)
- [ ] 2.8 Test dependency installation in clean venv
- [ ] 2.9 Download scispaCy model: `python -m spacy download en_core_sci_sm`
- [ ] 2.10 Verify all libraries import without errors

---

## 3. ChunkerPort Interface & Runtime Registry

- [x] 3.1 Define `ChunkerPort` Protocol in `src/Medical_KG_rev/services/chunking/port.py`:

  ```python
  class ChunkerPort(Protocol):
      def chunk(self, document: Document, profile: str) -> list[Chunk]: ...
  ```

- [x] 3.2 Define `Chunk` dataclass with required fields:
  - [ ] `chunk_id: str`
  - [ ] `doc_id: str`
  - [ ] `text: str`
  - [ ] `char_offsets: tuple[int, int]`
  - [ ] `section_label: str` (e.g., "Methods", "LOINC:34089-3")
  - [ ] `intent_hint: str` (e.g., "eligibility", "outcome", "ae", "dose")
  - [ ] `page_bbox: dict | None` (for PDFs)
  - [ ] `metadata: dict[str, Any]`
- [x] 3.3 Implement chunker registry:
  - [ ] `register_chunker(name: str, implementation: Type[ChunkerPort])`
  - [ ] `get_chunker(name: str) -> ChunkerPort`
- [x] 3.4 Add validation: raise if profile not registered
- [x] 3.5 Write unit tests for ChunkerPort protocol compliance

---

## 4. Profile-Based Chunking System

### 4.1 Profile Data Model

- [x] 4.1.1 Create `Profile` Pydantic model in `src/Medical_KG_rev/services/chunking/profiles/models.py`:

  ```python
  class Profile(BaseModel):
      name: str
      domain: str  # "literature", "registry", "label", "guideline"
      target_tokens: int = 512
      overlap_tokens: int = 50
      respect_boundaries: list[str]  # ["heading", "table", "section"]
      sentence_splitter: str = "syntok"  # or "scispacy"
      preserve_tables_as_html: bool = True
      filters: list[str] = ["drop_boilerplate", "exclude_references"]
  ```

- [x] 4.1.2 Load profiles from YAML: `config/chunking/profiles/*.yaml`
- [x] 4.1.3 Validate profiles on startup (Pydantic validation)

### 4.2 IMRaD Profile (PMC JATS)

- [x] 4.2.1 Create `config/chunking/profiles/pmc-imrad.yaml`:

  ```yaml
  name: pmc-imrad
  domain: literature
  target_tokens: 450
  overlap_tokens: 50
  respect_boundaries:
    - heading  # Never split across IMRaD sections
    - figure_caption
    - table
  sentence_splitter: scispacy  # Biomedical-aware
  preserve_tables_as_html: true
  filters:
    - drop_boilerplate
    - exclude_references
    - deduplicate_page_furniture
  ```

- [x] 4.2.2 Implement IMRaD chunker using LangChain `RecursiveCharacterTextSplitter`
- [ ] 4.2.3 Test on 10 PMC articles, verify heading alignment
- [ ] 4.2.4 Validate section labels: "Abstract", "Introduction", "Methods", "Results", "Discussion"

### 4.3 Registry Profile (CT.gov)

- [x] 4.3.1 Create `config/chunking/profiles/ctgov-registry.yaml`:

  ```yaml
  name: ctgov-registry
  domain: registry
  target_tokens: 300
  overlap_tokens: 0  # No overlap for atomic units
  respect_boundaries:
    - eligibility_criteria
    - outcome_measure
    - adverse_event_table
    - results_section
  sentence_splitter: syntok  # Fast, sufficient for structured data
  preserve_tables_as_html: true
  filters:
    - drop_boilerplate
  ```

- [ ] 4.3.2 Implement registry chunker with atomic units:
  - [ ] Eligibility criteria as single chunk
  - [ ] Each outcome measure as separate chunk
  - [ ] Adverse event tables as atomic chunks (preserve effect pairs)
- [ ] 4.3.3 Test on 10 CT.gov studies
- [ ] 4.3.4 Validate intent hints: "eligibility", "outcome", "ae", "results"

### 4.4 SPL Profile (DailyMed)

- [x] 4.4.1 Create `config/chunking/profiles/spl-label.yaml`:

  ```yaml
  name: spl-label
  domain: label
  target_tokens: 400
  overlap_tokens: 30
  respect_boundaries:
    - loinc_section  # LOINC-coded sections
    - table
  sentence_splitter: scispacy
  preserve_tables_as_html: true
  filters:
    - drop_boilerplate
    - exclude_references
  ```

- [ ] 4.4.2 Implement SPL chunker with LOINC section awareness:
  - [ ] Parse LOINC codes from SPL XML
  - [ ] Chunk by section: "Indications", "Dosage", "Warnings", "Adverse Reactions"
- [ ] 4.4.3 Test on 10 SPL labels
- [ ] 4.4.4 Validate section labels include LOINC codes (e.g., "LOINC:34089-3 Indications")

### 4.5 Guideline Profile

- [x] 4.5.1 Create `config/chunking/profiles/guideline.yaml`:

  ```yaml
  name: guideline
  domain: guideline
  target_tokens: 350
  overlap_tokens: 0  # Recommendations are atomic
  respect_boundaries:
    - recommendation_unit  # Statement + strength + grade
    - evidence_table
  sentence_splitter: syntok
  preserve_tables_as_html: true
  filters:
    - drop_boilerplate
  ```

- [ ] 4.5.2 Implement guideline chunker:
  - [ ] Isolate recommendation units (statement, strength, certainty/grade)
  - [ ] Keep evidence tables attached to recommendations
- [ ] 4.5.3 Test on 5 clinical guidelines
- [ ] 4.5.4 Validate intent hints: "recommendation", "evidence"

---

## 5. Library Integration Layer

### 5.1 LangChain Text Splitters Wrapper

- [x] 5.1.1 Create `src/Medical_KG_rev/services/chunking/wrappers/langchain_splitter.py`
- [x] 5.1.2 Implement `LangChainChunker` class:

  ```python
  class LangChainChunker:
      def __init__(self, profile: Profile):
          self.splitter = RecursiveCharacterTextSplitter(
              chunk_size=profile.target_tokens * 4,  # Approximate tokens→chars
              chunk_overlap=profile.overlap_tokens * 4,
              length_function=self._count_tokens,
              separators=["\n\n", "\n", ". ", " ", ""]
          )

      def chunk(self, document: Document, profile: str) -> list[Chunk]:
          # Implementation
  ```

- [ ] 5.1.3 Add boundary detection (heading, table, section)
- [ ] 5.1.4 Preserve char offsets for each chunk
- [ ] 5.1.5 Write unit tests with 5 test documents

### 5.2 LlamaIndex Node Parsers Wrapper

- [ ] 5.2.1 Create `src/Medical_KG_rev/services/chunking/wrappers/llamaindex_parser.py`
- [ ] 5.2.2 Implement `LlamaIndexChunker` class using `SentenceWindowNodeParser`:

  ```python
  class LlamaIndexChunker:
      def __init__(self, profile: Profile):
          self.parser = SentenceWindowNodeParser(
              window_size=3,  # 3 sentences per window
              window_metadata_key="window",
              original_text_metadata_key="original_sentence"
          )
  ```

- [ ] 5.2.3 Add sentence boundary detection via scispaCy/syntok
- [ ] 5.2.4 Map LlamaIndex nodes to `Chunk` dataclass
- [ ] 5.2.5 Write unit tests for coherence preservation

### 5.3 scispaCy Sentence Segmentation Wrapper

- [ ] 5.3.1 Create `src/Medical_KG_rev/services/chunking/wrappers/scispacy_segmenter.py`
- [ ] 5.3.2 Implement `SciSpaCySentenceSegmenter`:

  ```python
  import spacy
  nlp = spacy.load("en_core_sci_sm")

  def segment_sentences(text: str) -> list[tuple[int, int, str]]:
      doc = nlp(text)
      return [(sent.start_char, sent.end_char, sent.text) for sent in doc.sents]
  ```

- [ ] 5.3.3 Handle biomedical abbreviations (e.g., "Fig.", "et al.")
- [ ] 5.3.4 Preserve char offsets
- [ ] 5.3.5 Write unit tests with biomedical text samples

### 5.4 syntok Fast Sentence Splitter Wrapper

- [ ] 5.4.1 Create `src/Medical_KG_rev/services/chunking/wrappers/syntok_segmenter.py`
- [ ] 5.4.2 Implement `SyntokSentenceSegmenter`:

  ```python
  from syntok import segmenter

  def segment_sentences(text: str) -> list[tuple[int, int, str]]:
      # Implementation with offset tracking
  ```

- [ ] 5.4.3 Handle messy punctuation
- [ ] 5.4.4 Preserve char offsets
- [ ] 5.4.5 Benchmark throughput vs scispaCy (should be 5-10x faster)

### 5.5 Tokenizer Wrappers (HF / tiktoken)

- [ ] 5.5.1 Create `src/Medical_KG_rev/services/chunking/wrappers/tokenizers.py`
- [ ] 5.5.2 Implement HF tokenizer wrapper for Qwen3:

  ```python
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")

  def count_tokens(text: str) -> int:
      return len(tokenizer.encode(text))
  ```

- [ ] 5.5.3 Add token budget enforcement before chunking
- [ ] 5.5.4 Cache tokenizer instance (avoid re-loading)
- [ ] 5.5.5 Write unit tests for token counting accuracy

### 5.6 unstructured Wrapper (XML/HTML)

- [ ] 5.6.1 Create `src/Medical_KG_rev/services/parsing/wrappers/unstructured_parser.py`
- [ ] 5.6.2 Implement `UnstructuredParser`:

  ```python
  from unstructured.partition.xml import partition_xml
  from unstructured.partition.html import partition_html

  def parse_xml(content: str) -> Document:
      elements = partition_xml(text=content)
      # Map to IR Document
  ```

- [ ] 5.6.3 Map unstructured elements to IR blocks
- [ ] 5.6.4 Preserve metadata (element type, attributes)
- [ ] 5.6.5 Test on JATS XML, SPL XML, HTML guidelines

---

## 6. Filter Chain System

- [ ] 6.1 Create `src/Medical_KG_rev/services/chunking/filters/` directory
- [ ] 6.2 Implement `drop_boilerplate` filter:
  - [ ] Remove headers/footers via regex patterns
  - [ ] Remove "Page X of Y" artifacts
- [ ] 6.3 Implement `exclude_references` filter:
  - [ ] Detect "References" section start
  - [ ] Drop all text after that section
- [ ] 6.4 Implement `deduplicate_page_furniture` filter:
  - [ ] Detect repeated text across pages (e.g., running headers)
  - [ ] Remove duplicates
- [ ] 6.5 Implement `preserve_tables_html` filter:
  - [ ] Keep HTML for tables when `rectangularize_confidence < 0.8`
  - [ ] Tag chunk with `is_unparsed_table=true`
- [ ] 6.6 Create filter chain executor:

  ```python
  def apply_filters(chunks: list[Chunk], filters: list[str]) -> list[Chunk]:
      for filter_name in filters:
          chunks = FILTER_REGISTRY[filter_name](chunks)
      return chunks
  ```

- [ ] 6.7 Write unit tests for each filter

---

## 7. MinerU Two-Phase Gate (Hardened)

### 7.1 MinerU Integration

- [ ] 7.1.1 Verify MinerU service exists: `src/Medical_KG_rev/services/gpu/mineru_service.py`
- [ ] 7.1.2 Ensure MinerU runs on GPU only (no CPU fallback):

  ```python
  if not torch.cuda.is_available():
      raise GpuNotAvailableError("MinerU requires GPU")
  ```

- [ ] 7.1.3 MinerU outputs:
  - [ ] Markdown/JSON artifacts
  - [ ] Structured IR with page/bbox maps
  - [ ] Table HTML (preserved when rectangularization uncertain)
- [ ] 7.1.4 On success: Update ledger to `pdf_ir_ready`, pipeline HALTS
- [ ] 7.1.5 On failure: Mark ledger `mineru_failed`, abort (no retry on CPU)

### 7.2 Explicit `postpdf-start` Trigger

- [ ] 7.2.1 Add `/v1/jobs/{job_id}/postpdf-start` endpoint in gateway
- [ ] 7.2.2 Validate ledger state is `pdf_ir_ready` before resuming
- [ ] 7.2.3 Resume chunking/embedding stages after trigger
- [ ] 7.2.4 Update Dagster sensor to poll for `postpdf-start` events
- [ ] 7.2.5 Add Dagster UI shortcut for common `postpdf-start` workflows

### 7.3 Docling Scope Limitation

- [ ] 7.3.1 Audit where Docling is used: `grep -r "docling" src/`
- [ ] 7.3.2 Ensure Docling is NOT in PDF processing path
- [ ] 7.3.3 Keep Docling for:
  - [ ] Non-OCR HTML normalization
  - [ ] Local developer experiments (non-GPU contexts)
  - [ ] Text/XML parsing (not PDFs)
- [ ] 7.3.4 Add validation: Docling cannot be called with PDF inputs in prod

### 7.4 Ledger Schema Updates

- [ ] 7.4.1 Add fields to `JobLedgerEntry`:
  - [ ] `pdf_downloaded: bool = False`
  - [ ] `pdf_ir_ready: bool = False`
  - [ ] `postpdf_start_triggered: bool = False`
  - [ ] `mineru_bbox_map: dict | None = None`
- [ ] 7.4.2 Update ledger state machine diagram
- [ ] 7.4.3 Migrate existing ledger entries (add new fields with defaults)

---

## 8. I/O, Provenance & Failure Semantics

### 8.1 Chunk Schema Validation

- [ ] 8.1.1 Define Pydantic model for `Chunk`:

  ```python
  class Chunk(BaseModel):
      chunk_id: str
      doc_id: str
      text: str
      char_offsets: tuple[int, int]
      section_label: str  # e.g., "Methods", "LOINC:34089-3"
      intent_hint: str  # "eligibility", "outcome", "ae", "dose", "narrative"
      page_bbox: dict | None = None  # For PDFs
      metadata: dict[str, Any] = Field(default_factory=dict)
  ```

- [ ] 8.1.2 Validate all chunks before storage
- [ ] 8.1.3 Raise `ValidationError` if required fields missing

### 8.2 Provenance Tracking

- [ ] 8.2.1 Attach to every chunk:
  - [ ] `source_system: str` (e.g., "pmc", "ctgov", "dailymed")
  - [ ] `chunking_profile: str` (e.g., "pmc-imrad")
  - [ ] `chunker_version: str` (e.g., "langchain-v0.2.0")
  - [ ] `created_at: datetime`
- [ ] 8.2.2 Store in chunk metadata
- [ ] 8.2.3 Index in OpenSearch for retrieval filters

### 8.3 Failure Semantics

- [ ] 8.3.1 MinerU failure → `mineru_failed`, abort (no CPU fallback)
- [ ] 8.3.2 Chunking failure → `chunking_failed`, log error, retry once
- [ ] 8.3.3 Table rectangularization uncertainty → preserve HTML, tag `is_unparsed_table=true`
- [ ] 8.3.4 Sentence segmentation failure → fall back to naive splitting, log warning
- [ ] 8.3.5 Profile not found → raise `ProfileNotFoundError`, abort

---

## 9. Integration with Orchestration

### 9.1 Dagster Stage Definition

- [ ] 9.1.1 Create `ChunkStage` op in Dagster:

  ```python
  @op
  def chunk_stage(context, document: Document, profile: str) -> list[Chunk]:
      chunker = get_chunker(profile)
      return chunker.chunk(document, profile)
  ```

- [ ] 9.1.2 Add to auto pipeline (after IR creation)
- [ ] 9.1.3 Add to PDF two-phase pipeline (after `postpdf-start`)
- [ ] 9.1.4 Wire resilience policy: retry on transient failures

### 9.2 Gateway API Updates

- [ ] 9.2.1 Add `chunking_profile` parameter to `/v1/ingest/{source}`:

  ```json
  {
    "data": {
      "type": "IngestionRequest",
      "attributes": {
        "identifiers": ["NCT04267848"],
        "chunking_profile": "ctgov-registry"
      }
    }
  }
  ```

- [ ] 9.2.2 Default profile per source:
  - [ ] PMC → "pmc-imrad"
  - [ ] CT.gov → "ctgov-registry"
  - [ ] DailyMed → "spl-label"
  - [ ] Guidelines → "guideline"
- [ ] 9.2.3 Validate profile exists before submission
- [ ] 9.2.4 Update OpenAPI spec with new parameter

---

## 10. Testing Strategy

### 10.1 Unit Tests

- [ ] 10.1.1 ChunkerPort protocol compliance (5 tests)
- [ ] 10.1.2 Each profile (IMRaD, Registry, SPL, Guideline) (12 tests)
- [ ] 10.1.3 Each library wrapper (LangChain, LlamaIndex, scispaCy, syntok, unstructured) (15 tests)
- [ ] 10.1.4 Filter chain system (8 tests)
- [ ] 10.1.5 MinerU gate logic (6 tests)
- [ ] 10.1.6 Chunk schema validation (4 tests)
- [ ] **Total**: 50 unit tests, target coverage ≥90%

### 10.2 Integration Tests

- [ ] 10.2.1 End-to-end PMC article chunking (5 articles)
- [ ] 10.2.2 End-to-end CT.gov study chunking (5 studies)
- [ ] 10.2.3 End-to-end SPL label chunking (5 labels)
- [ ] 10.2.4 End-to-end guideline chunking (3 guidelines)
- [ ] 10.2.5 PDF two-phase gate with MinerU + `postpdf-start` (3 PDFs)
- [ ] **Total**: 21 integration tests

### 10.3 Quality Validation Tests

- [ ] 10.3.1 Verify chunk offsets are accurate (sample 100 chunks, manual inspection)
- [ ] 10.3.2 Verify section labels match expected IMRaD/LOINC/registry structure
- [ ] 10.3.3 Verify table chunks preserve HTML when uncertainty high
- [ ] 10.3.4 Verify no mid-sentence splits (sample 100 chunks)
- [ ] 10.3.5 Benchmark chunking throughput (should be ≥100 docs/sec for non-PDF)

### 10.4 Regression Tests

- [ ] 10.4.1 Compare chunk quality before/after for 10 PMC articles
- [ ] 10.4.2 Compare chunk quality before/after for 10 CT.gov studies
- [ ] 10.4.3 Verify downstream retrieval quality unchanged (P95 latency, Recall@10)

### 10.5 Performance Tests (New)

- [ ] 10.5.1 **Chunking Latency Benchmarks** (per profile):
  - [ ] pmc-imrad: P95 <2s per document (target: 30 docs/min)
  - [ ] ctgov-registry: P95 <1s per document (target: 60 docs/min)
  - [ ] spl-loinc: P95 <1.5s per document (target: 40 docs/min)
  - [ ] guideline: P95 <1s per document (target: 60 docs/min)

- [ ] 10.5.2 **MinerU Performance Benchmarks**:
  - [ ] Latency: P95 <30s per PDF (20-30 pages)
  - [ ] Throughput: 2-3 PDFs/second (GPU-accelerated)
  - [ ] Success Rate: >95% (excluding malformed PDFs)
  - [ ] GPU Utilization: 60-80% during processing

- [ ] 10.5.3 **Token Overflow Rate**:
  - [ ] Measure token overflows per profile
  - [ ] Target: <1% across all profiles
  - [ ] If >1%, tune token budgets in profiles

- [ ] 10.5.4 **Load Testing** (k6):
  - [ ] 100 concurrent ingestion jobs, 5-minute duration
  - [ ] Validate chunking service stability
  - [ ] Monitor memory usage (no leaks)
  - [ ] Check error rate <1%

- [ ] 10.5.5 **Soak Test** (24-hour):
  - [ ] Continuous chunking load (10 docs/sec)
  - [ ] Monitor memory growth
  - [ ] Validate no performance degradation

### 10.6 Contract Tests (New)

- [ ] 10.6.1 **REST API Contract**:
  - [ ] Schemathesis tests for `/v1/ingest/{source}` with `chunking_profile` parameter
  - [ ] Validate response includes chunk count, profile used
  - [ ] Test invalid profile name (expect 400 Bad Request)

- [ ] 10.6.2 **GraphQL API Contract**:
  - [ ] GraphQL Inspector for schema changes
  - [ ] Test `IngestClinicalTrial` mutation with chunking options
  - [ ] Validate response types match schema

- [ ] 10.6.3 **gRPC API Contract**:
  - [ ] Buf breaking change detection on proto files
  - [ ] Test `SubmitIngestionJob` RPC with chunking config
  - [ ] Validate proto message compatibility

### 10.7 Table Preservation Tests (New)

- [ ] 10.7.1 **HTML Preservation**:
  - [ ] Test 10 tables with high rectangularization uncertainty
  - [ ] Verify HTML preserved verbatim
  - [ ] Validate `is_unparsed_table=true` flag set

- [ ] 10.7.2 **Rectangularization Decision**:
  - [ ] Test 10 tables with high confidence
  - [ ] Verify rectangularization applied
  - [ ] Validate cell structure preserved

- [ ] 10.7.3 **Table Chunk Metadata**:
  - [ ] Verify `table_html` field populated
  - [ ] Verify `intent_hint="ae"` for adverse event tables
  - [ ] Verify `section_label` includes table context

---

## 11. API Integration (New)

### 11.1 REST API Updates

- [ ] 11.1.1 Update `/v1/ingest/{source}` endpoint:
  - [ ] Add `chunking_profile` parameter (optional, default="default")
  - [ ] Add `chunking_options` object (preserve_tables_html, sentence_splitter, custom_token_budget)
  - [ ] Update OpenAPI schema

- [ ] 11.1.2 Add `/v1/chunking/profiles` endpoint:
  - [ ] GET: List available chunking profiles
  - [ ] Response includes profile name, description, domain, target tokens

- [ ] 11.1.3 Add `/v1/chunking/validate` endpoint:
  - [ ] POST: Validate chunk quality for a document
  - [ ] Returns metrics: chunk count, token overflow rate, section label coverage

### 11.2 GraphQL API Updates

- [ ] 11.2.1 Update `IngestionInput` type:
  - [ ] Add `chunkingProfile: String = "default"` field
  - [ ] Add `options: ChunkingOptions` field

- [ ] 11.2.2 Add `ChunkingOptions` input type:
  - [ ] preserveTablesHtml: Boolean = true
  - [ ] sentenceSplitter: String = "syntok"
  - [ ] customTokenBudget: Int

- [ ] 11.2.3 Add `chunkingProfiles` query:
  - [ ] Returns list of available profiles
  - [ ] Includes profile metadata (name, domain, target tokens)

### 11.3 gRPC API Updates

- [ ] 11.3.1 Update `IngestionJobRequest` proto:
  - [ ] Add `chunking_profile` field
  - [ ] Add `ChunkingOptions` message type

- [ ] 11.3.2 Update `IngestionJobResponse` proto:
  - [ ] Add `estimated_chunks` field
  - [ ] Add `profile_used` field

- [ ] 11.3.3 Compile proto files:
  - [ ] Run `buf generate`
  - [ ] Run `buf breaking` to check for breaking changes

---

## 12. Error Handling & Taxonomy (New)

### 12.1 Define Error Types

- [ ] 12.1.1 **ProfileNotFoundError**:
  - [ ] Raised when requested profile doesn't exist
  - [ ] HTTP 400 Bad Request
  - [ ] Message: "Chunking profile '{profile}' not found. Available: {profiles}"

- [ ] 12.1.2 **TokenizerMismatchError**:
  - [ ] Raised when tokenizer doesn't align with embedding model
  - [ ] HTTP 500 Internal Server Error
  - [ ] Message: "Tokenizer '{tokenizer}' incompatible with embedding model '{model}'"

- [ ] 12.1.3 **ChunkingFailedError**:
  - [ ] Raised when chunking process fails
  - [ ] HTTP 500 Internal Server Error
  - [ ] Includes detailed error message and stack trace

- [ ] 12.1.4 **MineruOutOfMemoryError**:
  - [ ] Raised when GPU runs out of memory during PDF processing
  - [ ] HTTP 503 Service Unavailable
  - [ ] Message: "GPU out of memory. Retry later or reduce PDF size."

- [ ] 12.1.5 **MineruGpuUnavailableError**:
  - [ ] Raised when GPU not available for MinerU
  - [ ] HTTP 503 Service Unavailable
  - [ ] Message: "GPU required for PDF processing. Check GPU availability."

### 12.2 Error Handling Implementation

- [ ] 12.2.1 Add error handlers to gateway:
  - [ ] Map custom exceptions to HTTP status codes
  - [ ] Return RFC 7807 Problem Details format

- [ ] 12.2.2 Add error logging:
  - [ ] Log all errors with correlation ID
  - [ ] Include context (profile, document ID, stage)

- [ ] 12.2.3 Add error metrics:
  - [ ] Count errors by type: `medicalkg_chunking_errors_total{error_type}`

---

## 13. Performance Optimization

- [ ] 11.1 Batch sentence segmentation (process 10 documents at once)
- [ ] 11.2 Cache scispaCy model loading (singleton pattern)
- [ ] 11.3 Parallelize chunking for independent documents (asyncio)
- [ ] 11.4 Benchmark: scispaCy vs syntok throughput (choose per profile)
- [ ] 11.5 Validate: Chunking throughput ≥100 docs/sec for non-PDF sources

---

## 12. Monitoring & Observability

- [ ] 12.1 Add Prometheus metrics:
  - [ ] `chunking_documents_total` (by profile)
  - [ ] `chunking_duration_seconds` (P50, P95, P99 by profile)
  - [ ] `chunking_chunks_per_document` (histogram by profile)
  - [ ] `chunking_failures_total` (by profile, error type)
  - [ ] `mineru_gate_triggered_total`
  - [ ] `postpdf_start_triggered_total`
- [ ] 12.2 Add CloudEvents for chunking lifecycle:
  - [ ] `chunking.started`
  - [ ] `chunking.completed`
  - [ ] `chunking.failed`
  - [ ] `mineru.gate.waiting`
  - [ ] `postpdf.start.triggered`
- [ ] 12.3 Log chunk quality metrics:
  - [ ] Average chunk length (chars, tokens)
  - [ ] Section label distribution
  - [ ] Intent hint distribution
- [ ] 12.4 Create Grafana dashboard: `Medical_KG_Chunking_Quality.json`

---

## 13. Documentation

- [ ] 13.1 Update `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md`:
  - [ ] Add Section 3.5 "Clinical-Aware Chunking Architecture"
  - [ ] Remove legacy chunking references
  - [ ] Add profile configuration table
- [ ] 13.2 Create `docs/guides/chunking-profiles.md`:
  - [ ] IMRaD profile guide with examples
  - [ ] Registry profile guide with examples
  - [ ] SPL profile guide with examples
  - [ ] Guideline profile guide with examples
- [ ] 13.3 Create `docs/runbooks/mineru-two-phase-gate.md`:
  - [ ] How to trigger `postpdf-start`
  - [ ] Troubleshooting stuck PDF jobs
  - [ ] MinerU failure handling
- [ ] 13.4 Update `README.md` with new dependencies

---

## 14. Production Deployment

- [ ] 14.1 Deploy to production with feature branch merge (no legacy code remains)
- [ ] 14.2 Validate all 4 profiles end-to-end
- [ ] 14.3 Test MinerU two-phase gate with 3 PDF sources
- [ ] 14.4 Monitor chunk quality metrics for 48 hours
- [ ] 14.5 Verify codebase reduction: ≥40% fewer lines in chunking/parsing
- [ ] 14.6 Emergency rollback: revert entire feature branch if critical issues

---

**Total Tasks**: 240+ across 14 work streams
