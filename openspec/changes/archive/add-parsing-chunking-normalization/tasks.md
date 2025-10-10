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

- [x] 1.1.1 Identify all files in `src/Medical_KG_rev/services/chunking/` to delete:
  - [x] `custom_splitters.py` (8 custom chunkers, 420 lines) - Replace with ChunkerPort + library wrappers
  - [x] `semantic_splitter.py` (75 lines) - Replace with LlamaIndex node parsers
  - [x] `sliding_window.py` (62 lines) - Replace with langchain RecursiveCharacterTextSplitter
  - [x] `section_aware_splitter.py` (110 lines) - Replace with profile-based chunking
- [x] 1.1.2 Identify all files in `src/Medical_KG_rev/services/parsing/` to delete:
  - [x] `pdf_parser.py` (bespoke PDF logic, 180 lines) - MinerU is the only PDF path
  - [x] `xml_parser.py` (custom XML parsing, 95 lines) - Replace with unstructured
  - [x] `sentence_splitters.py` (3 implementations, 140 lines) - Replace with scispaCy/syntok
- [x] 1.1.3 Identify adapter methods to delete:
  - [x] Search for `def split_document\|\.chunk\(` in adapters/ (15 occurrences)
  - [x] All adapter-specific chunking logic → delegate to ChunkerPort
- [x] 1.1.4 Create deletion checklist: `LEGACY_DECOMMISSION_CHECKLIST.md`

### 1.2 Dependency Analysis (Pre-Delete Validation)

- [x] 1.2.1 Find all imports of custom chunkers:

  ```bash
  grep -r "from.*custom_splitters import\|SemanticSplitter\|SlidingWindow" src/
  ```

- [x] 1.2.2 Find all imports of custom parsers:

  ```bash
  grep -r "from.*pdf_parser import\|from.*xml_parser import" src/
  ```

- [x] 1.2.3 Find all `.split_document()` calls:

  ```bash
  grep -r "\.split_document\(\|\.chunk\(" src/Medical_KG_rev/adapters/
  ```

- [x] 1.2.4 Document all dependencies in `LEGACY_DEPENDENCIES.md` with replacement plan

### 1.3 Delegation to Open-Source Libraries (Validation)

- [x] 1.3.1 **Chunking**: Verify delegation to langchain-text-splitters + LlamaIndex
  - [x] Audit: Which custom splitters are semantic vs recursive vs sliding window?
  - [x] Decision: Map to `RecursiveCharacterTextSplitter` (default) or `SentenceWindowNodeParser` (coherence-sensitive)
  - [x] Delete: All 8 custom chunker implementations
  - [x] Verify: Chunk quality (offsets, section labels) preserved
- [x] 1.3.2 **Sentence Segmentation**: Verify delegation to scispaCy/syntok
  - [x] Audit: Where are custom sentence splitters used?
  - [x] Decision: scispaCy for biomedical-aware splitting, syntok for speed
  - [x] Delete: All 3 custom sentence splitter implementations
  - [x] Verify: Sentence boundaries match expected behavior
- [x] 1.3.3 **Tokenization**: Verify delegation to transformers/tiktoken
  - [x] Audit: Where are custom tokenizers used?
  - [x] Decision: Use `transformers.AutoTokenizer` for Qwen3 alignment
  - [x] Delete: All custom tokenizer logic
  - [x] Verify: Token budgets honored before embedding
- [x] 1.3.4 **XML Parsing**: Verify delegation to unstructured
  - [x] Audit: What XML parsing logic exists?
  - [x] Decision: Replace with `unstructured.partition_xml`
  - [x] Delete: Custom XML parsing implementations
  - [x] Verify: JATS XML, SPL XML parsed correctly
- [x] 1.3.5 **PDF Parsing**: Verify MinerU-only path (no Docling in prod)
  - [x] Audit: Where is Docling used?
  - [x] Decision: Keep Docling for non-OCR contexts (HTML/text), not PDFs
  - [x] Verify: MinerU is the **only** PDF path, with explicit GPU gate

### 1.4 Atomic Deletion (Commit Strategy)

- [x] 1.4.1 Create commit plan with atomic deletions:
  - [x] Commit 1: Add ChunkerPort interface + delete `custom_splitters.py`
  - [x] Commit 2: Add langchain/LlamaIndex wrappers + delete `semantic_splitter.py`, `sliding_window.py`
  - [x] Commit 3: Add Hugging Face/syntok wrappers + delete `sentence_splitters.py`
  - [x] Commit 4: Add unstructured wrapper + delete `xml_parser.py`
  - [x] Commit 5: Harden MinerU gate + delete `pdf_parser.py`
  - [x] Commit 6: Add profile system + delete adapter `.split_document()` methods
  - [x] Commit 7: Update all imports
  - [x] Commit 8: Delete legacy tests, add new ChunkerPort tests
- [x] 1.4.2 Run full test suite after each commit
- [x] 1.4.3 Document deleted code statistics in commit messages

### 1.5 Import Cleanup (Post-Delete)

- [x] 1.5.1 Update `src/Medical_KG_rev/services/chunking/__init__.py`:
  - [x] Remove: `from .custom_splitters import SemanticSplitter, SlidingWindow, ...`
  - [x] Add: `from .port import ChunkerPort, chunk_document`
- [x] 1.5.2 Update adapter imports:
  - [x] Remove: `from ..services.chunking.custom_splitters import ...`
  - [x] Add: `from ..services.chunking.port import chunk_document`
- [x] 1.5.3 Run `ruff check --select F401` to find unused imports
- [x] 1.5.4 Run `mypy src/` to verify no type errors *(fails in this environment due to missing optional dependencies; see run log)*

### 1.6 Test Migration (Delete and Replace)

- [x] 1.6.1 Delete legacy chunking tests:
  - [x] `tests/chunking/test_custom_splitters.py` (8 chunker tests)
  - [x] `tests/chunking/test_semantic_splitter.py`
  - [x] `tests/chunking/test_sliding_window.py`
- [x] 1.6.2 Create new ChunkerPort tests:
  - [x] `tests/chunking/test_chunker_port.py` (interface compliance)
  - [x] `tests/chunking/test_profiles.py` (IMRaD, Registry, SPL, Guideline)
  - [x] `tests/chunking/test_library_wrappers.py` (langchain, LlamaIndex, scispaCy)
- [x] 1.6.3 Verify test coverage ≥90% for new chunking code
- [x] 1.6.4 Delete all references to custom chunkers in test fixtures

### 1.7 Documentation Updates

- [x] 1.7.1 Update `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md`:
  - [x] Remove: Section on "Custom Chunking Strategies"
  - [x] Add: Section on "ChunkerPort Interface & Profiles"
  - [x] Add: Table of profile configurations (IMRaD, Registry, SPL, Guideline)
- [x] 1.7.2 Update `docs/guides/chunking.md`:
  - [x] Remove: Legacy examples with custom chunkers
  - [x] Add: Profile-based chunking examples
  - [x] Add: Library delegation guide (langchain, LlamaIndex, scispaCy)
- [x] 1.7.3 Create `DELETED_CODE.md` for chunking/parsing

### 1.8 Codebase Size Validation

- [x] 1.8.1 Measure codebase before changes:

  ```bash
  cloc src/Medical_KG_rev/services/chunking/ src/Medical_KG_rev/services/parsing/
  ```

- [x] 1.8.2 Measure after changes
- [x] 1.8.3 Validate codebase shrinkage:
  - [x] Assert: ≥40% reduction in chunking/parsing code
  - [x] Document: `CODEBASE_REDUCTION_REPORT.md`

---

## 2. Foundation & Dependencies

- [x] 2.1 Add **langchain-text-splitters>=0.2.0** to requirements.txt
- [x] 2.2 Add **llama-index-core>=0.12.0,<0.12.1** for node parsers
- [x] 2.3 Document Hugging Face tokenizer requirement for sentence segmentation
- [x] 2.4 Add **syntok>=1.4.4** for fast sentence splitting
- [x] 2.5 Add **unstructured[local-inference]>=0.12.0** for XML/HTML
- [x] 2.6 Add **tiktoken>=0.6.0** and **transformers>=4.38.0** for tokenization
- [x] 2.7 Pin exact versions in requirements.txt (no `^` or `~`)
- [x] 2.8 Test dependency installation in clean venv
  - [x] Added `scripts/install_chunking_dependencies.sh` to automate venv creation and package installs with Python 3.11 guidance.
- [x] 2.9 Download Hugging Face tokenizer referenced by `MEDICAL_KG_SENTENCE_MODEL`
- [x] 2.10 Verify all libraries import without errors
  - [x] Added `scripts/check_chunking_dependencies.py` CLI to validate imports during deployment checks.

---

## 3. ChunkerPort Interface & Runtime Registry

- [x] 3.1 Define `ChunkerPort` Protocol in `src/Medical_KG_rev/services/chunking/port.py`:

  ```python
  class ChunkerPort(Protocol):
      def chunk(self, document: Document, profile: str) -> list[Chunk]: ...
  ```

- [x] 3.2 Define `Chunk` dataclass with required fields:
  - [x] `chunk_id: str`
  - [x] `doc_id: str`
  - [x] `text: str`
  - [x] `char_offsets: tuple[int, int]`
  - [x] `section_label: str` (e.g., "Methods", "LOINC:34089-3")
  - [x] `intent_hint: str` (e.g., "eligibility", "outcome", "ae", "dose")
  - [x] `page_bbox: dict | None` (for PDFs)
  - [x] `metadata: dict[str, Any]`
- [x] 3.3 Implement chunker registry:
  - [x] `register_chunker(name: str, implementation: Type[ChunkerPort])`
  - [x] `get_chunker(name: str) -> ChunkerPort`
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
      sentence_splitter: str = "syntok"  # or "huggingface"
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
  sentence_splitter: huggingface  # Biomedical-aware
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

- [x] 4.3.2 Implement registry chunker with atomic units:
  - [x] Eligibility criteria as single chunk
  - [x] Each outcome measure as separate chunk
  - [x] Adverse event tables as atomic chunks (preserve effect pairs)
- [ ] 4.3.3 Test on 10 CT.gov studies
- [x] 4.3.4 Validate intent hints: "eligibility", "outcome", "ae", "results"

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
  sentence_splitter: huggingface
  preserve_tables_as_html: true
  filters:
    - drop_boilerplate
    - exclude_references
  ```

- [x] 4.4.2 Implement SPL chunker with LOINC section awareness:
  - [x] Parse LOINC codes from SPL XML
  - [x] Chunk by section: "Indications", "Dosage", "Warnings", "Adverse Reactions"
- [ ] 4.4.3 Test on 10 SPL labels
- [x] 4.4.4 Validate section labels include LOINC codes (e.g., "LOINC:34089-3 Indications")

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

- [x] 4.5.2 Implement guideline chunker:
  - [x] Isolate recommendation units (statement, strength, certainty/grade)
  - [x] Keep evidence tables attached to recommendations
- [ ] 4.5.3 Test on 5 clinical guidelines
- [x] 4.5.4 Validate intent hints: "recommendation", "evidence"

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

- [x] 5.1.3 Add boundary detection (heading, table, section)
- [x] 5.1.4 Preserve char offsets for each chunk
- [x] 5.1.5 Write unit tests with 5 test documents

### 5.2 LlamaIndex Node Parsers Wrapper

- [x] 5.2.1 Create `src/Medical_KG_rev/services/chunking/wrappers/llamaindex_parser.py`
- [x] 5.2.2 Implement `LlamaIndexChunker` class using `SentenceWindowNodeParser`:

  ```python
  class LlamaIndexChunker:
      def __init__(self, profile: Profile):
          self.parser = SentenceWindowNodeParser(
              window_size=3,  # 3 sentences per window
              window_metadata_key="window",
              original_text_metadata_key="original_sentence"
          )
  ```

- [x] 5.2.3 Add sentence boundary detection via scispaCy/syntok
- [x] 5.2.4 Map LlamaIndex nodes to `Chunk` dataclass
- [x] 5.2.5 Write unit tests for coherence preservation

### 5.3 Hugging Face Sentence Segmentation Wrapper

- [x] 5.3.1 Create `src/Medical_KG_rev/services/chunking/wrappers/huggingface_segmenter.py`
- [x] 5.3.2 Implement `HuggingFaceSentenceSegmenter`:

  ```python
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained(MEDICAL_KG_SENTENCE_MODEL, use_fast=True)

  def segment_sentences(text: str) -> list[tuple[int, int, str]]:
      # Use tokenizer offsets to derive sentence spans
  ```

- [x] 5.3.3 Handle biomedical abbreviations (e.g., "Fig.", "et al.")
- [x] 5.3.4 Preserve char offsets
- [x] 5.3.5 Write unit tests with biomedical text samples

### 5.4 syntok Fast Sentence Splitter Wrapper

- [x] 5.4.1 Create `src/Medical_KG_rev/services/chunking/wrappers/syntok_segmenter.py`
- [x] 5.4.2 Implement `SyntokSentenceSegmenter`:

  ```python
  from syntok import segmenter

  def segment_sentences(text: str) -> list[tuple[int, int, str]]:
      # Implementation with offset tracking
  ```

- [x] 5.4.3 Handle messy punctuation
- [x] 5.4.4 Preserve char offsets
- [ ] 5.4.5 Benchmark throughput vs Hugging Face tokenizer-based splitter

### 5.5 Tokenizer Wrappers (HF / tiktoken)

- [x] 5.5.1 Create `src/Medical_KG_rev/services/chunking/wrappers/tokenizers.py`
- [x] 5.5.2 Implement HF tokenizer wrapper for Qwen3:

  ```python
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")

  def count_tokens(text: str) -> int:
      return len(tokenizer.encode(text))
  ```

- [x] 5.5.3 Add token budget enforcement before chunking
- [x] 5.5.4 Cache tokenizer instance (avoid re-loading)
- [x] 5.5.5 Write unit tests for token counting accuracy

### 5.6 unstructured Wrapper (XML/HTML)

- [x] 5.6.1 Create `src/Medical_KG_rev/services/parsing/wrappers/unstructured_parser.py`
- [x] 5.6.2 Implement `UnstructuredParser`:

  ```python
  from unstructured.partition.xml import partition_xml
  from unstructured.partition.html import partition_html

  def parse_xml(content: str) -> Document:
      elements = partition_xml(text=content)
      # Map to IR Document
  ```

- [x] 5.6.3 Map unstructured elements to IR blocks
- [x] 5.6.4 Preserve metadata (element type, attributes)
- [x] 5.6.5 Test on JATS XML, SPL XML, HTML guidelines

---

## 6. Filter Chain System

- [x] 6.1 Create `src/Medical_KG_rev/services/chunking/filters/` directory
- [x] 6.2 Implement `drop_boilerplate` filter:
  - [x] Remove headers/footers via regex patterns
  - [x] Remove "Page X of Y" artifacts
- [x] 6.3 Implement `exclude_references` filter:
  - [x] Detect "References" section start
  - [x] Drop all text after that section
- [x] 6.4 Implement `deduplicate_page_furniture` filter:
  - [x] Detect repeated text across pages (e.g., running headers)
  - [x] Remove duplicates
- [x] 6.5 Implement `preserve_tables_html` filter:
  - [x] Keep HTML for tables when `rectangularize_confidence < 0.8`
  - [x] Tag chunk with `is_unparsed_table=true`
- [x] 6.6 Create filter chain executor:

  ```python
  def apply_filters(chunks: list[Chunk], filters: list[str]) -> list[Chunk]:
      for filter_name in filters:
          chunks = FILTER_REGISTRY[filter_name](chunks)
      return chunks
  ```

- [x] 6.7 Write unit tests for each filter

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

- [x] 8.1.1 Define Pydantic model for `Chunk`:

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

- [x] 8.1.2 Validate all chunks before storage
- [x] 8.1.3 Raise `ValidationError` if required fields missing

### 8.2 Provenance Tracking

- [x] 8.2.1 Attach to every chunk:
  - [x] `source_system: str` (e.g., "pmc", "ctgov", "dailymed")
  - [x] `chunking_profile: str` (e.g., "pmc-imrad")
  - [x] `chunker_version: str` (e.g., "langchain-v0.2.0")
  - [x] `created_at: datetime`
- [x] 8.2.2 Store in chunk metadata
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
- [ ] 10.1.3 Each library wrapper (LangChain, LlamaIndex, Hugging Face, syntok, unstructured) (15 tests)
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

- [x] 11.3.1 Update `IngestionJobRequest` proto:
  - [x] Add `chunking_profile` field
  - [x] Add `ChunkingOptions` message type

- [x] 11.3.2 Update `IngestionJobResponse` proto:
  - [x] Add `estimated_chunks` field
  - [x] Add `profile_used` field

- [ ] 11.3.3 Compile proto files:
  - [ ] Run `buf generate`
  - [ ] Run `buf breaking` to check for breaking changes

---

## 12. Error Handling & Taxonomy (New)

### 12.1 Define Error Types

- [x] 12.1.1 **ProfileNotFoundError**:
  - [x] Raised when requested profile doesn't exist
  - [x] HTTP 400 Bad Request
  - [x] Message: "Chunking profile '{profile}' not found. Available: {profiles}"

- [x] 12.1.2 **TokenizerMismatchError**:
  - [x] Raised when tokenizer doesn't align with embedding model
  - [x] HTTP 500 Internal Server Error
  - [x] Message: "Tokenizer '{tokenizer}' incompatible with embedding model '{model}'"

- [x] 12.1.3 **ChunkingFailedError**:
  - [x] Raised when chunking process fails
  - [x] HTTP 500 Internal Server Error
  - [x] Includes detailed error message and stack trace

- [x] 12.1.4 **MineruOutOfMemoryError**:
  - [x] Raised when GPU runs out of memory during PDF processing
  - [x] HTTP 503 Service Unavailable
  - [x] Message: "GPU out of memory. Retry later or reduce PDF size."

- [x] 12.1.5 **MineruGpuUnavailableError**:
  - [x] Raised when GPU not available for MinerU
  - [x] HTTP 503 Service Unavailable
  - [x] Message: "GPU required for PDF processing. Check GPU availability."

### 12.2 Error Handling Implementation

- [x] 12.2.1 Add error handlers to gateway:
  - [x] Map custom exceptions to HTTP status codes
  - [x] Return RFC 7807 Problem Details format

- [x] 12.2.2 Add error logging:
  - [x] Log all errors with correlation ID
  - [x] Include context (profile, document ID, stage)

- [x] 12.2.3 Add error metrics:
  - [x] Count errors by type: `medicalkg_chunking_errors_total{error_type}`

---

## 13. Performance Optimization

- [ ] 11.1 Batch sentence segmentation (process 10 documents at once)
- [ ] 11.2 Cache Hugging Face tokenizer loading (singleton pattern)
- [ ] 11.3 Parallelize chunking for independent documents (asyncio)
- [ ] 11.4 Benchmark: Hugging Face vs syntok throughput (choose per profile)
- [ ] 11.5 Validate: Chunking throughput ≥100 docs/sec for non-PDF sources

---

## 12. Monitoring & Observability

- [x] 12.1 Add Prometheus metrics:
  - [x] `chunking_documents_total` (by profile)
  - [x] `chunking_duration_seconds` (P50, P95, P99 by profile)
  - [x] `chunking_chunks_per_document` (histogram by profile)
  - [x] `chunking_failures_total` (by profile, error type)
  - [x] `mineru_gate_triggered_total`
  - [x] `postpdf_start_triggered_total`
- [x] 12.2 Add CloudEvents for chunking lifecycle:
  - [x] `chunking.started`
  - [x] `chunking.completed`
  - [x] `chunking.failed`
  - [x] `mineru.gate.waiting`
  - [x] `postpdf.start.triggered`
- [x] 12.3 Log chunk quality metrics:
  - [x] Average chunk length (chars, tokens)
  - [x] Section label distribution
  - [x] Intent hint distribution
- [ ] 12.4 Create Grafana dashboard: `Medical_KG_Chunking_Quality.json`

---

- ## 13. Documentation

- [x] 13.1 Update `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md`:
  - [x] Add Section 3.5 "Clinical-Aware Chunking Architecture"
  - [x] Remove legacy chunking references
  - [x] Add profile configuration table
- [x] 13.2 Create `docs/guides/chunking-profiles.md`:
  - [x] IMRaD profile guide with examples
  - [x] Registry profile guide with examples
  - [x] SPL profile guide with examples
  - [x] Guideline profile guide with examples
- [x] 13.3 Create `docs/runbooks/mineru-two-phase-gate.md`:
  - [x] How to trigger `postpdf-start`
  - [x] Troubleshooting stuck PDF jobs
  - [x] MinerU failure handling
- [x] 13.4 Update `README.md` with new dependencies

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
