# Chunking Capability: Spec Delta

## ADDED Requirements

### Requirement: ChunkerPort Protocol Interface

The system SHALL provide a unified `ChunkerPort` protocol interface that all chunking strategies implement, enabling pluggable chunking implementations without coupling to specific strategies.

**Rationale**: Eliminates fragmentation across 8 custom chunkers, enables testability, and supports profile-based chunking.

#### Scenario: Register and retrieve chunker by profile

- **GIVEN** a chunking profile "pmc-imrad" exists
- **WHEN** the system requests a chunker for that profile
- **THEN** the registry returns a `ChunkerPort` implementation configured for IMRaD literature chunking
- **AND** the chunker produces chunks with heading-aware boundaries

#### Scenario: Chunker protocol compliance validation

- **GIVEN** a custom chunker implementation
- **WHEN** the implementation is registered with the runtime registry
- **THEN** the system validates it implements the `chunk(document: Document, profile: str) -> list[Chunk]` signature
- **AND** raises `TypeError` if the signature is incorrect

---

### Requirement: Profile-Based Clinical Domain Chunking

The system SHALL support declarative YAML profiles that encode domain-specific chunking rules for literature (IMRaD), registry data (CT.gov), drug labels (SPL/LOINC), and clinical guidelines.

**Rationale**: Clinical document structure varies dramatically by domain; profiles enable domain-aware chunking without hardcoded logic per source.

#### Scenario: IMRaD profile respects heading boundaries

- **GIVEN** a PMC article with sections: Abstract, Introduction, Methods, Results, Discussion
- **WHEN** chunked with profile "pmc-imrad" (target_tokens=450, respect_boundaries=["heading"])
- **THEN** no chunk spans across IMRaD section boundaries
- **AND** each chunk includes `section_label` (e.g., "Methods", "Results")
- **AND** figure captions remain attached to their sections

#### Scenario: Registry profile produces atomic outcome units

- **GIVEN** a CT.gov study with 3 outcome measures
- **WHEN** chunked with profile "ctgov-registry" (overlap_tokens=0, respect_boundaries=["outcome_measure"])
- **THEN** each outcome measure becomes a separate chunk
- **AND** each chunk has `intent_hint="outcome"`
- **AND** effect pairs (intervention, comparator, measure, result) remain together

#### Scenario: SPL profile uses LOINC section codes

- **GIVEN** an SPL drug label with LOINC-coded sections
- **WHEN** chunked with profile "spl-label" (respect_boundaries=["loinc_section"])
- **THEN** each chunk's `section_label` includes LOINC code (e.g., "LOINC:34089-3 Indications")
- **AND** "Warnings and Precautions" and "Adverse Reactions" are separate chunks

#### Scenario: Guideline profile isolates recommendation units

- **GIVEN** a clinical guideline with 5 recommendations (statement + strength + grade)
- **WHEN** chunked with profile "guideline" (respect_boundaries=["recommendation_unit"])
- **THEN** each recommendation becomes an atomic chunk
- **AND** evidence tables remain attached to their recommendation
- **AND** chunk includes `intent_hint="recommendation"`

---

### Requirement: Library-Delegated Chunking Strategies

The system SHALL delegate chunking to proven open-source libraries (langchain-text-splitters, LlamaIndex, Hugging Face tokenizers, syntok) via thin wrappers, replacing all custom chunking implementations.

**Rationale**: Reduces maintenance burden by 43%, leverages community improvements, ensures industry-standard behavior.

#### Scenario: LangChain recursive character splitter for default chunking

- **GIVEN** a profile with `chunker_type: langchain_recursive`
- **WHEN** a document is chunked with that profile
- **THEN** the system uses `langchain_text_splitters.RecursiveCharacterTextSplitter` internally
- **AND** token counting uses the Qwen3-aligned tokenizer
- **AND** chunk boundaries respect profile-defined separators (["\n\n", "\n", ". ", " "])

#### Scenario: LlamaIndex sentence window for coherence-sensitive chunking

- **GIVEN** a profile with `chunker_type: llamaindex_sentence_window`
- **WHEN** a document is chunked with that profile
- **THEN** the system uses `llama_index.node_parser.SentenceWindowNodeParser` internally
- **AND** each chunk includes 3 sentences (window_size=3) for coherence
- **AND** char offsets map to the original document accurately

#### Scenario: Hugging Face tokenizers for biomedical sentence segmentation

- **GIVEN** a profile with `sentence_splitter: huggingface`
- **WHEN** a document with biomedical abbreviations ("Fig. 1", "et al.") is chunked
- **THEN** the system loads the configured Hugging Face tokenizer (`MEDICAL_KG_SENTENCE_MODEL`) for sentence boundaries
- **AND** biomedical abbreviations do not trigger false sentence splits
- **AND** sentence boundaries align with clinical context transitions

#### Scenario: syntok for fast sentence splitting

- **GIVEN** a profile with `sentence_splitter: syntok`
- **WHEN** a batch of 100 non-biomedical documents is chunked
- **THEN** the system uses `syntok.segmenter` for sentence boundaries
- **AND** throughput is ≥100 docs/sec
- **AND** char offsets are preserved for each sentence

---

### Requirement: Complete Chunk Provenance with Clinical Context

The system SHALL produce chunks with complete provenance metadata including `doc_id`, `char_offsets`, `section_label`, `intent_hint`, `page_bbox` (for PDFs), and chunking configuration, enabling span-grounded extraction and reproducibility.

**Rationale**: Downstream extraction, SHACL validation, and graph writes require precise offsets; clinical routing needs section labels and intent hints.

#### Scenario: Chunk provenance enables span-grounded extraction

- **GIVEN** a chunk with `char_offsets: (1450, 1920)` and `section_label: "Results"`
- **WHEN** an entity extractor identifies "significant reduction in HbA1c (p<0.001)" in that chunk
- **THEN** the extraction can link back to the exact source document span at chars 1450-1920
- **AND** the SHACL validator can verify the extraction came from the "Results" section

#### Scenario: PDF chunks include page and bounding box

- **GIVEN** a PDF processed by MinerU with page/bbox maps
- **WHEN** a chunk is created from page 5, bbox (120, 450, 480, 720)
- **THEN** the chunk includes `page_bbox: {"page": 5, "bbox": [120, 450, 480, 720]}`
- **AND** the chunk's `char_offsets` map to the MinerU-produced IR, not raw PDF bytes

#### Scenario: Chunk metadata enables A/B testing

- **GIVEN** two chunking profiles: "pmc-imrad-v1" and "pmc-imrad-v2" (different overlap)
- **WHEN** the same document is chunked with both profiles
- **THEN** each chunk includes `metadata.chunking_profile` and `metadata.chunker_version`
- **AND** downstream metrics (retrieval Recall@10) can be stratified by profile version

#### Scenario: Intent hints guide retrieval and extraction

- **GIVEN** a chunk from CT.gov with `intent_hint="eligibility"`
- **WHEN** a query asks "What are the inclusion criteria?"
- **THEN** the retrieval system can boost chunks with `intent_hint="eligibility"`
- **AND** the extraction system can prioritize eligibility-specific templates

---

### Requirement: Filter Chain for Normalization Without Evidence Loss

The system SHALL apply composable filters (drop_boilerplate, exclude_references, deduplicate_page_furniture, preserve_tables_html) to normalize chunks while preserving essential clinical evidence, particularly tables.

**Rationale**: Boilerplate pollutes retrieval; references are low-value; table HTML preservation prevents information loss from failed rectangularization.

#### Scenario: Boilerplate removal without losing content

- **GIVEN** a PDF with repeated running headers "Journal of Medicine 2024" on every page
- **WHEN** the `drop_boilerplate` filter is applied
- **THEN** running headers are removed from all chunks
- **AND** main content (Methods, Results) remains intact
- **AND** char offsets are adjusted to reflect removed text

#### Scenario: References section exclusion

- **GIVEN** a PMC article with "References" section at the end
- **WHEN** the `exclude_references` filter is applied
- **THEN** all text after the "References" heading is excluded
- **AND** the last chunk ends with the "Discussion" section
- **AND** no reference citations appear in chunks

#### Scenario: Table HTML preservation when rectangularization uncertain

- **GIVEN** a complex table with merged cells and nested headers (MinerU confidence=0.65)
- **WHEN** the `preserve_tables_html` filter is applied (threshold=0.8)
- **THEN** the table is preserved as HTML in a dedicated chunk
- **AND** the chunk is tagged with `is_unparsed_table=true`
- **AND** `table_html` field contains the original HTML structure

---

### Requirement: Token Budget Enforcement Aligned with Embedding Model

The system SHALL enforce token budgets using the tokenizer aligned with the target embedding model (Qwen3 via HuggingFace `transformers.AutoTokenizer`), preventing chunks from exceeding model context limits.

**Rationale**: Misaligned token counting causes embedding failures; Qwen3 tokenizer ensures budget honesty before embedding stage.

#### Scenario: Token budget prevents oversize chunks

- **GIVEN** a profile with `target_tokens=450` and embedding model Qwen3
- **WHEN** a document is chunked
- **THEN** the system uses `transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")` for token counting
- **AND** no chunk exceeds 450 tokens (measured with Qwen3 tokenizer)
- **AND** chunks are split at appropriate boundaries to respect budget

#### Scenario: Tokenizer caching for performance

- **GIVEN** a batch of 100 documents to chunk
- **WHEN** the chunking service initializes
- **THEN** the Qwen3 tokenizer is loaded once and cached
- **AND** subsequent chunking operations reuse the cached tokenizer
- **AND** tokenizer initialization overhead is amortized across the batch

---

## MODIFIED Requirements

### Requirement: Chunking Service API (Modified)

The chunking service API SHALL accept a `profile` parameter and produce chunks with complete provenance metadata.

**Previous Behavior**: `chunk_document(document: Document) -> list[str]` returned plain text chunks without metadata.

**New Behavior**: `chunk_document(document: Document, profile: str) -> list[Chunk]` SHALL return `Chunk` objects with provenance.

#### Scenario: Profile-aware chunking API

- **GIVEN** a document from PMC
- **WHEN** the gateway calls `chunk_document(document, profile="pmc-imrad")`
- **THEN** the chunking service returns `list[Chunk]` with section labels, offsets, and intent hints
- **AND** each `Chunk` object includes `chunking_profile="pmc-imrad"` in metadata

#### Scenario: Default profile per source

- **GIVEN** a document from CT.gov
- **WHEN** the gateway calls `chunk_document(document)` without explicit profile
- **THEN** the system defaults to profile "ctgov-registry"
- **AND** chunks include registry-specific intent hints ("eligibility", "outcome", "ae")

---

### Requirement: Chunking Error Handling (Modified)

Chunking error handling SHALL include profile validation, tokenizer alignment checks, and explicit failure modes with no silent fallbacks.

**Previous Behavior**: Chunking failures resulted in empty lists or generic exceptions.

**New Behavior**: The system SHALL raise explicit exceptions (`ProfileNotFoundError`, `TokenizerMismatchError`, `ChunkingFailedError`) with detailed error messages.

#### Scenario: Profile not found error

- **GIVEN** a request to chunk with profile "nonexistent-profile"
- **WHEN** the chunking service attempts to load the profile
- **THEN** the system raises `ProfileNotFoundError("Profile 'nonexistent-profile' not found in config/chunking/profiles/")`
- **AND** the error includes a list of available profiles
- **AND** the job status is marked `chunking_failed` in the ledger

#### Scenario: Tokenizer mismatch detection

- **GIVEN** a profile configured for embedding model "BGE" but current model is "Qwen3"
- **WHEN** the chunking service initializes
- **THEN** the system raises `TokenizerMismatchError("Profile expects BGE tokenizer, but Qwen3 is active")`
- **AND** the error includes remediation steps (update profile or switch embedding model)

---

## REMOVED Requirements

### Requirement: Custom Chunking Strategies (Removed)

**Removed**: The requirement for custom chunking strategies (`SemanticSplitter`, `SlidingWindow`, `CoherenceSplitter`, etc.) is **REMOVED** in favor of library-delegated implementations via `ChunkerPort`.

**Reason**: Custom chunkers created maintenance burden, duplicated proven library functionality, and lacked community support.

**Migration**: All custom chunker logic has been replaced with LangChain, LlamaIndex, scispaCy, and syntok wrappers. Existing chunks remain valid; new ingestion uses library-based implementations.

---

### Requirement: Source-Specific Chunking Logic (Removed)

**Removed**: The requirement for source-specific chunking logic embedded in adapters (`.split_document()` methods) is **REMOVED** in favor of profile-based chunking.

**Reason**: Coupling chunking to adapters created duplication and inconsistency; profiles provide better separation of concerns.

**Migration**: All adapter `.split_document()` methods have been deleted. Adapters now return IR Documents, and chunking is applied uniformly via profiles.

---

### Requirement: Token Counting with Approximate Heuristics (Removed)

**Removed**: The requirement for approximate token counting (`chars / 4`) is **REMOVED** in favor of model-aligned tokenizers.

**Reason**: Approximate token counting caused embedding failures when chunks exceeded model context limits; model-aligned tokenizers ensure accuracy.

**Migration**: All chunking operations now use the Qwen3 tokenizer (`transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")`) for token counting. No backwards compatibility for approximate counts.
