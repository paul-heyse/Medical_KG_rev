# Clinical-Aware Parsing, Chunking & Normalization

## Quick Reference

This proposal replaces fragmented, bespoke chunking/parsing with a unified, library-based architecture that respects clinical document structure and enforces GPU-only policies.

### Key Changes

- **ChunkerPort Interface**: Single protocol for all chunking strategies (replaces 8 custom chunkers)
- **Profile-Based Chunking**: Declarative YAML profiles for IMRaD, Registry, SPL, Guideline domains
- **Library Delegation**: LangChain, LlamaIndex, scispaCy, syntok replace custom implementations (43% code reduction)
- **MinerU Two-Phase Gate**: Explicit `postpdf-start` trigger after PDF OCR (no automatic resume)
- **Complete Provenance**: Every chunk has `doc_id`, `char_offsets`, `section_label`, `intent_hint`, `page_bbox`

### Breaking Changes

- ❌ `ChunkingService.chunk_document()` now requires `profile` parameter
- ❌ Chunks now require `section_label`, `intent_hint`, `char_offsets` (non-optional)
- ❌ PDF processing requires explicit `postpdf-start` call (no auto-resume)
- ❌ Table chunks preserve HTML by default (rectangularize opt-in only)

---

## Profile Examples

### IMRaD Profile (PMC Articles)

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
sentence_splitter: scispacy  # Biomedical-aware
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

**Usage**:

```python
from Medical_KG_rev.services.chunking.port import chunk_document

chunks = chunk_document(document, profile="pmc-imrad")
# Returns list[Chunk] with:
# - section_label="Methods", "Results", etc.
# - intent_hint="narrative" or "outcome"
# - char_offsets for span grounding
# - No mid-sentence splits across IMRaD boundaries
```

---

### Registry Profile (CT.gov Studies)

```yaml
# config/chunking/profiles/ctgov-registry.yaml
name: ctgov-registry
domain: registry
chunker_type: langchain_recursive
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
metadata:
  section_label_source: registry_section
  intent_hints:
    EligibilityCriteria: eligibility
    OutcomeMeasure: outcome
    AdverseEventsTable: ae
    ResultsSection: results
```

**Usage**:

```python
chunks = chunk_document(ct_gov_study, profile="ctgov-registry")
# Returns atomic units:
# - Eligibility criteria as single chunk (intent_hint="eligibility")
# - Each outcome measure separate (intent_hint="outcome")
# - AE tables atomic with effect pairs together (intent_hint="ae")
```

---

### SPL Profile (Drug Labels)

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
sentence_splitter: scispacy
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
chunks = chunk_document(spl_label, profile="spl-label")
# Returns chunks with LOINC-coded section labels:
# - section_label="LOINC:34089-3 Indications"
# - section_label="LOINC:34084-4 Adverse Reactions"
# - Enables mapping to RxNorm/MedDRA via LOINC codes
```

---

### Guideline Profile

```yaml
# config/chunking/profiles/guideline.yaml
name: guideline
domain: guideline
chunker_type: langchain_recursive
target_tokens: 350
overlap_tokens: 0  # Recommendations are atomic
respect_boundaries:
  - recommendation_unit  # Statement + strength + grade
  - evidence_table
sentence_splitter: syntok
preserve_tables_as_html: true
filters:
  - drop_boilerplate
metadata:
  section_label_source: recommendation_id
  intent_hints:
    Recommendation: recommendation
    EvidenceTable: evidence
```

**Usage**:

```python
chunks = chunk_document(guideline, profile="guideline")
# Returns recommendation units:
# - Each recommendation (statement + strength + grade) as single chunk
# - Evidence tables attached to recommendations
# - intent_hint="recommendation" enables facet summaries
```

---

## MinerU Two-Phase Gate

### Flow Diagram

```
1. Submit PDF Job
   ↓
2. Download PDF → ledger: pdf_downloaded=true
   ↓
3. MinerU Processing (GPU-only)
   ├─ Success → ledger: pdf_ir_ready=true → HALT
   └─ Failure → ledger: status=mineru_failed → ABORT (no CPU fallback)
   ↓
4. Manual Inspection (optional quality check)
   ↓
5. Trigger postpdf-start
   ├─ Manual: POST /v1/jobs/{job_id}/postpdf-start
   └─ Auto: Dagster sensor after 5 min delay
   ↓
6. Resume Pipeline: Chunking → Embedding → Indexing
```

### API Usage

```bash
# Submit PDF job
curl -X POST https://api.medical-kg.example.com/v1/ingest/pmc \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "data": {
      "type": "IngestionRequest",
      "attributes": {
        "identifiers": ["PMC8675309"],
        "include_pdf": true
      }
    }
  }'

# Response: job_id="job-abc123"

# Wait for MinerU to complete (poll ledger)
curl https://api.medical-kg.example.com/v1/jobs/job-abc123

# Response: status="pdf_ir_ready", waiting for postpdf-start

# Trigger postpdf-start (manual or automated)
curl -X POST https://api.medical-kg.example.com/v1/jobs/job-abc123/postpdf-start \
  -H "Authorization: Bearer $TOKEN"

# Response: status="chunking", postpdf_start_triggered_at="2025-10-07T14:30:00Z"
```

### Dagster Auto-Trigger Configuration

```yaml
# config/orchestration/sensors/pdf_ir_ready_sensor.yaml
name: pdf_ir_ready_sensor
poll_interval_seconds: 30
auto_trigger_delay_minutes: 5  # Auto-trigger after 5 min wait
trigger_source: auto_sensor
```

---

## Library Integration

### LangChain Recursive Character Splitter

**Use Case**: Default for structure-aware segmentation

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=450 * 4,  # Approximate tokens→chars
    chunk_overlap=50 * 4,
    length_function=lambda text: len(tokenizer.encode(text)),
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_text(document.text)
```

**Replaces**: `CustomSplitter`, `RecursiveSplitter` (8 custom implementations)

---

### LlamaIndex Sentence Window Node Parser

**Use Case**: Coherence-sensitive chunking for clinical narratives

```python
from llama_index.node_parser import SentenceWindowNodeParser

parser = SentenceWindowNodeParser(
    window_size=3,  # 3 sentences per window
    window_metadata_key="window",
    original_text_metadata_key="original_sentence"
)

nodes = parser.get_nodes_from_documents([document])
```

**Replaces**: `SemanticSplitter`, `CoherenceSplitter`

---

### scispaCy Biomedical Sentence Segmentation

**Use Case**: Biomedical-aware sentence boundaries for IMRaD literature

```python
import spacy

nlp = spacy.load("en_core_sci_sm")

def segment_sentences(text: str) -> list[tuple[int, int, str]]:
    doc = nlp(text)
    return [
        (sent.start_char, sent.end_char, sent.text)
        for sent in doc.sents
    ]
```

**Handles**: "Fig. 1", "et al.", "p<0.001" without false sentence splits

**Replaces**: `BiomedicalSentenceSplitter`

---

### syntok Fast Sentence Splitter

**Use Case**: High-throughput batches (10x faster than scispaCy)

```python
from syntok import segmenter

def segment_sentences_fast(text: str) -> list[tuple[int, int, str]]:
    sentences = []
    for paragraph in segmenter.process(text):
        for sentence in paragraph:
            tokens = [token.value for token in sentence]
            sent_text = " ".join(tokens)
            # Offset tracking implementation
            sentences.append((start_char, end_char, sent_text))
    return sentences
```

**Replaces**: `SimpleSentenceSplitter`

---

### unstructured XML/HTML Parser

**Use Case**: JATS XML, SPL XML, HTML guidelines

```python
from unstructured.partition.xml import partition_xml
from unstructured.partition.html import partition_html

# JATS XML
elements = partition_xml(filename="article.xml")
ir_document = map_elements_to_ir(elements)

# SPL XML
elements = partition_xml(filename="label.xml")
loinc_sections = extract_loinc_codes(elements)

# HTML Guidelines
elements = partition_html(filename="guideline.html")
recommendations = extract_recommendations(elements)
```

**Replaces**: `XMLParser`, `JATSParser`, `SPLParser`, `HTMLParser`

---

## Chunk Schema

### Pydantic Model

```python
from pydantic import BaseModel, Field
from typing import Any

class Chunk(BaseModel):
    """A chunk with complete provenance."""

    # Identity
    chunk_id: str = Field(description="Unique chunk identifier")
    doc_id: str = Field(description="Source document ID")

    # Content
    text: str = Field(description="Chunk text")
    char_offsets: tuple[int, int] = Field(description="Start/end chars in source")

    # Clinical Structure
    section_label: str = Field(description="IMRaD section, LOINC code, or registry section")
    intent_hint: str = Field(description="narrative, eligibility, outcome, ae, dose, recommendation")

    # PDF Provenance (optional)
    page_bbox: dict | None = Field(default=None, description="Page + bounding box")

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="source_system, chunking_profile, chunker_version, created_at"
    )

    # Table Handling
    is_unparsed_table: bool = Field(default=False, description="HTML preserved?")
    table_html: str | None = Field(default=None, description="Original HTML if unparsed")
```

### Example Chunk (IMRaD)

```json
{
  "chunk_id": "pmc:PMC8675309:chunk-42",
  "doc_id": "pmc:PMC8675309",
  "text": "We observed a significant reduction in HbA1c levels (7.2% to 6.1%, p<0.001) after 12 weeks of treatment. The treatment group showed superior glycemic control compared to placebo.",
  "char_offsets": [14502, 14680],
  "section_label": "Results",
  "intent_hint": "outcome",
  "page_bbox": {
    "page": 5,
    "bbox": [120, 450, 480, 520]
  },
  "metadata": {
    "source_system": "pmc",
    "chunking_profile": "pmc-imrad",
    "chunker_version": "langchain-v0.2.0",
    "created_at": "2025-10-07T14:30:00Z"
  },
  "is_unparsed_table": false,
  "table_html": null
}
```

### Example Chunk (Registry, AE Table)

```json
{
  "chunk_id": "ctgov:NCT04267848:chunk-15",
  "doc_id": "ctgov:NCT04267848",
  "text": "Adverse Event: Nausea | Treatment Group: 23/100 (23%) | Placebo Group: 8/98 (8%) | Risk Ratio: 2.8 (95% CI: 1.3-6.1)",
  "char_offsets": [8920, 9050],
  "section_label": "AdverseEventsTable",
  "intent_hint": "ae",
  "page_bbox": null,
  "metadata": {
    "source_system": "ctgov",
    "chunking_profile": "ctgov-registry",
    "chunker_version": "langchain-v0.2.0",
    "created_at": "2025-10-07T14:32:00Z"
  },
  "is_unparsed_table": true,
  "table_html": "<table><tr><th>Adverse Event</th><th>Treatment</th><th>Placebo</th><th>Risk Ratio</th></tr>...</table>"
}
```

---

## Filter Chain

### Available Filters

1. **drop_boilerplate**: Remove headers, footers, "Page X of Y"
2. **exclude_references**: Drop "References" section
3. **deduplicate_page_furniture**: Remove repeated running headers
4. **preserve_tables_html**: Keep HTML when rectangularization confidence <0.8

### Configuration

```yaml
# In profile YAML
filters:
  - drop_boilerplate
  - exclude_references
  - deduplicate_page_furniture
  - preserve_tables_html
```

### Implementation

```python
def apply_filters(chunks: list[Chunk], filters: list[str]) -> list[Chunk]:
    """Apply filter chain."""
    for filter_name in filters:
        chunks = FILTER_REGISTRY[filter_name](chunks)
    return chunks

# Filter registry
FILTER_REGISTRY = {
    "drop_boilerplate": drop_boilerplate_filter,
    "exclude_references": exclude_references_filter,
    "deduplicate_page_furniture": deduplicate_page_furniture_filter,
    "preserve_tables_html": preserve_tables_html_filter,
}
```

---

## Dependencies Added

```txt
# Parsing & Chunking
langchain-text-splitters>=0.2.0
llama-index-core>=0.10.0
scispacy>=0.5.4
en-core-sci-sm @ https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/en_core_sci_sm-0.5.4/en_core_sci_sm-0.5.4.tar.gz
syntok>=1.4.4
unstructured[local-inference]>=0.12.0
tiktoken>=0.6.0
transformers>=4.38.0
```

---

## Codebase Reduction

### Before

| Component | Files | Lines |
|-----------|-------|-------|
| Custom chunkers | 6 | 420 |
| Custom parsers | 3 | 415 |
| Sentence splitters | 1 | 140 |
| **Total** | **10** | **975** |

### After

| Component | Files | Lines |
|-----------|-------|-------|
| ChunkerPort interface | 1 | 50 |
| Profile system | 3 | 120 |
| Library wrappers | 5 | 310 |
| **Total** | **9** | **480** |

### Reduction

- **Lines Removed**: 975
- **Lines Added**: 480
- **Net Reduction**: 495 lines (51% reduction)
- **Files Reduced**: 10 → 9

---

## Migration Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Download scispaCy model: `python -m spacy download en_core_sci_sm`
- [ ] Create profile YAMLs in `config/chunking/profiles/`
- [ ] Run ledger migration: `python scripts/migrate_ledger_for_pdf_gate.py`
- [ ] Update gateway endpoints to accept `chunking_profile` parameter
- [ ] Test all 4 profiles end-to-end (IMRaD, Registry, SPL, Guideline)
- [ ] Test MinerU two-phase gate with 3 PDF sources
- [ ] Deploy to production (no legacy code remains)
- [ ] Monitor chunk quality metrics for 48 hours

---

## Validation

```bash
# Validate proposal
cd /home/paul/Medical_KG_rev
openspec validate add-parsing-chunking-normalization --strict

# Expected: Change is valid
```

---

## Status

**Created**: 2025-10-07
**Status**: Ready for implementation
**Timeline**: 6 weeks (2 weeks build, 2 weeks testing, 2 weeks deployment)
**Breaking Changes**: 4 (ChunkingService API, Chunk schema, PDF gate, table handling)
**Affected Capabilities**: 4 (chunking, parsing, orchestration, storage)
