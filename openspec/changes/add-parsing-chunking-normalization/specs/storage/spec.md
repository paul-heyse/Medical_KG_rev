# Storage Capability: Spec Delta

## MODIFIED Requirements

### Requirement: Chunk Storage Schema with Complete Provenance (Modified)

The chunk storage schema SHALL include mandatory provenance fields (`doc_id`, `char_offsets`, `section_label`, `intent_hint`, `page_bbox`, `metadata`) and optional table preservation fields (`is_unparsed_table`, `table_html`).

**Previous Behavior**: Chunks were stored as plain text strings with minimal metadata (doc_id, chunk_id).

**New Behavior**: Chunks SHALL be stored as structured objects with complete provenance, enabling span-grounded extraction, clinical routing, and reproducibility.

#### Scenario: Chunk schema validation enforces required fields

- **GIVEN** a chunk produced by the chunking service
- **WHEN** the chunk is written to storage (OpenSearch, Neo4j)
- **THEN** the storage layer validates the chunk has all required fields:
  - `chunk_id: str`
  - `doc_id: str`
  - `text: str`
  - `char_offsets: tuple[int, int]`
  - `section_label: str`
  - `intent_hint: str`
  - `metadata: dict`
- **AND** if any field is missing, raises `ValidationError` with detailed message
- **AND** the chunk write is rejected (not silently accepted with null fields)

#### Scenario: Chunk metadata enables A/B testing and provenance

- **GIVEN** a chunk stored in OpenSearch
- **WHEN** the chunk record is retrieved
- **THEN** `metadata` includes:
  - `source_system: "pmc"`
  - `chunking_profile: "pmc-imrad"`
  - `chunker_version: "langchain-v0.2.0"`
  - `created_at: "2025-10-07T14:30:00Z"`
- **AND** retrieval filters can stratify by profile version
- **AND** downstream analysis can compare chunk quality across profiles

#### Scenario: PDF chunks store page/bbox for citation

- **GIVEN** a chunk from a PDF processed by MinerU
- **WHEN** the chunk is stored
- **THEN** `page_bbox` includes `{"page": 5, "bbox": [120, 450, 480, 720]}`
- **AND** retrieval results can display "Found on page 5" to users
- **AND** span highlighting can render the exact bounding box overlay

#### Scenario: Table chunks preserve HTML when uncertain

- **GIVEN** a chunk containing a complex table with MinerU confidence=0.65
- **WHEN** the chunk is stored
- **THEN** `is_unparsed_table=true`
- **AND** `table_html` contains the original HTML structure
- **AND** downstream extraction can parse the HTML directly (skips rectangularization)
- **AND** retrieval can render the table as HTML to users

#### Scenario: Char offsets enable span-grounded extraction

- **GIVEN** a chunk with `char_offsets=(14502, 14550)` stored in Neo4j
- **WHEN** an entity extractor identifies "HbA1c reduction" in the chunk
- **THEN** the extraction links back to the source document at chars 14502-14550
- **AND** the SHACL validator verifies the extraction came from the correct section
- **AND** the knowledge graph edge includes `text_span=(14502, 14550)` for provenance

---

### Requirement: Chunk Indexing with Clinical Structure (Modified)

Chunk indexing in OpenSearch SHALL include fields for `section_label`, `intent_hint`, and `chunking_profile`, enabling clinical-aware retrieval filtering and boosting.

**Previous Behavior**: OpenSearch indexed chunks as plain text with `doc_id` and `text` fields only.

**New Behavior**: OpenSearch SHALL index chunks with clinical structure fields, enabling queries like "Results sections only" or "eligibility criteria chunks".

#### Scenario: OpenSearch mapping includes clinical structure fields

- **GIVEN** the OpenSearch chunks index mapping
- **WHEN** the mapping is created
- **THEN** the mapping includes:

  ```json
  {
    "properties": {
      "chunk_id": {"type": "keyword"},
      "doc_id": {"type": "keyword"},
      "text": {"type": "text"},
      "section_label": {"type": "keyword"},
      "intent_hint": {"type": "keyword"},
      "chunking_profile": {"type": "keyword"},
      "page_bbox": {
        "properties": {
          "page": {"type": "integer"},
          "bbox": {"type": "float", "index": false}
        }
      },
      "metadata": {"type": "object", "enabled": false}
    }
  }
  ```

#### Scenario: Retrieval filters by section label

- **GIVEN** a query "What are the adverse events?"
- **WHEN** the retrieval system searches OpenSearch
- **THEN** the query includes filter: `section_label IN ["Adverse Reactions", "LOINC:34084-4"]`
- **AND** only chunks from AE sections are returned
- **AND** "Methods" and "Results" chunks are excluded

#### Scenario: Retrieval boosts by intent hint

- **GIVEN** a query "What are the eligibility criteria?"
- **WHEN** the retrieval system searches OpenSearch
- **THEN** chunks with `intent_hint="eligibility"` receive 3x boost
- **AND** chunks with `intent_hint="outcome"` receive no boost
- **AND** the ranking reflects clinical relevance

#### Scenario: Retrieval stratifies by chunking profile

- **GIVEN** two profile versions "pmc-imrad-v1" and "pmc-imrad-v2" in production
- **WHEN** an analyst queries retrieval metrics
- **THEN** results can be stratified by `chunking_profile` field
- **AND** Recall@10 can be compared across profile versions
- **AND** A/B testing determines optimal profile configuration
