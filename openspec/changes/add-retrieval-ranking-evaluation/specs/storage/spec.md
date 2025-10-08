# Storage Capability: Spec Delta

## MODIFIED Requirements

### Requirement: OpenSearch Index Configuration for Field Boosting

OpenSearch index configuration SHALL include field-level boosting to support BM25F and clinical intent boosting.

**Previous Behavior**: Uniform field weights in OpenSearch mapping, no boosting configuration.

**New Behavior**: Field boosting SHALL be configured in mapping with priorities: `title^3, section_label^2, facet_json^2, intent_hint^2, text^1`.

#### Scenario: Field boosting in OpenSearch mapping

- **GIVEN** the OpenSearch chunks index
- **WHEN** the mapping is created or updated
- **THEN** field boosting is configured:

  ```json
  {
    "mappings": {
      "properties": {
        "title": {"type": "text", "boost": 3.0},
        "section_label": {"type": "keyword", "boost": 2.0},
        "facet_json": {"type": "text", "boost": 2.0},
        "intent_hint": {"type": "keyword", "boost": 2.0},
        "text": {"type": "text", "boost": 1.0}
      }
    }
  }
  ```

#### Scenario: BM25F query with field boosting

- **GIVEN** a query "diabetes treatment"
- **WHEN** BM25F search is executed
- **THEN** OpenSearch applies field boosts in scoring
- **AND** matches in title are weighted 3x higher than matches in text
- **AND** matches in section_label are weighted 2x higher than matches in text

---

### Requirement: OpenSearch BM25 Parameter Configuration

OpenSearch BM25 parameters SHALL be configurable (k1, b) to tune term frequency and document length normalization.

**Previous Behavior**: Default BM25 parameters (k1=1.2, b=0.75) with no configuration.

**New Behavior**: BM25 parameters SHALL be configurable per index, with defaults k1=1.2, b=0.75 (standard).

#### Scenario: Configure custom BM25 parameters

- **GIVEN** an OpenSearch index with custom BM25 settings
- **WHEN** the index settings are applied
- **THEN** BM25 parameters are configured:

  ```json
  {
    "settings": {
      "index": {
        "similarity": {
          "custom_bm25": {
            "type": "BM25",
            "k1": 1.5,
            "b": 0.8
          }
        }
      }
    }
  }
  ```

- **AND** search queries use custom_bm25 similarity

#### Scenario: Tune BM25 parameters for biomedical text

- **GIVEN** biomedical documents with high length variance (abstracts vs full-text)
- **WHEN** BM25 parameter tuning is performed on test set
- **THEN** optimal parameters are determined (e.g., k1=1.3, b=0.7 for biomedical corpus)
- **AND** parameters are updated in index settings

---

### Requirement: OpenSearch Domain-Specific Analyzer

OpenSearch SHALL use a domain-specific analyzer with biomedical synonym filter, lowercase filter, and stopword filter.

**Previous Behavior**: Standard analyzer with generic stopwords.

**New Behavior**: Custom analyzer SHALL include biomedical synonyms (e.g., "diabetes" → "hyperglycemia"), lowercase, and domain-specific stopwords.

#### Scenario: Custom analyzer with biomedical synonyms

- **GIVEN** the OpenSearch chunks index
- **WHEN** the analyzer is configured
- **THEN** custom analyzer includes:

  ```json
  {
    "settings": {
      "analysis": {
        "analyzer": {
          "biomedical_analyzer": {
            "type": "custom",
            "tokenizer": "standard",
            "filter": [
              "lowercase",
              "biomedical_synonym",
              "english_stop"
            ]
          }
        },
        "filter": {
          "biomedical_synonym": {
            "type": "synonym",
            "synonyms_path": "analysis/biomedical_synonyms.txt"
          }
        }
      }
    }
  }
  ```

- **AND** text fields use biomedical_analyzer

#### Scenario: Query expansion via synonym filter

- **GIVEN** a query "diabetes treatment"
- **WHEN** the query is analyzed with biomedical_analyzer
- **THEN** synonyms are expanded: "diabetes" → ["diabetes", "hyperglycemia", "DM"]
- **AND** expanded terms are included in BM25 matching

---

### Requirement: FAISS Index Parameter Optimization

FAISS HNSW index parameters SHALL be optimized for biomedical retrieval (M=32, efConstruction=200, efSearch=64).

**Previous Behavior**: Default FAISS HNSW parameters with no configuration.

**New Behavior**: HNSW parameters SHALL be configured for optimal recall/latency trade-off: M=32 connections, efConstruction=200 build quality, efSearch=64 search quality.

#### Scenario: FAISS HNSW index creation with optimized parameters

- **GIVEN** a FAISS HNSW index for 10M vectors
- **WHEN** the index is created
- **THEN** parameters are set:
  - M=32 (connections per node)
  - efConstruction=200 (build-time candidate list size)
  - efSearch=64 (search-time candidate list size)
- **AND** index achieves P95 <50ms for KNN search with 95%+ recall

#### Scenario: FAISS IVF+HNSW for large-scale

- **GIVEN** a FAISS index with >10M vectors
- **WHEN** index size exceeds 10M threshold
- **THEN** IVF+HNSW index is used instead of HNSW-only
- **AND** parameters: nlist=1024 (IVF clusters), M=32 (HNSW connections)
- **AND** search latency remains <100ms P95 with 93%+ recall

---

### Requirement: Storage Partitioning for Multi-Tenancy

Storage indexes SHALL be partitioned by tenant_id to enforce multi-tenant isolation.

**Previous Behavior**: Single shared index with tenant_id filtering in queries.

**New Behavior**: Separate indexes per tenant (e.g., `tenant-123-chunks`, `tenant-456-chunks`) for strict isolation.

#### Scenario: Per-tenant OpenSearch indexes

- **GIVEN** two tenants: tenant-123 and tenant-456
- **WHEN** chunks are indexed
- **THEN** tenant-123 chunks are written to `tenant-123-chunks` index
- **AND** tenant-456 chunks are written to `tenant-456-chunks` index
- **AND** no cross-tenant data leakage is possible

#### Scenario: Per-tenant FAISS indexes

- **GIVEN** two tenants: tenant-123 and tenant-456
- **WHEN** dense vectors are indexed
- **THEN** tenant-123 vectors are written to `/data/faiss/tenant-123-chunks.bin`
- **AND** tenant-456 vectors are written to `/data/faiss/tenant-456-chunks.bin`
- **AND** search queries target tenant-specific FAISS index

---

## REMOVED Requirements

None (Storage enhancements are additive, no requirements removed)
