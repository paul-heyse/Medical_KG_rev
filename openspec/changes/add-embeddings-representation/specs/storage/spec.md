# Storage Capability: Spec Delta

## MODIFIED Requirements

### Requirement: Dense Vector Storage (FAISS Primary)

Dense vector storage SHALL use FAISS HNSW index as primary storage with GPU-accelerated search, replacing scattered storage across multiple backends.

**Previous Behavior**: Dense vectors stored in FAISS (primary), Neo4j (secondary), and ad-hoc pickle files (tertiary) with unclear source of truth.

**New Behavior**: FAISS SHALL be the single source of truth for dense vectors, with optional Neo4j vector index for graph-constrained KNN queries only (<5% of queries).

#### Scenario: Add dense embeddings to FAISS index

- **GIVEN** a batch of 150 dense embeddings (4096-D vectors) from vLLM
- **WHEN** the storage service writes embeddings to FAISS
- **THEN** vectors are added to FAISS HNSW index with chunk IDs as index keys
- **AND** index is persisted to disk at `/data/faiss/chunks_qwen3_v1.bin`
- **AND** GPU-accelerated index is used if GPU available

#### Scenario: FAISS KNN search with GPU acceleration

- **GIVEN** a query vector (4096-D) and k=10
- **WHEN** the retrieval service searches FAISS index
- **THEN** FAISS returns 10 nearest neighbors with distances
- **AND** search latency is P95 <50ms for 10M vectors (GPU)
- **AND** recall@10 is ≥95% (HNSW quality)

#### Scenario: FAISS index save/load for persistence

- **GIVEN** a FAISS index with 1M vectors
- **WHEN** the service saves the index to disk
- **THEN** the index is moved from GPU to CPU before saving
- **AND** saved to `/data/faiss/chunks_qwen3_v1.bin`
- **WHEN** the service loads the index on startup
- **THEN** the index is loaded from disk and moved to GPU
- **AND** search performance is restored (<50ms P95)

#### Scenario: Incremental indexing (append mode)

- **GIVEN** an existing FAISS index with 1M vectors
- **WHEN** the service adds 1000 new vectors
- **THEN** vectors are appended to the existing index (no full rebuild)
- **AND** search remains available during indexing
- **AND** new vectors are searchable immediately after addition

#### Scenario: Neo4j vector index opt-in for graph queries

- **GIVEN** a query requiring graph-constrained KNN (e.g., "Find similar documents within 2-hop neighborhood of Document X")
- **WHEN** the retrieval service determines graph constraint is needed
- **THEN** the service uses Neo4j vector index (not FAISS)
- **AND** Neo4j vectors are synced from FAISS (FAISS is source of truth)
- **AND** query includes both graph traversal and vector similarity

---

### Requirement: Sparse Signal Storage (OpenSearch rank_features)

Sparse signal storage SHALL use OpenSearch `rank_features` field to store SPLADE term weights, enabling BM25+SPLADE fusion queries without separate index.

**Previous Behavior**: Sparse signals stored as custom JSON fields or separate index with no standardized fusion support.

**New Behavior**: SPLADE term weights SHALL be stored in OpenSearch `rank_features` field, enabling native BM25+SPLADE fusion in single query.

#### Scenario: Write SPLADE term weights to rank_features

- **GIVEN** a sparse embedding with term weights: `{"hba1c": 2.8, "reduction": 2.1, "significant": 1.9, ...}` (400 terms)
- **WHEN** the storage service writes to OpenSearch
- **THEN** term weights are stored in `splade_terms` field (type: `rank_features`)
- **AND** the field is indexed for efficient rank_feature queries
- **AND** storage overhead is ~300 bytes per chunk

#### Scenario: OpenSearch mapping for rank_features

- **GIVEN** the OpenSearch chunks index
- **WHEN** the mapping is created or updated
- **THEN** the mapping includes:
  ```json
  {
    "properties": {
      "splade_terms": {
        "type": "rank_features"
      }
    }
  }
  ```
- **AND** rank_features field accepts term-weight dictionaries

#### Scenario: BM25 + SPLADE fusion query

- **GIVEN** a query "diabetes treatment" requiring hybrid search
- **WHEN** the retrieval service searches OpenSearch
- **THEN** the query combines BM25 (match on `text` field) and SPLADE (rank_feature on `splade_terms` field)
- **AND** query structure:
  ```json
  {
    "query": {
      "bool": {
        "should": [
          {"match": {"text": {"query": "diabetes treatment", "boost": 1.0}}},
          {"rank_feature": {"field": "splade_terms", "boost": 2.0}}
        ]
      }
    }
  }
  ```
- **AND** query latency is P95 <200ms (acceptable 50ms overhead for +15% recall)

#### Scenario: Index size impact of SPLADE terms

- **GIVEN** 1M chunks in OpenSearch without SPLADE terms (baseline index size: 10GB)
- **WHEN** SPLADE terms are added (400 terms per chunk, ~300 bytes/chunk)
- **THEN** index size increases to ~13GB (+30%)
- **AND** this is acceptable trade-off for +15% recall improvement

---

### Requirement: Embedding Metadata Storage (Neo4j)

Embedding metadata SHALL be stored in Neo4j linking chunks to embedding model, version, namespace, and generation timestamp, enabling provenance tracking and A/B testing analysis.

**Previous Behavior**: Minimal or no embedding metadata stored, making it unclear which model generated which embeddings.

**New Behavior**: Neo4j SHALL store comprehensive embedding metadata for each chunk, including namespace, model version, generation timestamp, and performance metrics.

#### Scenario: Store embedding metadata in Neo4j

- **GIVEN** a chunk with dense embedding from vLLM and sparse embedding from Pyserini
- **WHEN** the storage service writes embedding metadata to Neo4j
- **THEN** Neo4j stores:
  ```cypher
  MATCH (c:Chunk {chunk_id: $chunk_id})
  CREATE (e1:Embedding {
      embedding_id: $embedding_id_dense,
      namespace: "single_vector.qwen3.4096.v1",
      model: "Qwen/Qwen2.5-Coder-1.5B",
      model_version: "v1",
      dim: 4096,
      generated_at: datetime(),
      generation_duration_ms: 15
  })
  CREATE (c)-[:HAS_EMBEDDING]->(e1)
  ```
- **AND** similar metadata for sparse embedding

#### Scenario: Query embedding metadata for A/B testing

- **GIVEN** chunks embedded with two namespaces: `qwen3.v1` (control) and `qwen3.v2` (treatment)
- **WHEN** an analyst queries retrieval metrics by namespace
- **THEN** Neo4j returns:
  ```cypher
  MATCH (c:Chunk)-[:HAS_EMBEDDING]->(e:Embedding)
  WHERE e.namespace IN ["single_vector.qwen3.4096.v1", "single_vector.qwen3.4096.v2"]
  RETURN e.namespace, count(c) as chunk_count, avg(retrieval_recall) as avg_recall
  ```
- **AND** analyst can compare `qwen3.v1` vs `qwen3.v2` recall

#### Scenario: Embedding provenance for compliance

- **GIVEN** a retrieved chunk with embedding-based ranking
- **WHEN** a compliance officer requests provenance
- **THEN** Neo4j provides:
  - Source chunk ID
  - Embedding namespace and model version
  - Generation timestamp
  - Embedding service used (vLLM vs Pyserini)
- **AND** provenance trail is complete for audit

---

## REMOVED Requirements

### Requirement: Neo4j as Primary Dense Vector Storage (Removed)

**Removed**: The requirement for Neo4j vector index as primary dense vector storage is **REMOVED** in favor of FAISS.

**Reason**: Neo4j vector search is slower (200ms P95 vs FAISS 50ms P95), less efficient for large-scale KNN, and creates unclear source of truth when vectors exist in both Neo4j and FAISS.

**Migration**: Neo4j vector index is now opt-in for graph-constrained KNN queries only (<5% of queries). All standard KNN queries route to FAISS.

---

### Requirement: Ad-Hoc Pickle File Storage (Removed)

**Removed**: The requirement for storing embeddings in ad-hoc pickle files is **REMOVED** entirely.

**Reason**: Pickle files are brittle, lack versioning, create operational confusion, and do not support efficient search.

**Migration**: All pickle file embeddings migrated to FAISS. No new pickle file storage allowed.

