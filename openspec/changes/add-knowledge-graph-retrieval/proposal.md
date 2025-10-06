# Change Proposal: Knowledge Graph & Retrieval System

## Why

Implement the knowledge graph (Neo4j) for storing entities, claims, and relationships with full provenance, plus multi-strategy retrieval using OpenSearch (BM25 + SPLADE), FAISS (dense vectors), and fusion ranking. Includes chunking, embedding indexing, span-grounded extraction, and SHACL validation.

## What Changes

- Neo4j graph database setup with schema (nodes: Document, Entity, Claim, Evidence, ExtractionActivity; relationships with provenance)
- Cypher query templates for MERGE operations with idempotency
- OpenSearch integration for full-text and SPLADE sparse retrieval
- FAISS integration for dense vector similarity
- Semantic chunking service (paragraph, section, table-aware)
- Multi-strategy retrieval (BM25 + SPLADE + dense + fusion)
- Reranking with cross-encoder
- Span validation and highlighting
- SHACL shapes for graph validation
- Query DSL for complex graph traversals
- Graph export and import utilities

## Impact

- **Affected specs**: NEW capabilities `knowledge-graph`, `retrieval-system`
- **Affected code**:
  - `src/Medical_KG_rev/kg/` - Neo4j integration and Cypher queries
  - `src/Medical_KG_rev/retrieval/` - Multi-strategy retrieval
  - `src/Medical_KG_rev/chunking/` - Semantic chunking
  - `src/Medical_KG_rev/indexing/` - OpenSearch and FAISS indexing
  - `docker-compose.yml` - Add Neo4j, OpenSearch services
  - `tests/kg/`, `tests/retrieval/` - Graph and retrieval tests
