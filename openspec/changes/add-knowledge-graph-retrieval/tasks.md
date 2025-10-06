# Implementation Tasks: Knowledge Graph & Retrieval

## 1. Neo4j Knowledge Graph

- [ ] 1.1 Define graph schema (node labels, relationship types)
- [ ] 1.2 Add Neo4j to docker-compose.yml
- [ ] 1.3 Implement Neo4jClient wrapper
- [ ] 1.4 Create Cypher query templates with MERGE for idempotency
- [ ] 1.5 Add provenance tracking (ExtractionActivity nodes)
- [ ] 1.6 Implement SHACL validation for graph constraints
- [ ] 1.7 Write graph ingestion tests

## 2. Semantic Chunking

- [ ] 2.1 Implement ChunkingService
- [ ] 2.2 Add paragraph-based chunking
- [ ] 2.3 Add section-aware chunking (preserve headers)
- [ ] 2.4 Add table-aware chunking
- [ ] 2.5 Implement max token limits per chunk
- [ ] 2.6 Write chunking tests with sample documents

## 3. OpenSearch Integration

- [ ] 3.1 Add OpenSearch to docker-compose.yml
- [ ] 3.2 Implement OpenSearchClient
- [ ] 3.3 Create index templates for documents and chunks
- [ ] 3.4 Add BM25 full-text indexing
- [ ] 3.5 Add SPLADE sparse vector indexing
- [ ] 3.6 Implement search queries with filters
- [ ] 3.7 Write OpenSearch tests

## 4. FAISS Integration

- [ ] 4.1 Implement FAISSIndex wrapper
- [ ] 4.2 Add dense vector indexing (Qwen embeddings)
- [ ] 4.3 Implement similarity search with k-NN
- [ ] 4.4 Add index persistence and loading
- [ ] 4.5 Write FAISS tests

## 5. Multi-Strategy Retrieval

- [ ] 5.1 Implement RetrievalService
- [ ] 5.2 Add BM25 retrieval strategy
- [ ] 5.3 Add SPLADE sparse retrieval
- [ ] 5.4 Add dense vector retrieval (FAISS)
- [ ] 5.5 Implement fusion ranking (RRF or weighted)
- [ ] 5.6 Add reranker with cross-encoder
- [ ] 5.7 Implement span highlighting
- [ ] 5.8 Write retrieval tests with relevance assertions

## 6. Indexing Pipeline

- [ ] 6.1 Implement IndexingService
- [ ] 6.2 Add document → chunk → embed → index pipeline
- [ ] 6.3 Implement incremental indexing
- [ ] 6.4 Add index refresh and optimization
- [ ] 6.5 Write indexing integration tests

## 7. Query DSL

- [ ] 7.1 Design query DSL for complex retrievals
- [ ] 7.2 Add filter support (date, source, status)
- [ ] 7.3 Add faceted search
- [ ] 7.4 Implement query parsing and validation
- [ ] 7.5 Write query DSL tests
