# Implementation Tasks: Knowledge Graph & Retrieval

## 1. Neo4j Knowledge Graph

- [x] 1.1 Define graph schema (node labels, relationship types)
- [x] 1.2 Add Neo4j to docker-compose.yml
- [x] 1.3 Implement Neo4jClient wrapper
- [x] 1.4 Create Cypher query templates with MERGE for idempotency
- [x] 1.5 Add provenance tracking (ExtractionActivity nodes)
- [x] 1.6 Implement SHACL validation for graph constraints
- [x] 1.7 Write graph ingestion tests

## 2. Semantic Chunking

- [x] 2.1 Implement ChunkingService
- [x] 2.2 Add paragraph-based chunking
- [x] 2.3 Add section-aware chunking (preserve headers)
- [x] 2.4 Add table-aware chunking
- [x] 2.5 Implement max token limits per chunk
- [x] 2.6 Write chunking tests with sample documents

## 3. OpenSearch Integration

- [x] 3.1 Add OpenSearch to docker-compose.yml
- [x] 3.2 Implement OpenSearchClient
- [x] 3.3 Create index templates for documents and chunks
- [x] 3.4 Add BM25 full-text indexing
- [x] 3.5 Add SPLADE sparse vector indexing
- [x] 3.6 Implement search queries with filters
- [x] 3.7 Write OpenSearch tests

## 4. FAISS Integration

- [x] 4.1 Implement FAISSIndex wrapper
- [x] 4.2 Add dense vector indexing (Qwen embeddings)
- [x] 4.3 Implement similarity search with k-NN
- [x] 4.4 Add index persistence and loading
- [x] 4.5 Write FAISS tests

## 5. Multi-Strategy Retrieval

- [x] 5.1 Implement RetrievalService
- [x] 5.2 Add BM25 retrieval strategy
- [x] 5.3 Add SPLADE sparse retrieval
- [x] 5.4 Add dense vector retrieval (FAISS)
- [x] 5.5 Implement fusion ranking (RRF or weighted)
- [x] 5.6 Add reranker with cross-encoder
- [x] 5.7 Implement span highlighting
- [x] 5.8 Write retrieval tests with relevance assertions

## 6. Indexing Pipeline

- [x] 6.1 Implement IndexingService
- [x] 6.2 Add document → chunk → embed → index pipeline
- [x] 6.3 Implement incremental indexing
- [x] 6.4 Add index refresh and optimization
- [x] 6.5 Write indexing integration tests

## 7. Query DSL

- [x] 7.1 Design query DSL for complex retrievals
- [x] 7.2 Add filter support (date, source, status)
- [x] 7.3 Add faceted search
- [x] 7.4 Implement query parsing and validation
- [x] 7.5 Write query DSL tests
