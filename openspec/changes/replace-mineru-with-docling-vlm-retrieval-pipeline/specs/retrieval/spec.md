## ADDED Requirements

### Requirement: Hybrid Retrieval System
The system SHALL provide a hybrid retrieval system combining BM25, SPLADE, and Qwen3 retrieval strategies for improved accuracy.

#### Scenario: Parallel retrieval execution
- **WHEN** a search query is submitted
- **THEN** the system SHALL execute BM25, SPLADE, and Qwen3 retrieval in parallel
- **AND** SHALL join results on chunk_id
- **AND** SHALL apply fusion ranking to combine results
- **AND** SHALL return unified results with provenance tracking

#### Scenario: Retrieval method identification
- **WHEN** returning search results
- **THEN** the system SHALL identify which retrieval method contributed each result
- **AND** SHALL include confidence scores from each method
- **AND** SHALL provide method-specific metadata
- **AND** SHALL maintain query performance metrics

#### Scenario: Retrieval performance optimization
- **WHEN** executing retrieval operations
- **THEN** the system SHALL optimize for query performance across strategies
- **AND** SHALL implement caching for repeated queries
- **AND** SHALL provide query result caching and invalidation
- **AND** SHALL monitor retrieval performance and accuracy

### Requirement: BM25 Retrieval with Medical Structure
The system SHALL provide BM25 retrieval with structured fields optimized for medical document search.

#### Scenario: Multi-field BM25 indexing
- **WHEN** indexing documents for BM25 retrieval
- **THEN** the system SHALL create separate fields for title, section_headers, paragraph, caption, table_text
- **AND** SHALL apply appropriate field boosts (high for title, moderate for caption)
- **AND** SHALL use medical term-preserving analyzers
- **AND** SHALL support MeSH/UMLS synonym expansion

#### Scenario: BM25 query processing
- **WHEN** processing a search query for BM25
- **THEN** the system SHALL generate multi-field queries with appropriate boosts
- **AND** SHALL expand terms using medical synonym filters
- **AND** SHALL return ranked results with field-specific scoring
- **AND** SHALL include result provenance and confidence scores

### Requirement: SPLADE-v3 with Rep-Max Aggregation
The system SHALL provide SPLADE-v3 retrieval with Rep-Max aggregation for learned sparse retrieval.

#### Scenario: SPLADE chunk segmentation
- **WHEN** processing chunks for SPLADE indexing
- **THEN** the system SHALL segment chunks into ≤512-token segments
- **AND** SHALL use the same SPLADE tokenizer for chunking and retrieval
- **AND** SHALL maintain segment order and boundaries for aggregation
- **AND** SHALL handle edge cases (very short/long chunks)

#### Scenario: SPLADE Rep-Max aggregation
- **WHEN** aggregating SPLADE segments for a chunk
- **THEN** the system SHALL merge segment vectors by taking maximum weight per term
- **AND** SHALL create one learned-sparse vector per chunk
- **AND** SHALL apply sparsity thresholds and quantization
- **AND** SHALL store as Lucene impact index for efficient retrieval

#### Scenario: SPLADE query processing
- **WHEN** processing a search query for SPLADE
- **THEN** the system SHALL encode the query with SPLADE tokenizer/model
- **AND** SHALL score against the SPLADE impact index
- **AND** SHALL return ranked results with sparse similarity scores
- **AND** SHALL include query preprocessing and normalization

### Requirement: Qwen3 Dense Retrieval
The system SHALL provide Qwen3 4096-dimension dense retrieval for semantic search.

#### Scenario: Qwen3 embedding generation
- **WHEN** processing chunks for Qwen3 indexing
- **THEN** the system SHALL generate 4096-dimension vectors
- **AND** SHALL use contextualized text for embedding input
- **AND** SHALL include section_path and caption context
- **AND** SHALL store vectors with chunk_id mapping

#### Scenario: Qwen3 query processing
- **WHEN** processing a search query for Qwen3
- **THEN** the system SHALL generate query embedding with Qwen3 model
- **AND** SHALL perform ANN search against Qwen3 index
- **AND** SHALL return ranked results with cosine similarity scores
- **AND** SHALL include vector normalization and distance calculation

## MODIFIED Requirements

### Requirement: Retrieval Service Interface
The retrieval service interface SHALL support hybrid retrieval combining multiple strategies.

#### Scenario: Hybrid retrieval configuration
- **WHEN** configuring retrieval strategies
- **THEN** the service SHALL allow selection of BM25, SPLADE, Qwen3, or hybrid modes
- **AND** SHALL support feature flag control for gradual migration
- **AND** SHALL maintain backward compatibility with existing retrieval APIs
- **AND** SHALL provide method-specific configuration options

#### Scenario: Retrieval performance optimization
- **WHEN** executing retrieval operations
- **THEN** the service SHALL optimize for query performance across strategies
- **AND** SHALL implement caching for repeated queries
- **AND** SHALL provide query result caching and invalidation
- **AND** SHALL monitor retrieval performance and accuracy

## REMOVED Requirements

### Requirement: Single Retrieval Strategy Limitation
**Reason**: Replaced by hybrid retrieval system
**Migration**: Hybrid retrieval maintains backward compatibility while providing improved accuracy

The system no longer limits retrieval to a single strategy but combines multiple complementary approaches.
