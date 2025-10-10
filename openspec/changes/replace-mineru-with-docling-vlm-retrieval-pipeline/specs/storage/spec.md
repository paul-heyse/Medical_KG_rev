## ADDED Requirements

### Requirement: Chunk Store Database
The system SHALL provide a chunk store database using DuckDB for storing processed document chunks with comprehensive metadata.

#### Scenario: Chunk storage and retrieval
- **WHEN** chunks are processed from Docling VLM output
- **THEN** the system SHALL store chunks in DuckDB with all metadata
- **AND** SHALL include chunk_id, doc_id, doctags_sha, page_no, bbox, element_label
- **AND** SHALL store contextualized_text and content_only_text fields
- **AND** SHALL provide efficient retrieval by doc_id or chunk_id

#### Scenario: Chunk store analytics and views
- **WHEN** analyzing chunk processing results
- **THEN** the system SHALL provide convenience views for analytics
- **AND** SHALL include views for chunks by label, token length distribution
- **AND** SHALL support queries for chunk quality metrics
- **AND** SHALL provide views for chunk processing statistics

#### Scenario: Chunk store validation and consistency
- **WHEN** storing or retrieving chunks
- **THEN** the system SHALL validate chunk data integrity
- **AND** SHALL ensure chunk_id uniqueness and doc_id relationships
- **AND** SHALL detect and quarantine malformed chunks
- **AND** SHALL maintain referential integrity between chunks and documents

### Requirement: Separate Index Storage Model
The system SHALL provide separate storage for BM25, SPLADE, and Qwen3 indexes with manifest-based version tracking.

#### Scenario: Index storage layout
- **WHEN** storing retrieval indexes
- **THEN** the system SHALL use separate directories for each index type
- **AND** SHALL store BM25 as Lucene index in `/indexes/bm25_index/`
- **AND** SHALL store SPLADE as Lucene impact index in `/indexes/splade_v3_repmax/`
- **AND** SHALL store Qwen3 as FAISS files in `/vectors/qwen3_8b_4096.*`
- **AND** SHALL maintain separate manifests for each index type

#### Scenario: Index manifest management
- **WHEN** creating or updating indexes
- **THEN** the system SHALL create/update manifest files
- **AND** SHALL record model versions, preprocessing parameters, build timestamps
- **AND** SHALL include input checksums and versions in manifests
- **AND** SHALL support manifest validation and consistency checks

#### Scenario: Index rebuild capability
- **WHEN** model versions or parameters change
- **THEN** the system SHALL support rebuilding individual indexes
- **AND** SHALL allow rebuilding from chunk store without touching other indexes
- **AND** SHALL maintain index consistency during rebuild operations
- **AND** SHALL provide rollback capability for failed rebuilds

### Requirement: Raw Artifact and DocTags Storage
The system SHALL provide storage for raw PDF artifacts and DocTags blobs with integrity verification.

#### Scenario: Raw PDF storage
- **WHEN** processing PDF documents
- **THEN** the system SHALL store original PDFs in `/data/raw/`
- **AND** SHALL compute and verify SHA-256 checksums
- **AND** SHALL maintain file organization by doc_id
- **AND** SHALL support PDF retrieval for reprocessing

#### Scenario: DocTags blob storage
- **WHEN** receiving DocTags from Docling VLM
- **THEN** the system SHALL store DocTags blobs in `/data/doctags/`
- **AND** SHALL compute doctags_sha from DocTags payload
- **AND** SHALL store DocTags in gzipped format for space efficiency
- **AND** SHALL verify DocTags integrity on retrieval

## MODIFIED Requirements

### Requirement: Storage Architecture
The storage architecture SHALL support chunk store + separate indexes model for reproducibility and rebuild capability.

#### Scenario: Storage model flexibility
- **WHEN** implementing storage for different retrieval strategies
- **THEN** the storage system SHALL support independent index updates
- **AND** SHALL allow rebuilding indexes without affecting chunk data
- **AND** SHALL maintain chunk store as the system of record
- **AND** SHALL provide manifest-based version tracking for all assets

#### Scenario: Storage performance optimization
- **WHEN** optimizing storage for retrieval performance
- **THEN** the storage system SHALL support efficient chunk retrieval
- **AND** SHALL provide optimized index storage for fast queries
- **AND** SHALL implement caching strategies for frequently accessed data
- **AND** SHALL monitor storage performance and capacity

## REMOVED Requirements

### Requirement: Monolithic Index Storage
**Reason**: Replaced by chunk store + separate indexes model
**Migration**: Existing indexes can be migrated to new storage model with rebuild capability

The system no longer requires storing all retrieval data in a single monolithic index structure.
