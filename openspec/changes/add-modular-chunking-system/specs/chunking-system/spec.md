# Specification: Chunking System

## ADDED Requirements

### Requirement: Security & Multi-Tenant Integration

The chunking system SHALL integrate with the existing OAuth 2.0 authentication and multi-tenant isolation infrastructure.

#### Scenario: Tenant isolation in chunk metadata

- **WHEN** chunks are created from a document
- **THEN** each Chunk SHALL include tenant_id extracted from authenticated context
- **AND** tenant_id SHALL be validated against the document's tenant_id
- **AND** tenant_id SHALL be immutable after chunk creation

#### Scenario: Scope-based access control

- **WHEN** chunking service is invoked
- **THEN** the system SHALL verify `ingest:write` scope in JWT token
- **AND** return 403 Forbidden if scope is missing
- **AND** log unauthorized access attempts

#### Scenario: Audit logging for chunking operations

- **WHEN** chunking operations complete
- **THEN** the system SHALL log: user_id, tenant_id, doc_id, chunker_strategy, chunk_count, duration
- **AND** include correlation_id for distributed tracing
- **AND** scrub PII from log messages

---

### Requirement: Error Handling & Status Codes

The chunking system SHALL provide comprehensive error handling with standardized HTTP status codes and RFC 7807 Problem Details.

#### Scenario: Invalid document format error

- **WHEN** chunker receives malformed document
- **THEN** the system SHALL return 400 Bad Request
- **AND** include RFC 7807 Problem Details with type, title, detail, instance
- **AND** log error with correlation_id

#### Scenario: Chunker configuration error

- **WHEN** invalid chunker strategy or parameters specified
- **THEN** the system SHALL return 422 Unprocessable Entity
- **AND** include validation errors in Problem Details
- **AND** suggest valid alternatives

#### Scenario: Resource exhaustion error

- **WHEN** chunking operation exceeds memory or time limits
- **THEN** the system SHALL return 503 Service Unavailable
- **AND** include Retry-After header
- **AND** trigger circuit breaker after repeated failures

#### Scenario: GPU unavailable for semantic chunking

- **WHEN** semantic chunker requires GPU but none available
- **THEN** the system SHALL return 503 Service Unavailable
- **AND** include clear error message about GPU requirement
- **AND** fail-fast without CPU fallback (as per design)

---

### Requirement: Versioning & Backward Compatibility

The chunking system SHALL support versioning for chunker implementations and output schemas.

#### Scenario: Chunker version tracking

- **WHEN** chunks are created
- **THEN** each Chunk SHALL include chunker_version field (e.g., "semantic_splitter:v1.2")
- **AND** version SHALL be immutable after creation
- **AND** enable querying by chunker version

#### Scenario: Schema evolution

- **WHEN** Chunk model schema changes (new fields added)
- **THEN** new fields SHALL be optional with defaults
- **AND** existing chunks SHALL remain queryable
- **AND** migration scripts SHALL handle schema updates

#### Scenario: Deprecated chunker migration

- **WHEN** chunker implementation is deprecated
- **THEN** deprecation warning SHALL be logged
- **AND** migration path SHALL be documented
- **AND** deprecated chunker SHALL remain functional for 2 major versions

---

### Requirement: Performance SLOs & Circuit Breakers

The chunking system SHALL enforce performance SLOs and implement circuit breakers for failing operations.

#### Scenario: Chunking latency SLO

- **WHEN** chunking operation executes
- **THEN** P95 latency SHALL be <500ms for documents <10K tokens
- **AND** P95 latency SHALL be <2s for documents <100K tokens
- **AND** operations exceeding 5× SLO SHALL trigger alerts

#### Scenario: Circuit breaker on repeated failures

- **WHEN** chunker fails 5 consecutive times
- **THEN** circuit breaker SHALL open
- **AND** subsequent requests SHALL fail-fast with 503
- **AND** circuit SHALL attempt recovery after exponential backoff

#### Scenario: Resource monitoring

- **WHEN** chunking operations execute
- **THEN** the system SHALL monitor memory usage per operation
- **AND** reject operations exceeding 2GB memory limit
- **AND** emit metrics for memory usage, CPU time, GPU utilization

---

### Requirement: Comprehensive Testing Requirements

The chunking system SHALL include comprehensive test coverage with contract, performance, and integration tests.

#### Scenario: Contract tests for chunker interface

- **WHEN** new chunker is implemented
- **THEN** contract tests SHALL verify BaseChunker protocol compliance
- **AND** validate Chunk output schema
- **AND** test granularity parameter handling

#### Scenario: Performance regression tests

- **WHEN** chunker implementation changes
- **THEN** performance tests SHALL verify latency within SLO
- **AND** measure throughput (chunks/second)
- **AND** compare against baseline to detect regressions

#### Scenario: Integration tests with downstream services

- **WHEN** chunking completes
- **THEN** integration tests SHALL verify chunks are indexable by embedding service
- **AND** verify chunks are queryable by retrieval service
- **AND** test end-to-end pipeline (doc → chunk → embed → index → retrieve)

---

### Requirement: BaseChunker Interface

The system SHALL provide a universal `BaseChunker` protocol interface that all chunking adapters must implement, enabling pluggable chunking strategies without modifying core ingestion logic.

#### Scenario: Chunker implements required interface

- **GIVEN** a new chunking adapter is created
- **WHEN** the adapter is registered in the chunker registry
- **THEN** the adapter SHALL implement the `chunk()` method accepting Document, Block, and Table inputs
- **AND** the adapter SHALL return a list of `Chunk` objects with complete provenance metadata
- **AND** the adapter SHALL optionally implement the `explain()` method for debugging

#### Scenario: Chunker accepts granularity hint

- **GIVEN** a chunker is invoked through the pipeline
- **WHEN** the granularity parameter is provided (e.g., "paragraph", "section")
- **THEN** the chunker SHALL tag output chunks with the specified granularity level
- **AND** chunk_ids SHALL namespace by granularity to ensure uniqueness

### Requirement: Chunk Data Model

The system SHALL define a `Chunk` Pydantic model representing a semantically coherent text segment with complete provenance tracking.

#### Scenario: Chunk contains required fields

- **GIVEN** a chunker produces output
- **WHEN** a Chunk object is created
- **THEN** the Chunk SHALL include chunk_id with format `{doc_id}:{chunker}:{granularity}:{index}`
- **AND** the Chunk SHALL include doc_id linking to source document
- **AND** the Chunk SHALL include body text content
- **AND** the Chunk SHALL include start_char and end_char offset positions
- **AND** the Chunk SHALL include tenant_id for multi-tenant isolation

#### Scenario: Chunk preserves document structure

- **GIVEN** a document with hierarchical sections
- **WHEN** chunks are created from the document
- **THEN** each Chunk SHALL include title_path as breadcrumb trail from root to current section
- **AND** each Chunk SHALL include current section title
- **AND** each Chunk SHALL include page_no if available from PDF source

#### Scenario: Chunk includes granularity classification

- **GIVEN** chunks are produced with different segmentation strategies
- **WHEN** chunks are stored
- **THEN** each Chunk SHALL have granularity field with values from: "window", "paragraph", "section", "document", "table"
- **AND** granularity SHALL enable filtering at retrieval time

### Requirement: Chunker Registry and Factory

The system SHALL provide a chunker registry mapping strategy names to implementation classes, and a factory for config-driven instantiation.

#### Scenario: Registry supports stable chunkers

- **GIVEN** the chunker registry is initialized
- **WHEN** the registry is queried for available stable chunkers
- **THEN** the registry SHALL include: section_aware, sliding_window, semantic_splitter, table, clinical_role, layout_heuristic
- **AND** each chunker SHALL be instantiable via the factory

#### Scenario: Registry supports framework adapters

- **GIVEN** framework integration dependencies are installed
- **WHEN** framework chunkers are requested
- **THEN** the registry SHALL include LangChain adapters (langchain.recursive, langchain.token, langchain.markdown, etc.)
- **AND** the registry SHALL include LlamaIndex adapters (llama.semantic, llama.hierarchical, llama.sentence)
- **AND** the registry SHALL include Haystack and Unstructured adapters

#### Scenario: Registry supports experimental chunkers

- **GIVEN** experimental chunkers are enabled in configuration
- **WHEN** the registry is queried
- **THEN** the registry SHALL include: texttiling, c99, bayesseg, lda_topic, semantic_cluster, graph_partition, llm_chaptering, discourse_segmenter, grobid, layout_aware, graphrag
- **AND** experimental chunkers SHALL be clearly marked in metadata

#### Scenario: Factory creates chunker from config

- **GIVEN** a configuration specifying chunker strategy and parameters
- **WHEN** the factory creates a chunker instance
- **THEN** the factory SHALL retrieve the appropriate class from registry
- **AND** the factory SHALL instantiate with config parameters
- **AND** the factory SHALL validate configuration against chunker requirements
- **AND** the factory SHALL raise ConfigurationError for invalid settings

### Requirement: SectionAwareChunker

The system SHALL provide a production-ready `SectionAwareChunker` that respects biomedical document section boundaries (IMRaD, clinical trial structures, drug labels).

#### Scenario: IMRaD paper chunking

- **GIVEN** a research paper with Introduction, Methods, Results, Discussion sections
- **WHEN** SectionAwareChunker processes the document
- **THEN** chunks SHALL cut at top-level section boundaries
- **AND** chunks SHALL accumulate paragraphs until target_tokens ± 15%
- **AND** chunks SHALL include title_path showing section hierarchy
- **AND** chunks SHALL preserve complete paragraphs (no mid-paragraph splits)

#### Scenario: Clinical trial structure

- **GIVEN** a ClinicalTrials.gov document with eligibility, interventions, outcomes sections
- **WHEN** SectionAwareChunker with ctgov profile processes the document
- **THEN** chunks SHALL preserve eligibility criteria as atomic units
- **AND** chunks SHALL keep endpoint+effect pairs together
- **AND** chunks SHALL use target_tokens of 350 for structured fields

#### Scenario: Drug label chunking

- **GIVEN** an SPL-format drug label with indications, contraindications, dosing sections
- **WHEN** SectionAwareChunker with dailymed profile processes the document
- **THEN** chunks SHALL respect LOINC section boundaries
- **AND** chunks SHALL keep adverse event tables atomic
- **AND** chunks SHALL use target_tokens of 450 for dense content

### Requirement: SemanticSplitterChunker

The system SHALL provide a `SemanticSplitterChunker` that uses sentence embedding coherence to detect semantic boundaries, reducing mid-thought splits.

#### Scenario: Embedding-drift boundary detection

- **GIVEN** a document with varying topic coherence
- **WHEN** SemanticSplitterChunker processes the document with tau_coh=0.82
- **THEN** the chunker SHALL encode sentences using configured embedding model (default: BGE-small-en)
- **AND** the chunker SHALL compute sliding window cosine similarity
- **AND** the chunker SHALL place boundaries where cosine < tau_coh AND tokens >= min_tokens
- **AND** the chunker SHALL merge small tail chunks to avoid fragments

#### Scenario: GPU availability check with fail-fast

- **GIVEN** SemanticSplitterChunker configured with gpu_semantic_checks: true
- **WHEN** the chunker is initialized on a system without CUDA
- **THEN** the chunker SHALL raise RuntimeError indicating GPU unavailability
- **AND** the system SHALL fail-fast without attempting CPU fallback

#### Scenario: Hard boundary enforcement

- **GIVEN** a document with headings and tables
- **WHEN** SemanticSplitterChunker detects potential boundaries
- **THEN** the chunker SHALL enforce hard stops at heading boundaries
- **AND** the chunker SHALL NOT split within table blocks
- **AND** semantic boundaries SHALL align with structural boundaries when possible

### Requirement: SlidingWindowChunker

The system SHALL provide a `SlidingWindowChunker` as a robust fallback for documents with missing structure or noisy OCR.

#### Scenario: Fixed window creation

- **GIVEN** a long unstructured text document
- **WHEN** SlidingWindowChunker processes with target_tokens=512 and overlap_ratio=0.25
- **THEN** chunks SHALL be approximately 512 tokens with 128-token overlap
- **AND** chunks SHALL align to sentence boundaries when possible
- **AND** chunks SHALL include previous context for continuity

#### Scenario: Table atomic preservation

- **GIVEN** a document containing tables
- **WHEN** SlidingWindowChunker processes the document
- **THEN** table blocks SHALL remain atomic regardless of token count
- **AND** table chunks SHALL be tagged with granularity="table"
- **AND** table chunks SHALL set meta.is_table=true

### Requirement: TableChunker

The system SHALL provide a `TableChunker` with three modes (row, rowgroup, summary) for flexible table segmentation.

#### Scenario: Row mode chunking

- **GIVEN** a table with multiple rows
- **WHEN** TableChunker processes with mode="row"
- **THEN** each row SHALL produce one chunk
- **AND** column headers SHALL be prepended to each row chunk for context
- **AND** units SHALL be preserved from header if present

#### Scenario: Rowgroup mode chunking

- **GIVEN** an outcomes table with multiple intervention arms
- **WHEN** TableChunker processes with mode="rowgroup"
- **THEN** rows SHALL be grouped by arm/grade/category
- **AND** each group SHALL produce one coherent chunk
- **AND** related rows SHALL stay together (e.g., all Grade 3+ AEs)

#### Scenario: Summary mode chunking

- **GIVEN** a complex multi-row table
- **WHEN** TableChunker processes with mode="summary"
- **THEN** a structured table digest SHALL be generated and stored in facet_json
- **AND** a summary chunk SHALL index high-level table semantics
- **AND** full table details SHALL be retrievable via facet_json

### Requirement: ClinicalRoleChunker

The system SHALL provide a `ClinicalRoleChunker` that detects clinical roles (PICO, eligibility, endpoints, adverse events, dosing) and segments at role boundaries.

#### Scenario: Role detection and boundary placement

- **GIVEN** a clinical document with mixed content types
- **WHEN** ClinicalRoleChunker processes the document
- **THEN** sentences SHALL be tagged with roles: pico_population, pico_intervention, pico_comparison, pico_outcome, eligibility, endpoint, adverse_event, dose_regimen, general
- **AND** boundaries SHALL be placed when roles change
- **AND** endpoint+effect pairs SHALL stay together in single chunk

#### Scenario: Facet type metadata

- **GIVEN** a chunk with confidently detected clinical role
- **WHEN** the chunk is created
- **THEN** the chunk SHALL include facet_type in metadata
- **AND** facet_type SHALL enable faceted search downstream
- **AND** low-confidence role detections SHALL default to general

### Requirement: Multi-Granularity Pipeline

The system SHALL support concurrent execution of multiple chunkers to produce chunks at different granularity levels for hierarchical retrieval.

#### Scenario: Parallel chunker execution

- **GIVEN** configuration with primary chunker and two auxiliaries
- **WHEN** multi-granularity mode is enabled
- **THEN** the pipeline SHALL execute primary and auxiliary chunkers in parallel using asyncio
- **AND** execution time SHALL be approximately max(chunker_times) rather than sum
- **AND** each chunker SHALL produce chunks tagged with its target granularity

#### Scenario: Granularity-specific chunk namespacing

- **GIVEN** multiple chunkers producing output for same document
- **WHEN** chunks are assigned IDs
- **THEN** chunk_ids SHALL include granularity in namespace: `{doc_id}:{chunker}:{granularity}:{index}`
- **AND** chunks SHALL be uniquely identifiable across granularities
- **AND** retrieval SHALL filter by granularity at query time

#### Scenario: Multi-granularity toggle

- **GIVEN** multi-granularity configuration
- **WHEN** enable_multi_granularity is set to false
- **THEN** only the primary chunker SHALL execute
- **AND** auxiliary chunkers SHALL be skipped
- **AND** system behavior SHALL be compatible with single-granularity retrieval

### Requirement: Configuration-Driven Strategy Selection

The system SHALL support YAML-based configuration for chunker selection, parameters, and per-source profiles without code changes.

#### Scenario: Strategy selection via config

- **GIVEN** a configuration file specifying chunker strategy
- **WHEN** the ingestion service initializes chunking pipeline
- **THEN** the system SHALL load the specified chunker from registry
- **AND** the system SHALL apply strategy-specific parameters from config
- **AND** the system SHALL validate all parameters before instantiation

#### Scenario: Per-source profile application

- **GIVEN** configuration with profiles for pmc, dailymed, ctgov sources
- **WHEN** a document from specific source is ingested
- **THEN** the system SHALL detect appropriate profile based on doc.source
- **AND** the system SHALL apply profile-specific chunker and parameters
- **AND** profile SHALL override default configuration

#### Scenario: Experimental chunker opt-in

- **GIVEN** experimental chunkers in registry
- **WHEN** experimental.enabled is set to false (default)
- **THEN** experimental chunkers SHALL NOT be available for selection
- **AND** attempts to use experimental chunkers SHALL raise ConfigurationError
- **WHEN** experimental.enabled is set to true
- **THEN** experimental chunkers SHALL be available in registry

### Requirement: Framework Adapter Integration

The system SHALL provide adapters wrapping LangChain, LlamaIndex, Haystack, and Unstructured chunkers behind the BaseChunker interface.

#### Scenario: LangChain RecursiveCharacterTextSplitter adapter

- **GIVEN** LangChain library is installed
- **WHEN** langchain.recursive strategy is selected
- **THEN** the adapter SHALL instantiate RecursiveCharacterTextSplitter with config params
- **AND** the adapter SHALL convert IR blocks to text for splitting
- **AND** the adapter SHALL map split results back to Chunk objects with offsets

#### Scenario: LlamaIndex SemanticSplitterNodeParser adapter

- **GIVEN** LlamaIndex library is installed
- **WHEN** llama.semantic strategy is selected
- **THEN** the adapter SHALL instantiate SemanticSplitterNodeParser with embedding model and threshold
- **AND** the adapter SHALL convert IR to LlamaIndex Document format
- **AND** the adapter SHALL convert resulting nodes to Chunk objects

#### Scenario: Unstructured chunk_by_title adapter

- **GIVEN** Unstructured library is installed
- **WHEN** unstructured.by_title strategy is selected
- **THEN** the adapter SHALL use partitioned elements from IR
- **AND** the adapter SHALL chunk by heading/title elements
- **AND** the adapter SHALL preserve layout structure from partitioning

### Requirement: Classical Chunker Implementation

The system SHALL provide classical topic segmentation chunkers (TextTiling, C99, BayesSeg) as experimental research options.

#### Scenario: TextTiling lexical cohesion boundaries

- **GIVEN** a narrative text document
- **WHEN** TextTiling chunker processes with default parameters
- **THEN** the chunker SHALL compute lexical cohesion via term overlap in blocks
- **AND** the chunker SHALL place boundaries at cohesion dips
- **AND** the chunker SHALL apply smoothing to reduce noise
- **AND** resulting segments SHALL be merged to meet target_tokens

#### Scenario: C99 rank matrix segmentation

- **GIVEN** a multi-topic document
- **WHEN** C99 chunker processes the document
- **THEN** the chunker SHALL build cosine similarity matrix for text blocks
- **AND** the chunker SHALL apply rank-based quantization
- **AND** the chunker SHALL identify low-cohesion boundaries via matrix analysis

### Requirement: LLM-Assisted Chunking

The system SHALL provide an experimental `LLMChapteringChunker` that uses few-shot prompting to propose human-like section breaks validated with semantic drift checks.

#### Scenario: LLM section boundary generation

- **GIVEN** a long guideline document
- **WHEN** LLMChapteringChunker is invoked with max_section_tokens=1200
- **THEN** the chunker SHALL send few-shot prompt to vLLM endpoint
- **AND** the prompt SHALL request coherent sections under token limit
- **AND** LLM-proposed boundaries SHALL be validated with SemanticSplitter
- **AND** hallucinated boundaries SHALL be discarded

#### Scenario: Boundary caching for efficiency

- **GIVEN** LLM chunking is expensive
- **WHEN** a document is processed multiple times
- **THEN** boundaries SHALL be cached by hash(doc_id, prompt_version)
- **AND** cached boundaries SHALL be reused for identical inputs
- **AND** cache SHALL be invalidated if prompt_version changes

### Requirement: Provenance Tracking

The system SHALL maintain complete provenance for every chunk including character offsets, source paths, and creation metadata.

#### Scenario: Character offset preservation

- **GIVEN** chunks created from document blocks
- **WHEN** chunks are stored
- **THEN** each chunk SHALL have start_char and end_char mapping to original document
- **AND** offsets SHALL be accurate within ±1 character
- **AND** offsets SHALL enable exact text extraction from source

#### Scenario: Title path breadcrumb

- **GIVEN** a document with nested section hierarchy
- **WHEN** chunks are created from sections
- **THEN** each chunk SHALL have title_path as ordered list from root to current section
- **AND** title_path SHALL enable navigation and citation generation
- **AND** title_path SHALL be preserved even for multi-granularity chunks

#### Scenario: Creation timestamp and tenant isolation

- **GIVEN** chunks are created for a tenant
- **WHEN** chunks are stored
- **THEN** each chunk SHALL include created_at timestamp in UTC
- **AND** each chunk SHALL include tenant_id for multi-tenant isolation
- **AND** retrieval SHALL filter by tenant_id automatically

### Requirement: Evaluation Harness

The system SHALL provide an evaluation harness to measure segmentation quality (boundary F1) and retrieval impact (Recall@K, nDCG@K) for chunker comparison.

#### Scenario: Boundary F1 computation

- **GIVEN** gold standard boundary annotations for test documents
- **WHEN** a chunker is evaluated
- **THEN** the harness SHALL compute predicted boundaries from chunk offsets
- **AND** the harness SHALL calculate precision, recall, and F1 vs gold standard
- **AND** boundaries within ±N characters SHALL be considered matches (N configurable, default 50)

#### Scenario: Retrieval impact measurement

- **GIVEN** a fixed retrieval pipeline (embedding model, search algorithm)
- **WHEN** chunking strategy is varied
- **THEN** the harness SHALL measure Recall@10, Recall@20, nDCG@10 on QA benchmark
- **AND** metrics SHALL isolate chunking impact by holding other factors constant
- **AND** results SHALL be aggregated across multiple queries

#### Scenario: Latency profiling

- **GIVEN** multiple chunkers under evaluation
- **WHEN** latency is measured
- **THEN** the harness SHALL report P50, P95, P99 latency per chunker
- **AND** latency SHALL include end-to-end chunking time (preprocessing + segmentation + assembly)
- **AND** outliers SHALL be identified and logged for investigation

#### Scenario: Leaderboard generation

- **GIVEN** evaluation results for all chunkers
- **WHEN** leaderboard is generated
- **THEN** the leaderboard SHALL rank by configurable metric (default: nDCG@10)
- **AND** the leaderboard SHALL display boundary F1, retrieval metrics, and latency
- **AND** the leaderboard SHALL highlight production-recommended chunkers
- **AND** the leaderboard SHALL be exportable as JSON and HTML

### Requirement: Integration with Ingestion Service

The system SHALL integrate chunking pipeline into the ingestion service orchestration flow after PDF parsing and before embedding.

#### Scenario: Profile detection from document source

- **GIVEN** a document enters the ingestion pipeline
- **WHEN** the ingestion service prepares for chunking
- **THEN** the service SHALL detect profile from doc.source (pmc, dailymed, ctgov, default)
- **AND** the service SHALL load profile-specific configuration
- **AND** the service SHALL instantiate appropriate chunkers via factory

#### Scenario: Chunking pipeline invocation

- **GIVEN** a parsed document with IR blocks and tables
- **WHEN** chunking is enabled in configuration
- **THEN** the ingestion service SHALL invoke chunking pipeline
- **AND** the pipeline SHALL execute configured chunkers (primary + auxiliaries if multi-granularity)
- **AND** the pipeline SHALL return unified list of Chunk objects
- **AND** chunks SHALL be passed to downstream embedding and indexing stages

#### Scenario: Telemetry and monitoring

- **GIVEN** chunking is active
- **WHEN** documents are processed
- **THEN** the system SHALL emit telemetry metrics: chunks_created, chunking_latency_ms, chunk_size_tokens
- **AND** telemetry SHALL be tagged by chunker strategy and granularity
- **AND** telemetry SHALL enable performance monitoring and alerting

### Requirement: Retrieval Service Multi-Granularity Support

The system SHALL extend retrieval service to filter and fuse results across multiple granularity levels.

#### Scenario: Granularity-based filtering

- **GIVEN** chunks indexed at multiple granularities
- **WHEN** a retrieval query specifies granularity preference
- **THEN** the retrieval service SHALL filter results by requested granularity (paragraph, section, document)
- **AND** filtering SHALL occur before fusion to avoid mixing granularities unintentionally

#### Scenario: Multi-granularity fusion

- **GIVEN** retrieval results from paragraph and section granularities
- **WHEN** fusion is enabled
- **THEN** the retrieval service SHALL apply per-granularity scoring weights
- **AND** the service SHALL use RRF or weighted fusion across granularities
- **AND** final ranking SHALL consider both granularity-specific relevance and granularity weights

#### Scenario: Micro-chunk neighbor merging

- **GIVEN** retrieval results include window-granularity micro-chunks
- **WHEN** reranking is about to occur
- **THEN** the retrieval service SHALL merge adjacent micro-chunks from same document
- **AND** merging SHALL create larger context windows for reranker
- **AND** merged chunks SHALL preserve original span metadata

### Requirement: Testing and Validation

The system SHALL include comprehensive tests for chunker implementations, pipeline orchestration, and integration points.

#### Scenario: Unit tests for each chunker

- **GIVEN** a chunker implementation
- **WHEN** unit tests execute
- **THEN** tests SHALL verify chunk count, size distribution, and boundary placement
- **AND** tests SHALL validate offset accuracy against known documents
- **AND** tests SHALL check granularity tagging and metadata completeness

#### Scenario: Integration tests for multi-granularity pipeline

- **GIVEN** a multi-granularity configuration
- **WHEN** integration tests run
- **THEN** tests SHALL verify parallel execution of multiple chunkers
- **AND** tests SHALL validate unique chunk_id namespacing across granularities
- **AND** tests SHALL confirm correct profile selection and application

#### Scenario: Performance regression tests

- **GIVEN** baseline chunking performance metrics
- **WHEN** code changes are made
- **THEN** performance tests SHALL measure chunking latency and compare to baseline
- **AND** regressions beyond 20% SHALL fail CI pipeline
- **AND** performance improvements SHALL update baseline metrics

### Requirement: Documentation and Examples

The system SHALL provide comprehensive documentation for chunking system usage, configuration, and extension.

#### Scenario: Configuration examples

- **GIVEN** a developer needs to configure chunking
- **WHEN** documentation is consulted
- **THEN** examples SHALL be provided for common scenarios: PMC papers, drug labels, clinical trials
- **AND** examples SHALL show YAML configuration with parameters explained
- **AND** examples SHALL demonstrate multi-granularity setup

#### Scenario: Adapter development guide

- **GIVEN** a developer wants to add a new chunking algorithm
- **WHEN** developer guide is followed
- **THEN** guide SHALL explain BaseChunker interface requirements
- **AND** guide SHALL provide template code for new adapter
- **AND** guide SHALL explain registry registration and testing requirements

#### Scenario: Evaluation harness usage

- **GIVEN** a researcher wants to evaluate chunking strategies
- **WHEN** evaluation documentation is consulted
- **THEN** documentation SHALL explain how to create gold standard annotations
- **AND** documentation SHALL provide commands to run evaluation
- **AND** documentation SHALL explain metric interpretation and leaderboard usage

---

## Implementation Notes

### Monitoring & Alerting Thresholds

**Prometheus Metrics** (all labeled by chunker_strategy, granularity, tenant_id):

- `chunking_operations_total` (counter) - Total chunking operations
- `chunking_operations_duration_seconds` (histogram) - Operation latency with buckets: [0.1, 0.5, 1, 2, 5, 10]
- `chunking_errors_total` (counter) - Errors by error_type
- `chunking_chunks_produced_total` (counter) - Total chunks created
- `chunking_memory_bytes` (gauge) - Memory usage per operation
- `chunking_gpu_utilization_percent` (gauge) - GPU utilization for semantic chunking
- `chunking_circuit_breaker_state` (gauge) - Circuit breaker states (0=closed, 1=open, 2=half-open)

**Alert Rules**:

- `ChunkingHighLatency`: P95 > 1s for 5 minutes → Page on-call
- `ChunkingHighErrorRate`: Error rate > 5% for 5 minutes → Page on-call
- `ChunkingCircuitBreakerOpen`: Circuit breaker open > 1 minute → Notify team
- `ChunkingMemoryHigh`: Memory usage > 1.5GB → Warning
- `ChunkingGPUUnavailable`: GPU required but unavailable > 2 minutes → Page on-call

### Data Validation Rules

**Chunk Validation**:

- `chunk_id` format: `^[a-z0-9_-]+:[a-z0-9_-]+:[a-z_]+:\d+$`
- `doc_id` format: `^[a-z]+:[A-Za-z0-9_-]+#[a-z0-9]+:[a-f0-9]{12}$`
- `tenant_id` format: `^[a-z0-9-]{8,64}$`
- `body` length: 10 ≤ len ≤ 50,000 characters
- `start_char` < `end_char` and both ≥ 0
- `granularity` ∈ {"window", "paragraph", "section", "document", "table"}

**Configuration Validation**:

- `target_tokens`: 100 ≤ value ≤ 4096
- `overlap_ratio`: 0.0 ≤ value ≤ 0.5
- `tau_coh` (semantic coherence): 0.5 ≤ value ≤ 1.0
- `delta_drift` (embedding drift): 0.1 ≤ value ≤ 0.8

### API Versioning

**Chunking API Endpoints**:

- `/v1/chunk` - Current stable API
- `/v2/chunk` - Future breaking changes (reserved)

**Version Headers**:

- Request: `Accept: application/vnd.medkg.chunk.v1+json`
- Response: `Content-Type: application/vnd.medkg.chunk.v1+json`
- Response: `X-API-Version: 1.0`

**Breaking Change Policy**:

- Breaking changes require new major version
- Old version supported for 12 months after new version release
- Deprecation warnings logged 6 months before sunset
- Migration guide published with new version

### Security Considerations

**Input Validation**:

- Reject documents > 100MB uncompressed
- Sanitize all text content (remove control characters, validate UTF-8)
- Validate all IDs against format regex before processing

**Rate Limiting**:

- Per-tenant: 100 chunking operations/minute
- Per-user: 50 chunking operations/minute
- Burst: 20 operations
- Return 429 with Retry-After header when exceeded

**Secrets Management**:

- GPU service endpoints: Environment variables or Vault
- Model paths: Configuration files (not hardcoded)
- API keys: Rotate every 90 days

### Dependencies

- **Upstream**: `add-foundation-infrastructure` (Document, Block models), `add-security-auth` (OAuth, multi-tenant)
- **Downstream**: `add-universal-embedding-system` (consumes chunks), `add-retrieval-pipeline-orchestration` (integrates chunking stage)
- **Python packages**: `spacy`, `nltk`, `torch` (GPU), `transformers`, `langchain`, `llama-index`, `haystack-ai`, `unstructured`
- **Models**: `en_core_web_sm` (spaCy), `BAAI/bge-small-en-v1.5` (semantic splitting)
