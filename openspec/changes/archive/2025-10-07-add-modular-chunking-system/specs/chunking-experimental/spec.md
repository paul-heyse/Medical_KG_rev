# Specification: Experimental Chunking Adapters

## ADDED Requirements

### Requirement: Experimental Chunker Registry

The system SHALL maintain a separate registry for experimental/research-grade chunking adapters that are opt-in and clearly marked as non-production.

#### Scenario: Experimental flag guards access

- **GIVEN** experimental chunkers exist in the registry
- **WHEN** experimental.enabled configuration is false (default)
- **THEN** experimental chunkers SHALL NOT be instantiable via factory
- **AND** attempts to use experimental chunkers SHALL raise ConfigurationError with clear message
- **WHEN** experimental.enabled is true
- **THEN** experimental chunkers SHALL be available alongside stable chunkers

#### Scenario: Experimental metadata annotation

- **GIVEN** an experimental chunker in registry
- **WHEN** registry metadata is queried
- **THEN** chunker SHALL be tagged with experimental: true
- **AND** registry SHALL include warnings about stability and performance
- **AND** documentation SHALL clearly state experimental status

### Requirement: TextTiling Chunker (Hearst, 1997)

The system SHALL provide a `TextTilingChunker` implementing lexical cohesion-based segmentation for topic boundary detection.

#### Scenario: Lexical cohesion computation

- **GIVEN** a multi-topic narrative document
- **WHEN** TextTilingChunker processes with default parameters
- **THEN** the chunker SHALL divide text into sentence blocks of configurable size
- **AND** the chunker SHALL compute term overlap between adjacent blocks
- **AND** cohesion scores SHALL be smoothed using configurable window
- **AND** boundaries SHALL be placed at cohesion valleys below threshold

#### Scenario: Parameter configuration

- **GIVEN** TextTiling configuration
- **WHEN** chunker is instantiated
- **THEN** block_size parameter SHALL control sentence grouping (default: 10 sentences)
- **AND** step parameter SHALL control sliding window stride (default: 5)
- **AND** smooth_width SHALL control smoothing kernel size (default: 3)
- **AND** cutoff_z SHALL control Z-score threshold for boundary placement (default: 1.2)

#### Scenario: Post-processing to target tokens

- **GIVEN** TextTiling produces variable-length segments
- **WHEN** segments are finalized
- **THEN** small segments SHALL be merged with neighbors to meet min_tokens
- **AND** large segments SHALL be split using sliding window
- **AND** final chunks SHALL approximate target_tokens ± 20%

### Requirement: C99 Chunker (Choi, 2000)

The system SHALL provide a `C99Chunker` implementing rank-based similarity matrix segmentation for domain-independent text segmentation.

#### Scenario: Similarity matrix construction

- **GIVEN** a document with topic shifts
- **WHEN** C99Chunker processes the document
- **THEN** the chunker SHALL build sentence-level cosine similarity matrix
- **AND** the matrix SHALL use configurable window for computing similarities
- **AND** similarities SHALL be rank-transformed for robustness

#### Scenario: Matrix smoothing and boundary detection

- **GIVEN** a rank similarity matrix
- **WHEN** boundaries are detected
- **THEN** the matrix SHALL be smoothed using configurable kernel
- **AND** boundaries SHALL be identified at low-rank valleys
- **AND** dynamic programming MAY be used to optimize boundary placement
- **AND** minimum segment size constraints SHALL be enforced

#### Scenario: Configuration parameters

- **GIVEN** C99 configuration
- **WHEN** chunker is instantiated
- **THEN** rank_width parameter SHALL control ranking window size (default: 11)
- **AND** smoothing_width SHALL control smoothing kernel (default: 1)
- **AND** min_segment_length SHALL prevent tiny segments (default: 100 tokens)

### Requirement: BayesSeg Chunker (Eisenstein & Barzilay, 2008)

The system SHALL provide a `BayesSegChunker` implementing Bayesian probabilistic topic segmentation for robust boundary detection.

#### Scenario: Probabilistic topic model

- **GIVEN** a document with latent topic structure
- **WHEN** BayesSegChunker processes the document
- **THEN** the chunker SHALL model segments as draws from topic distributions
- **AND** the chunker SHALL use prior strength to control segment granularity
- **AND** boundaries SHALL be inferred via Bayesian model comparison

#### Scenario: Dynamic programming inference

- **GIVEN** probabilistic segment hypotheses
- **WHEN** optimal segmentation is computed
- **THEN** the chunker SHALL use dynamic programming to find maximum likelihood boundaries
- **AND** the chunker SHALL balance segment coherence vs boundary cost
- **AND** computation SHALL be efficient for documents up to 10k sentences

#### Scenario: Configuration and priors

- **GIVEN** BayesSeg configuration
- **WHEN** chunker is instantiated
- **THEN** prior_strength SHALL control segment size preference (default: 0.5)
- **AND** min_segment_tokens SHALL enforce minimum segment size (default: 200)
- **AND** lexical_model MAY use unigrams or bigrams (default: unigrams)

### Requirement: LDA Topic Chunker (Riedl & Biemann, 2012)

The system SHALL provide an `LDATopicChunker` implementing TopicTiling algorithm based on LDA topic variation for topical segmentation.

#### Scenario: LDA topic modeling

- **GIVEN** a document with distinct topical sections
- **WHEN** LDATopicChunker processes the document
- **THEN** the chunker SHALL train LDA model on document with K topics (configurable, default: 50)
- **AND** the chunker SHALL assign topic distributions to sentence blocks
- **AND** topic distributions SHALL use configurable block size (default: 10 sentences)

#### Scenario: Topic shift detection

- **GIVEN** topic distributions for sentence blocks
- **WHEN** boundaries are identified
- **THEN** the chunker SHALL compute Jensen-Shannon divergence between adjacent blocks
- **AND** boundaries SHALL be placed where divergence exceeds threshold
- **AND** local minima in divergence SHALL be prioritized as boundary points

#### Scenario: Parameter tuning

- **GIVEN** LDA configuration
- **WHEN** chunker is instantiated
- **THEN** num_topics parameter SHALL control LDA K (default: 50 for scientific text)
- **AND** block_size SHALL control topic distribution granularity (default: 10 sentences)
- **AND** divergence_threshold SHALL control boundary sensitivity (default: 0.3)

### Requirement: Semantic Cluster Chunker

The system SHALL provide a `SemanticClusterChunker` using hierarchical clustering or HDBSCAN on sentence embeddings to identify topical segments.

#### Scenario: Sentence embedding clustering

- **GIVEN** a document with distinct topic clusters
- **WHEN** SemanticClusterChunker processes the document
- **THEN** the chunker SHALL encode all sentences using configured embedding model
- **AND** the chunker SHALL apply HAC or HDBSCAN clustering to embeddings
- **AND** clustering SHALL use configurable linkage method (default: ward for HAC)

#### Scenario: Contiguous span projection

- **GIVEN** cluster assignments for sentences
- **WHEN** clusters are mapped to chunks
- **THEN** the chunker SHALL project clusters to contiguous text spans
- **AND** non-contiguous cluster members SHALL be split into separate chunks
- **AND** ordering SHALL be preserved from original document
- **AND** token count constraints SHALL be enforced per cluster-chunk

#### Scenario: Configuration options

- **GIVEN** clustering configuration
- **WHEN** chunker is instantiated
- **THEN** method SHALL be "hac" or "hdbscan" (default: hac)
- **AND** min_cluster_size SHALL prevent tiny clusters (default: 3 sentences)
- **AND** max_tokens_per_segment SHALL enforce upper bound (default: 800)
- **AND** linkage SHALL be "ward", "average", or "complete" for HAC

### Requirement: Graph Partition Chunker

The system SHALL provide a `GraphPartitionChunker` using community detection on sentence similarity graphs for section discovery.

#### Scenario: Sentence similarity graph construction

- **GIVEN** a heterogeneous document
- **WHEN** GraphPartitionChunker processes the document
- **THEN** the chunker SHALL build graph with sentences as nodes
- **AND** edges SHALL connect sentences with cosine similarity > threshold (default: 0.7)
- **AND** edge weights SHALL equal cosine similarity scores

#### Scenario: Community detection

- **GIVEN** a sentence similarity graph
- **WHEN** communities are detected
- **THEN** the chunker SHALL apply Louvain or Leiden algorithm (configurable, default: Louvain)
- **AND** community detection SHALL maximize modularity
- **AND** resolution parameter SHALL control granularity (default: 1.0)

#### Scenario: Contiguous community mapping

- **GIVEN** community assignments
- **WHEN** chunks are generated
- **THEN** the chunker SHALL merge contiguous sentences from same community
- **AND** non-contiguous community members SHALL form separate chunks
- **AND** chunks SHALL be ordered by first sentence position
- **AND** token limits SHALL be respected

#### Scenario: NetworkX integration

- **GIVEN** graph operations required
- **WHEN** chunker is implemented
- **THEN** NetworkX library SHALL be used for graph construction and algorithms
- **AND** sparse graph representation SHALL be used for efficiency
- **AND** computation SHALL be feasible for documents with 1000+ sentences

### Requirement: LLM Chaptering Chunker

The system SHALL provide an `LLMChapteringChunker` using few-shot prompting to propose human-like section breaks validated with semantic checks.

#### Scenario: Few-shot prompt construction

- **GIVEN** a long guideline document
- **WHEN** LLM chaptering is invoked
- **THEN** the chunker SHALL construct prompt with 2-3 examples of good section breaks
- **AND** prompt SHALL include document outline (title_path from IR)
- **AND** prompt SHALL specify max_section_tokens constraint
- **AND** prompt SHALL request markdown-formatted section boundaries

#### Scenario: LLM boundary generation

- **GIVEN** a constructed prompt
- **WHEN** LLM is queried via vLLM endpoint
- **THEN** the chunker SHALL use configured model (default: Qwen2.5-7B-Instruct)
- **AND** generation SHALL use low temperature (0.1) for consistency
- **AND** output SHALL be parsed to extract section boundaries (line numbers or offsets)

#### Scenario: Semantic validation of boundaries

- **GIVEN** LLM-proposed boundaries
- **WHEN** validation occurs
- **THEN** the chunker SHALL instantiate SemanticSplitter for validation
- **AND** proposed boundaries SHALL be checked for semantic coherence drops
- **AND** boundaries without coherence support SHALL be discarded as hallucinations
- **AND** validated boundaries SHALL align with nearest heading if available

#### Scenario: Boundary caching

- **GIVEN** LLM chaptering is expensive
- **WHEN** document is processed
- **THEN** boundaries SHALL be cached by hash(doc_id, prompt_version, model_name)
- **AND** cache hits SHALL skip LLM query and reuse stored boundaries
- **AND** cache SHALL be invalidated when prompt_version or model changes
- **AND** cache SHALL use persistent storage (Redis or file-based)

#### Scenario: Fallback on failure

- **GIVEN** LLM chunking fails or produces invalid output
- **WHEN** error is detected
- **THEN** the chunker SHALL log failure with doc_id and error details
- **AND** the chunker SHALL fall back to SemanticSplitter for that document
- **AND** fallback SHALL maintain pipeline continuity without raising exceptions

### Requirement: Discourse Segmenter Chunker

The system SHALL provide a `DiscourseSegmenterChunker` using rhetorical structure theory (RST) or discourse connectives for precision segmentation.

#### Scenario: EDU detection

- **GIVEN** a document with complex discourse structure
- **WHEN** DiscourseSegmenterChunker processes the document
- **THEN** the chunker SHALL identify Elementary Discourse Units (EDUs)
- **AND** EDU boundaries SHALL be detected via parser or heuristics
- **AND** EDUs SHALL be treated as atomic units

#### Scenario: Connective-driven segmentation

- **GIVEN** EDUs with discourse connectives
- **WHEN** segment boundaries are placed
- **THEN** strong connectives SHALL trigger boundaries (however, therefore, in contrast, nevertheless)
- **AND** weak connectives SHALL be ignored (and, but, or)
- **AND** connective strength SHALL be configurable via lexicon

#### Scenario: EDU grouping to target tokens

- **GIVEN** variable-length EDUs
- **WHEN** final chunks are assembled
- **THEN** EDUs SHALL be grouped until target_tokens is reached
- **AND** groups SHALL respect connective boundaries
- **AND** discourse coherence SHALL be maximized within groups

### Requirement: GROBID Section Chunker

The system SHALL provide a `GrobidSectionChunker` integrating with GROBID TEI XML output for academic paper segmentation.

#### Scenario: TEI XML parsing

- **GIVEN** a GROBID-processed academic paper in TEI format
- **WHEN** GrobidSectionChunker processes the TEI
- **THEN** the chunker SHALL extract structured sections from `<div>` elements
- **AND** section headers SHALL be extracted from `<head>` elements
- **AND** paragraphs SHALL be extracted from `<p>` elements

#### Scenario: IMRaD structure exploitation

- **GIVEN** TEI with Introduction, Methods, Results, Discussion sections
- **WHEN** chunks are created
- **THEN** sections SHALL be chunked according to IMRaD profile
- **AND** title_path SHALL include IMRaD section names
- **AND** Results tables SHALL be preserved atomically
- **AND** References section SHALL be optionally excluded (configurable)

#### Scenario: GROBID integration

- **GIVEN** GROBID server is available
- **WHEN** PDF is ingested
- **THEN** system MAY call GROBID as alternative to MinerU for academic papers
- **AND** TEI output SHALL be stored alongside IR
- **AND** GrobidSectionChunker SHALL be automatically selected for TEI documents

### Requirement: Layout-Aware Chunker

The system SHALL provide a `LayoutAwareChunker` integrating with LayoutParser, DocTR, or Docling for layout-driven segmentation.

#### Scenario: Layout element detection

- **GIVEN** a complex document with varied layout
- **WHEN** LayoutAwareChunker processes the document
- **THEN** the chunker SHALL use layout detection to identify regions (text, tables, figures, headers)
- **AND** layout detection SHALL use LayoutParser, DocTR, or Docling (configurable)
- **AND** detected regions SHALL be preserved atomically

#### Scenario: Semantic chunking within regions

- **GIVEN** detected text regions
- **WHEN** chunking within regions
- **THEN** SemanticSplitter SHALL be applied within each text region independently
- **AND** region boundaries SHALL act as hard stops
- **AND** tables and figures SHALL remain atomic regardless of size

#### Scenario: Multi-column handling

- **GIVEN** a document with multi-column layout
- **WHEN** reading order is determined
- **THEN** layout detection SHALL establish correct reading order
- **AND** chunks SHALL respect column boundaries
- **AND** column order SHALL be preserved in chunk sequence

### Requirement: GraphRAG Chunker

The system SHALL provide a `GraphRAGChunker` implementing Microsoft GraphRAG's community summary approach for hierarchical chunks.

#### Scenario: Community detection and summarization

- **GIVEN** a large document corpus
- **WHEN** GraphRAGChunker processes documents
- **THEN** the chunker SHALL build knowledge graph from entities and relationships
- **AND** the chunker SHALL detect communities using Leiden algorithm
- **AND** community summaries SHALL be generated as coarse-grain chunks

#### Scenario: Hierarchical chunk structure

- **GIVEN** community structure
- **WHEN** hierarchical chunks are created
- **THEN** the chunker SHALL produce document-level chunks (leaves)
- **AND** the chunker SHALL produce community summary chunks (intermediate)
- **AND** the chunker SHALL produce global summary chunks (root)
- **AND** chunks SHALL be linked via parent-child relationships

#### Scenario: Multi-granularity integration

- **GIVEN** GraphRAG hierarchical chunks
- **WHEN** integrated with multi-granularity system
- **THEN** document-level SHALL map to granularity="paragraph"
- **AND** community summaries SHALL map to granularity="section"
- **AND** global summaries SHALL map to granularity="document"
- **AND** retrieval SHALL support navigation up/down hierarchy

### Requirement: Experimental Chunker Evaluation

The system SHALL provide enhanced evaluation for experimental chunkers to assess their research value vs production chunkers.

#### Scenario: Side-by-side comparison

- **GIVEN** experimental and stable chunkers
- **WHEN** evaluation harness runs
- **THEN** all chunkers SHALL be evaluated on same test set
- **AND** results SHALL be grouped by stable vs experimental
- **AND** experimental SHALL be compared against best stable baseline

#### Scenario: Convergence and robustness testing

- **GIVEN** experimental chunkers with hyperparameters
- **WHEN** robustness is tested
- **THEN** evaluation SHALL vary parameters ±20% to measure sensitivity
- **AND** evaluation SHALL test on diverse document types
- **AND** brittle chunkers SHALL be flagged with warnings

#### Scenario: Computational cost analysis

- **GIVEN** experimental chunkers may be expensive
- **WHEN** cost is measured
- **THEN** evaluation SHALL report wall-clock time, CPU usage, memory usage
- **AND** GPU-dependent chunkers SHALL be separately profiled
- **AND** cost/benefit ratio SHALL be computed vs quality improvements

### Requirement: Experimental Chunker Testing

The system SHALL include tests for experimental chunkers with focus on boundary accuracy and fallback behavior.

#### Scenario: Boundary accuracy tests

- **GIVEN** an experimental chunker
- **WHEN** tested on synthetic documents with known boundaries
- **THEN** tests SHALL measure boundary detection precision and recall
- **AND** tests SHALL validate that boundaries align with topic shifts
- **AND** edge cases (very short/long segments) SHALL be tested

#### Scenario: Fallback and error handling

- **GIVEN** experimental chunkers may fail
- **WHEN** failures occur (e.g., convergence issues, timeouts)
- **THEN** tests SHALL verify graceful fallback to stable chunker
- **AND** tests SHALL verify error logging includes diagnostic information
- **AND** partial results SHALL be discarded without pipeline corruption

#### Scenario: Integration with stable system

- **GIVEN** experimental chunkers in production system
- **WHEN** integration tests run
- **THEN** tests SHALL verify experimental flag correctly gates access
- **AND** tests SHALL verify stable system unaffected when experimental disabled
- **AND** tests SHALL verify configuration validation prevents misconfiguration

### Requirement: Experimental Chunker Documentation

The system SHALL provide documentation clearly marking experimental status and usage guidelines.

#### Scenario: Experimental warning

- **GIVEN** a user wants to use experimental chunker
- **WHEN** documentation is consulted
- **THEN** documentation SHALL display prominent experimental warning
- **AND** warning SHALL note stability and performance are not guaranteed
- **AND** warning SHALL recommend stable alternatives for production

#### Scenario: Research usage guidance

- **GIVEN** a researcher wants to evaluate new chunking algorithm
- **WHEN** developer guide is followed
- **THEN** guide SHALL explain how to add experimental chunker to registry
- **AND** guide SHALL provide template for experimental chunker with required methods
- **AND** guide SHALL explain evaluation harness for research comparison

#### Scenario: Migration path to stable

- **GIVEN** an experimental chunker proves valuable
- **WHEN** promotion to stable is desired
- **THEN** documentation SHALL outline requirements: test coverage, performance benchmarks, documentation
- **AND** documentation SHALL explain registry migration process
- **AND** documentation SHALL note backward compatibility considerations
