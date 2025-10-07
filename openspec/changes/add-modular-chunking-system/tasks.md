# Implementation Tasks: Modular Chunking System

## 1. Core Infrastructure (12 tasks)

- [x] 1.1 Define `BaseChunker` interface in `chunking/ports.py` with `chunk()`, `explain()` methods
- [x] 1.2 Create `Chunk` Pydantic model with chunk_id, doc_id, body, title_path, section, start_char, end_char, granularity, meta fields
- [x] 1.3 Define `Granularity` literal type ("window", "paragraph", "section", "document", "table")
- [x] 1.4 Create `ChunkerConfig` Pydantic model for configuration validation
- [x] 1.5 Implement chunker registry in `chunking/registry.py` with factory pattern
- [x] 1.6 Create `ChunkerFactory` with config-driven instantiation
- [x] 1.7 Build `MultiGranularityPipeline` orchestrator for parallel chunking
- [x] 1.8 Implement provenance utilities for offset tracking and title_path construction
- [x] 1.9 Create table handler utilities (atomic preservation, row/rowgroup/summary modes)
- [x] 1.10 Implement sentence splitter adapters (spaCy, NLTK Punkt, PySBD) with English focus
- [x] 1.11 Create coherence calculator utilities for semantic drift detection
- [x] 1.12 Build chunk assembly utilities for mapping IR blocks to Chunk objects

## 2. Stable Production Chunkers (10 tasks)

- [x] 2.1 Implement `SectionAwareChunker` with IMRaD/CT.gov/SPL/guideline section rules
- [x] 2.2 Add clinical section taxonomy data files (eligibility, endpoints, outcomes, AE, dose mappings)
- [x] 2.3 Implement `LayoutHeuristicChunker` with heading depth, font deltas, whitespace analysis
- [x] 2.4 Create `TableChunker` with row/rowgroup/summary modes and header preservation
- [x] 2.5 Implement `SlidingWindowChunker` with token windows and overlap
- [x] 2.6 Create `SemanticSplitterChunker` with embedding-drift boundaries (BGE-small-en default)
- [x] 2.7 Add coherence threshold and drift detection logic for semantic splitter
- [x] 2.8 Implement `ClinicalRoleChunker` with lightweight role classifier/rules
- [x] 2.9 Add role tagging for PICO, eligibility, endpoint, AE, dose sections
- [x] 2.10 Implement endpoint+effect pair preservation logic

## 3. Framework Integration Adapters (8 tasks)

- [x] 3.1 Create `LangChainSplitterChunker` wrapper for RecursiveCharacterTextSplitter
- [x] 3.2 Add LangChain adapters for TokenTextSplitter, MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter
- [x] 3.3 Add LangChain adapters for NLTKTextSplitter, SpacyTextSplitter
- [x] 3.4 Create `LlamaIndexNodeParserChunker` wrapper for SemanticSplitterNodeParser
- [x] 3.5 Add LlamaIndex adapters for HierarchicalNodeParser, SentenceSplitterNodeParser
- [x] 3.6 Create `HaystackPreprocessorChunker` wrapper with split_by modes (word/sentence/passage)
- [x] 3.7 Create `UnstructuredChunker` wrapper for chunk_by_title, chunk_by_element, chunk_by_page
- [x] 3.8 Implement offset mapping utilities for framework adapters to preserve provenance

## 4. Classical Lexical/Topic Segmentation Chunkers (8 tasks)

- [x] 4.1 Implement `TextTilingChunker` with Gensim integration
- [x] 4.2 Add TextTiling parameter tuning (block_size, step, similarity_window, smooth_width, cutoff)
- [x] 4.3 Implement `C99Chunker` with rank matrix and quantization
- [x] 4.4 Add cosine similarity matrix computation and smoothing for C99
- [x] 4.5 Implement `BayesSegChunker` with probabilistic topic switches
- [x] 4.6 Add dynamic programming or Bayesian inference for BayesSeg
- [x] 4.7 Implement `LDATopicChunker` with Gensim LDA and topic variation detection
- [x] 4.8 Add TopicTiling heuristic on top of LDA topics

## 5. Embedding-Driven Semantic Chunkers (6 tasks)

- [x] 5.1 Enhance `SemanticSplitterChunker` with configurable embedding models
- [x] 5.2 Add GPU availability check and fail-fast when `gpu_semantic_checks: true`
- [x] 5.3 Implement `SemanticClusterChunker` with HAC/HDBSCAN clustering
- [x] 5.4 Add contiguous span projection for cluster-based segmentation
- [x] 5.5 Implement `GraphPartitionChunker` with Louvain/Leiden community detection
- [x] 5.6 Add sentence similarity graph construction and contiguous cluster mapping

## 6. LLM-Assisted Chunking (5 tasks)

- [ ] 6.1 Implement `LLMChapteringChunker` with few-shot prompting
- [ ] 6.2 Create prompt templates for section boundary detection
- [ ] 6.3 Add boundary validation with semantic drift checks
- [ ] 6.4 Implement caching layer for LLM-generated boundaries (hash by doc_id + prompt_ver)
- [ ] 6.5 Add fallback to `SemanticSplitterChunker` for hallucinated boundaries

## 7. Advanced/Discourse Chunkers (5 tasks)

- [ ] 7.1 Implement `DiscourseSegmenterChunker` with EDU detection
- [ ] 7.2 Add connective-driven segmentation (however, therefore, in contrast)
- [ ] 7.3 Implement `GrobidSectionChunker` with TEI XML parsing
- [ ] 7.4 Create `LayoutAwareChunker` integration with LayoutParser/DocTR/Docling
- [ ] 7.5 Implement `GraphRAGChunker` with community summaries and hierarchical chunks

## 8. Configuration System (8 tasks)

- [x] 8.1 Create `config/chunking.yaml` with strategy selection and parameters
- [x] 8.2 Add per-family configuration blocks (lexical, semantic, llm, tables)
- [x] 8.3 Create profile configurations for PMC, DailyMed, CT.gov sources
- [x] 8.4 Add multi-granularity toggle and auxiliary chunker configuration
- [x] 8.5 Implement configuration validation with Pydantic models
- [x] 8.6 Add default parameters for each chunker with English-first models
- [x] 8.7 Create chunker registry population from configuration
- [ ] 8.8 Add environment-based configuration overrides

## 9. Ingestion Service Integration (6 tasks)

- [ ] 9.1 Extend `IngestionService` to detect document source/profile
- [ ] 9.2 Add chunking pipeline invocation after PDF parsing
- [ ] 9.3 Implement parallel chunker execution for multi-granularity mode
- [ ] 9.4 Add chunk_id generation with namespace (doc_id + chunker + index)
- [ ] 9.5 Integrate chunk storage with granularity tagging
- [ ] 9.6 Add telemetry for chunking latency and chunk size distributions

## 10. Retrieval Service Integration (5 tasks)

- [ ] 10.1 Extend retrieval service to support granularity-based filtering
- [ ] 10.2 Add multi-granularity fusion logic (RRF/weighted per granularity)
- [ ] 10.3 Implement neighbor merging for micro-chunks before reranking
- [ ] 10.4 Add granularity scoring weights in configuration
- [ ] 10.5 Update response models to include granularity metadata

## 11. Evaluation Harness (8 tasks)

- [ ] 11.1 Create `eval/chunking_eval.py` evaluation runner
- [ ] 11.2 Implement segmentation quality metrics (boundary F1 vs hand labels)
- [ ] 11.3 Add retrieval impact metrics (Recall@20, nDCG@10, nDCG@20)
- [ ] 11.4 Create gold standard boundary annotations for PMC/SPL/CT.gov samples (10-20 docs each)
- [ ] 11.5 Implement A/B testing framework for chunker comparison
- [ ] 11.6 Add latency distribution measurement per chunker
- [ ] 11.7 Create leaderboard visualization and regression guards
- [ ] 11.8 Integrate evaluation into CI pipeline for regression testing

## 12. Testing (12 tasks)

- [x] 12.1 Create unit tests for BaseChunker interface and Chunk model
- [x] 12.2 Add tests for each stable chunker with mock document inputs
- [ ] 12.3 Create tests for framework adapters with real library integration
- [ ] 12.4 Add tests for classical chunkers (TextTiling, C99, BayesSeg)
- [ ] 12.5 Create tests for semantic chunkers with mock embeddings
- [ ] 12.6 Add tests for LLM chunker with mock LLM responses
- [ ] 12.7 Create integration tests for multi-granularity pipeline
- [ ] 12.8 Add tests for provenance tracking and offset accuracy
- [ ] 12.9 Create tests for table atomic preservation
- [ ] 12.10 Add tests for clinical role detection and boundary rules
- [ ] 12.11 Create performance tests for chunking latency
- [x] 12.12 Add tests for configuration validation and registry

## 13. Documentation (5 tasks)

- [ ] 13.1 Write developer guide for adding new chunking adapters
- [ ] 13.2 Document each chunker's algorithm, parameters, and use cases
- [ ] 13.3 Create configuration examples for common scenarios
- [ ] 13.4 Write evaluation harness usage guide
- [ ] 13.5 Add API documentation for chunking module

## 14. Dependencies & Setup (4 tasks)

- [ ] 14.1 Add all required dependencies to `pyproject.toml`
- [ ] 14.2 Download/configure NLTK data (punkt tokenizer)
- [ ] 14.3 Download/configure spaCy model (en_core_web_sm)
- [ ] 14.4 Add dependency installation documentation

**Total: 102 tasks across 14 categories**
