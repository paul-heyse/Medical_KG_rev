# Add Modular Chunking System

## Why

The current system lacks a flexible, extensible chunking infrastructure capable of handling diverse biomedical document types (clinical trials, drug labels, research papers, guidelines) with domain-aware segmentation strategies. We need a modular chunking system that:

1. Supports multiple chunking algorithms through a unified interface (ports & adapters pattern)
2. Handles domain-specific requirements (IMRaD sections, clinical trial structures, SPL labels, LOINC sections)
3. Enables multi-granularity chunking (paragraph, section, window, table, document levels) for superior retrieval
4. Provides both production-ready and experimental/research chunking methods
5. Integrates classical algorithms (TextTiling, C99, BayesSeg), modern semantic splitters (embedding-drift), and framework adapters (LangChain, LlamaIndex, Unstructured, Haystack)
6. Maintains atomic table handling, preserves provenance (span offsets), and enforces clinical section rules

This is essential for achieving high English-first retrieval accuracy across heterogeneous biomedical corpora while maintaining the flexibility to evaluate and swap chunking strategies via configuration.

## What Changes

### Core Infrastructure

- **BaseChunker interface** with universal `chunk()` method accepting Document/Block/Table inputs
- **Chunker registry** with 15+ adapter implementations (stable + experimental)
- **Multi-granularity pipeline** that runs multiple chunkers concurrently and tags outputs
- **Configuration-driven strategy selection** via YAML with per-source profiles (PMC, DailyMed, CT.gov)
- **Provenance tracking** with `start_char`, `end_char`, `title_path`, and `granularity` labels
- **Table-aware processing** with row/rowgroup/summary modes

### Stable Chunking Adapters (Production-Ready)

- `SectionAwareChunker` - IMRaD/CT.gov/SPL/guideline section boundaries
- `LayoutHeuristicChunker` - Heading hierarchy, font deltas, whitespace analysis
- `TableChunker` - Row/rowgroup/summary modes with header preservation
- `SlidingWindowChunker` - Fixed windows with overlap (robust fallback)
- `SemanticSplitterChunker` - Embedding-drift boundaries with coherence thresholds
- `ClinicalRoleChunker` - Role-based segmentation (PICO, eligibility, endpoints, AE, dose)

### Framework Integration Adapters

- `LangChainSplitterChunker` - Recursive/token/markdown/HTML/spacy/NLTK splitters
- `LlamaIndexNodeParserChunker` - Semantic/hierarchical/sentence parsers
- `HaystackPreprocessorChunker` - Word/sentence/passage splitting
- `UnstructuredChunker` - Layout-aware chunk_by_title/element/page

### Experimental/Research Adapters

- `TextTilingChunker` - Lexical cohesion segmentation (Hearst, 1997)
- `C99Chunker` - Rank-based similarity matrix (Choi, 2000)
- `BayesSegChunker` - Bayesian topic boundaries (Eisenstein & Barzilay, 2008)
- `LDATopicChunker` - LDA-based topic variation segmentation
- `SemanticClusterChunker` - HAC/HDBSCAN clustering of embeddings
- `GraphPartitionChunker` - Community detection (Louvain/Leiden)
- `LLMChapteringChunker` - Few-shot prompted section breaks with semantic validation
- `DiscourseSegmenterChunker` - RST/PDTB-style rhetorical units
- `GrobidSectionChunker` - TEI-based academic PDF sections
- `LayoutAwareChunker` - LayoutParser/DocTR/Docling integration
- `GraphRAGChunker` - Community summaries with hierarchical chunks

### Configuration & Orchestration

- YAML-driven chunker selection with target tokens, overlap ratios, and strategy-specific parameters
- Profile-based configuration for different document sources
- Multi-granularity toggle (`enable_multi_granularity: true|false`)
- GPU semantic check enforcement (`gpu_semantic_checks: true` for fail-fast)
- Evaluation harness for boundary F1 and retrieval nDCG@K metrics

### Integration Points

- `IngestionService` orchestration for profile detection and parallel chunking
- Unified `Chunk` Pydantic model output with metadata (chunk_id, doc_id, body, title_path, section, offsets, granularity, meta)
- Storage integration with granularity-based indexing for multi-level retrieval
- Retrieval service fusion of multi-granularity results

## Impact

### Affected Specs

- **New**: `chunking-system` (core interfaces, registry, orchestration)
- **New**: `chunking-experimental` (research-grade adapters)
- **Modified**: `ingestion-orchestration` (adds chunking pipeline integration)
- **Modified**: `knowledge-graph` (Chunk model provenance tracking)
- **Modified**: `retrieval-system` (multi-granularity fusion)

### Affected Code

- **New**: `src/Medical_KG_rev/chunking/` (entire module)
  - `ports.py` - BaseChunker interface
  - `registry.py` - Chunker registry and factory
  - `models.py` - Chunk, ChunkerConfig models
  - `stable/` - Production chunkers
  - `frameworks/` - LangChain/LlamaIndex/Unstructured/Haystack adapters
  - `experimental/` - Research chunkers
  - `utils/` - Shared utilities (sentence splitters, coherence calculators, table handlers)
- **Modified**: `src/Medical_KG_rev/orchestration/ingestion_service.py`
- **Modified**: `src/Medical_KG_rev/services/retrieval_service.py`
- **New**: `config/chunking.yaml` - Chunking configuration
- **New**: `config/profiles/` - Per-source chunking profiles
- **New**: `tests/chunking/` - Comprehensive test suite
- **New**: `eval/chunking_eval.py` - Evaluation harness

### Dependencies Added

- `nltk>=3.8` (TextTiling, sentence tokenization)
- `gensim>=4.3.0` (TextTiling implementation)
- `scikit-learn>=1.3.0` (clustering, cohesion metrics)
- `networkx>=3.1` (graph partitioning)
- `langchain-text-splitters>=0.0.1` (LangChain integration)
- `llama-index>=0.10.0` (LlamaIndex node parsers)
- `haystack-ai>=2.0.0` (Haystack preprocessor)
- `unstructured[local-inference]>=0.12.0` (Unstructured partitioning)
- `spacy>=3.7.0` (sentence segmentation, NLP features)
- `pysbd>=0.3.4` (sentence boundary detection for biomedical text)

### Breaking Changes

None - this is a new capability that integrates with existing ingestion pipeline without modifying current behavior unless explicitly configured.

### Migration Path

1. Existing ingestion continues unchanged (no chunking by default)
2. Enable chunking via `chunker.enabled: true` in configuration
3. Select chunking strategy and profile per document source
4. Gradually enable multi-granularity for improved retrieval
5. Evaluate different strategies using provided harness
6. Set production defaults based on evaluation results

### Benefits

- **Modularity**: Swap chunking strategies via configuration without code changes
- **Flexibility**: Support 20+ chunking methods from classical to cutting-edge
- **Domain-awareness**: Respect biomedical document structure and clinical boundaries
- **Multi-granularity**: Enable paragraph, section, and document-level retrieval simultaneously
- **Research-friendly**: Pre-wired experimental adapters for academic exploration
- **English-first**: Optimized for English biomedical text with proper sentence boundaries
- **Performance**: Fail-fast GPU checks, efficient batch processing, caching for expensive operations
- **Eval-driven**: Built-in evaluation harness for data-driven strategy selection
