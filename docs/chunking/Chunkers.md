# Chunker Catalogue

This document summarises the algorithms, parameters, and intended use cases for the bundled
chunkers.

## Stable Chunkers

| Chunker | Algorithm | Key Parameters | Use Cases |
|---------|-----------|----------------|-----------|
| `section_aware` | Rule-based section segmentation using IMRaD/SPL heuristics | `section_rules` | Regulatory filings, PubMed articles |
| `semantic_splitter` | Embedding coherence drift detection | `tau_coh`, `min_tokens`, `model_name` | Dense retrieval pipelines, Q&A |
| `sliding_window` | Fixed token windows with overlap | `target_tokens`, `overlap_ratio` | Dense embedding backfill, fallback |
| `table` | Table preservation with row/summary modes | `mode`, `include_caption` | Structured data ingestion |
| `clinical_role` | PICO and endpoint heuristics | `taxonomy_path`, `role_threshold` | Clinical trial extraction |

## Experimental Chunkers

| Chunker | Description |
|---------|-------------|
| `llm_chaptering` | Few-shot prompted boundary detection with semantic drift validation |
| `discourse_segmenter` | Connective-driven EDU segmentation |
| `grobid_section` | Aligns MinerU output with Grobid TEI sections |
| `layout_aware` | Groups blocks by layout regions (DocTR/Docling output) |
| `graph_rag` | Jaccard similarity graph with community summarisation |
| `text_tiling`, `c99`, `bayes_seg`, `lda_topic` | Classical lexical and topic segmentation techniques |
| `semantic_cluster`, `graph_partition` | Embedding clustering and community detection |

## Adapter Chunkers

| Adapter | Framework | Notes |
|---------|-----------|-------|
| `langchain.*` | LangChain text splitters | Requires `langchain-text-splitters` |
| `llama_index.*` | LlamaIndex node parsers | Requires `llama-index-core` |
| `haystack.preprocessor` | Haystack PreProcessor | Requires `haystack-ai` |
| `unstructured.adapter` | Unstructured partitioner | Requires `unstructured` |

Refer to `docs/chunking/AdapterGuide.md` for integration details.
