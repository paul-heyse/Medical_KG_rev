# Legacy Decommission Checklist

| Legacy Component | Status | Notes |
| --- | --- | --- |
| `services/chunking/custom_splitters.py` | Removed | Replaced by protocol-driven chunkers (`simple`, `langchain_recursive`, `llamaindex_sentence_window`). |
| `services/chunking/semantic_splitter.py` | Removed | LlamaIndex wrapper supersedes semantic window behaviour. |
| `services/chunking/sliding_window.py` | Removed | Recursive character splitter handles sliding window needs. |
| `services/chunking/section_aware_splitter.py` | Removed | Profile-boundary logic lives in runtime grouping helpers. |
| `services/parsing/pdf_parser.py` | Removed | MinerU-only PDF parsing retained per proposal. |
| `services/parsing/xml_parser.py` | Removed | Unstructured parser handles XML/HTML. |
| `services/parsing/sentence_splitters.py` | Removed | Replaced with scispaCy and syntok wrappers. |
| Adapter `.split_document()` helpers | Removed | Adapters now delegate to `chunk_document` helper. |

All legacy test modules referencing the deleted implementations have also been removed in earlier commits. No residual imports of the removed modules remain (`rg` searches for `custom_splitters`, `pdf_parser`, and `xml_parser` returned no results under `src/`).
