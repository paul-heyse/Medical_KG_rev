# Deleted Chunking and Parsing Modules

The add-parsing-chunking-normalization change permanently removed the bespoke
chunking/parsing codepaths in favor of the new profile-aware runtime. Legacy
modules eliminated as part of this migration:

| Legacy module | Replacement |
|---------------|-------------|
| `src/Medical_KG_rev/services/chunking/custom_splitters.py` | ChunkerPort with LangChain/Hugging Face wrappers |
| `src/Medical_KG_rev/services/chunking/semantic_splitter.py` | `LangChainChunker` (recursive splitter) |
| `src/Medical_KG_rev/services/chunking/sliding_window.py` | LangChain RecursiveCharacterTextSplitter profiles |
| `src/Medical_KG_rev/services/chunking/section_aware_splitter.py` | Profile chunkers (`profile_chunkers.py`) |
| `src/Medical_KG_rev/services/parsing/pdf_parser.py` | MinerU GPU pipeline |
| `src/Medical_KG_rev/services/parsing/xml_parser.py` | `unstructured_parser.py` |
| `src/Medical_KG_rev/services/parsing/sentence_splitters.py` | `huggingface_segmenter.py` & `syntok_segmenter.py` |

All adapter-specific `.split_document()` helpers were removed; adapters now call
`Medical_KG_rev.services.chunking.chunk_document` to obtain chunks.
