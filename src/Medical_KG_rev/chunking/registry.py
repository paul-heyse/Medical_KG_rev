"""Chunker registry and factory helpers."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

from .exceptions import ChunkerConfigurationError, ChunkerRegistryError
from .models import ChunkerConfig
from .ports import BaseChunker


@dataclass(slots=True)
class ChunkerEntry:
    name: str
    factory: Callable[..., BaseChunker]
    experimental: bool = False


class ChunkerRegistry:
    """Registry storing mappings from names to chunker factories."""

    def __init__(self) -> None:
        self._entries: dict[str, ChunkerEntry] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., BaseChunker],
        *,
        experimental: bool = False,
    ) -> None:
        if name in self._entries:
            raise ChunkerRegistryError(f"Chunker '{name}' already registered")
        self._entries[name] = ChunkerEntry(name, factory, experimental=experimental)

    def create(self, config: ChunkerConfig, *, allow_experimental: bool = False) -> BaseChunker:
        entry = self._entries.get(config.name)
        if entry is None:
            raise ChunkerConfigurationError(f"Chunker '{config.name}' is not registered")
        if entry.experimental and not allow_experimental:
            raise ChunkerConfigurationError(
                f"Chunker '{config.name}' is experimental and not enabled"
            )
        return entry.factory(**config.params)

    def list_chunkers(self, *, include_experimental: bool = False) -> dict[str, ChunkerEntry]:
        if include_experimental:
            return dict(self._entries)
        return {name: entry for name, entry in self._entries.items() if not entry.experimental}


def _maybe_register(
    registry: ChunkerRegistry,
    module_path: str,
    attribute: str,
    name: str,
    *,
    experimental: bool = False,
) -> None:
    try:
        module = importlib.import_module(module_path)
        factory = getattr(module, attribute)
    except Exception:
        return
    registry.register(name, factory, experimental=experimental)


def default_registry() -> ChunkerRegistry:
    from .chunkers.clinical_role import ClinicalRoleChunker
    from .chunkers.layout import LayoutHeuristicChunker
    from .chunkers.section import SectionAwareChunker
    from .chunkers.sliding_window import SlidingWindowChunker
    from .chunkers.table import TableChunker

    registry = ChunkerRegistry()
    registry.register("section_aware", SectionAwareChunker)
    registry.register("sliding_window", SlidingWindowChunker)
    registry.register("table", TableChunker)
    registry.register("clinical_role", ClinicalRoleChunker)
    registry.register("layout_heuristic", LayoutHeuristicChunker)

    optional_specs = [
        (
            "Medical_KG_rev.chunking.chunkers.semantic",
            "SemanticSplitterChunker",
            "semantic_splitter",
            False,
        ),
        (
            "Medical_KG_rev.chunking.chunkers.semantic",
            "SemanticClusterChunker",
            "semantic_cluster",
            True,
        ),
        (
            "Medical_KG_rev.chunking.chunkers.semantic",
            "GraphPartitionChunker",
            "graph_partition",
            True,
        ),
        ("Medical_KG_rev.chunking.chunkers.advanced", "GraphRAGChunker", "graph_rag", True),
        (
            "Medical_KG_rev.chunking.chunkers.advanced",
            "DiscourseSegmenterChunker",
            "discourse_segmenter",
            True,
        ),
        (
            "Medical_KG_rev.chunking.chunkers.advanced",
            "GrobidSectionChunker",
            "grobid_section",
            True,
        ),
        ("Medical_KG_rev.chunking.chunkers.advanced", "LayoutAwareChunker", "layout_aware", True),
        ("Medical_KG_rev.chunking.chunkers.classical", "TextTilingChunker", "text_tiling", True),
        ("Medical_KG_rev.chunking.chunkers.classical", "C99Chunker", "c99", True),
        ("Medical_KG_rev.chunking.chunkers.classical", "BayesSegChunker", "bayes_seg", True),
        ("Medical_KG_rev.chunking.chunkers.classical", "LDATopicChunker", "lda_topic", True),
        ("Medical_KG_rev.chunking.chunkers.llm", "LLMChapteringChunker", "llm_chaptering", True),
    ]

    optional_adapters = [
        (
            "Medical_KG_rev.chunking.adapters.langchain",
            "LangChainSplitterChunker",
            "langchain.recursive_character",
            False,
        ),
        (
            "Medical_KG_rev.chunking.adapters.langchain",
            "LangChainTokenSplitterChunker",
            "langchain.token",
            False,
        ),
        (
            "Medical_KG_rev.chunking.adapters.langchain",
            "LangChainMarkdownChunker",
            "langchain.markdown",
            False,
        ),
        (
            "Medical_KG_rev.chunking.adapters.langchain",
            "LangChainHTMLChunker",
            "langchain.html",
            False,
        ),
        (
            "Medical_KG_rev.chunking.adapters.langchain",
            "LangChainNLTKChunker",
            "langchain.nltk",
            False,
        ),
        (
            "Medical_KG_rev.chunking.adapters.langchain",
            "LangChainSpacyChunker",
            "langchain.spacy",
            False,
        ),
        (
            "Medical_KG_rev.chunking.adapters.llamaindex",
            "LlamaIndexNodeParserChunker",
            "llama_index.semantic_splitter",
            True,
        ),
        (
            "Medical_KG_rev.chunking.adapters.llamaindex",
            "LlamaIndexHierarchicalChunker",
            "llama_index.hierarchical",
            True,
        ),
        (
            "Medical_KG_rev.chunking.adapters.llamaindex",
            "LlamaIndexSentenceChunker",
            "llama_index.sentence",
            True,
        ),
        (
            "Medical_KG_rev.chunking.adapters.haystack",
            "HaystackPreprocessorChunker",
            "haystack.preprocessor",
            True,
        ),
        (
            "Medical_KG_rev.chunking.adapters.unstructured_adapter",
            "UnstructuredChunker",
            "unstructured.adapter",
            True,
        ),
    ]

    for module_path, attribute, name, experimental in [*optional_specs, *optional_adapters]:
        _maybe_register(
            registry,
            module_path,
            attribute,
            name,
            experimental=experimental,
        )

    return registry
