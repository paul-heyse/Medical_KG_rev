"""Chunker registry and factory helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Type

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


def default_registry() -> ChunkerRegistry:
    from .adapters import (
        HaystackPreprocessorChunker,
        LangChainHTMLChunker,
        LangChainMarkdownChunker,
        LangChainNLTKChunker,
        LangChainSpacyChunker,
        LangChainSplitterChunker,
        LangChainTokenSplitterChunker,
        LlamaIndexHierarchicalChunker,
        LlamaIndexNodeParserChunker,
        LlamaIndexSentenceChunker,
        UnstructuredChunker,
    )
    from .chunkers import (
        BayesSegChunker,
        C99Chunker,
        ClinicalRoleChunker,
        DiscourseSegmenterChunker,
        GraphPartitionChunker,
        GraphRAGChunker,
        GrobidSectionChunker,
        LDATopicChunker,
        LayoutAwareChunker,
        LayoutHeuristicChunker,
        SectionAwareChunker,
        SemanticClusterChunker,
        SemanticSplitterChunker,
        SlidingWindowChunker,
        TableChunker,
        TextTilingChunker,
    )
    from .chunkers.llm import LLMChapteringChunker

    registry = ChunkerRegistry()
    registry.register("section_aware", SectionAwareChunker)
    registry.register("sliding_window", SlidingWindowChunker)
    registry.register("table", TableChunker)
    registry.register("semantic_splitter", SemanticSplitterChunker)
    registry.register("clinical_role", ClinicalRoleChunker)
    registry.register("layout_heuristic", LayoutHeuristicChunker)
    registry.register("semantic_cluster", SemanticClusterChunker, experimental=True)
    registry.register("graph_partition", GraphPartitionChunker, experimental=True)
    registry.register("graph_rag", GraphRAGChunker, experimental=True)
    registry.register("text_tiling", TextTilingChunker, experimental=True)
    registry.register("c99", C99Chunker, experimental=True)
    registry.register("bayes_seg", BayesSegChunker, experimental=True)
    registry.register("lda_topic", LDATopicChunker, experimental=True)
    registry.register("discourse_segmenter", DiscourseSegmenterChunker, experimental=True)
    registry.register("grobid_section", GrobidSectionChunker, experimental=True)
    registry.register("layout_aware", LayoutAwareChunker, experimental=True)
    registry.register("llm_chaptering", LLMChapteringChunker, experimental=True)
    registry.register("langchain.recursive_character", LangChainSplitterChunker)
    registry.register("langchain.token", LangChainTokenSplitterChunker)
    registry.register("langchain.markdown", LangChainMarkdownChunker)
    registry.register("langchain.html", LangChainHTMLChunker)
    registry.register("langchain.nltk", LangChainNLTKChunker)
    registry.register("langchain.spacy", LangChainSpacyChunker)
    registry.register("llama_index.semantic_splitter", LlamaIndexNodeParserChunker, experimental=True)
    registry.register("llama_index.hierarchical", LlamaIndexHierarchicalChunker, experimental=True)
    registry.register("llama_index.sentence", LlamaIndexSentenceChunker, experimental=True)
    registry.register("haystack.preprocessor", HaystackPreprocessorChunker, experimental=True)
    registry.register("unstructured.adapter", UnstructuredChunker, experimental=True)
    return registry
