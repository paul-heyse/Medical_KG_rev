"""Chunker registry and factory helpers."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .exceptions import ChunkerConfigurationError, ChunkerRegistryError
from .models import ChunkerConfig
from .ports import BaseChunker

logger = logging.getLogger(__name__)


@dataclass
class ChunkerRegistry:
    """Registry for chunker implementations."""

    _chunkers: dict[str, type[BaseChunker]] = None

    def __post_init__(self):
        if self._chunkers is None:
            self._chunkers = {}

    def register(self, name: str, chunker_class: type[BaseChunker]) -> None:
        """Register a chunker class."""
        self._chunkers[name] = chunker_class
        logger.debug("chunker.registered", name=name, class_name=chunker_class.__name__)

    def get(self, name: str) -> type[BaseChunker] | None:
        """Get a chunker class by name."""
        return self._chunkers.get(name)

    def list_chunkers(self) -> list[str]:
        """List all registered chunker names."""
        return list(self._chunkers.keys())

    def create_chunker(self, config: ChunkerConfig) -> BaseChunker:
        """Create a chunker instance from configuration."""
        chunker_class = self.get(config.name)
        if chunker_class is None:
            raise ChunkerRegistryError(f"Unknown chunker: {config.name}")

        try:
            return chunker_class(**config.parameters)
        except Exception as exc:
            raise ChunkerConfigurationError(
                f"Failed to create chunker '{config.name}': {exc}"
            ) from exc


def _maybe_register(
    registry: ChunkerRegistry,
    module_path: str,
    attribute: str,
    name: str,
    *,
    experimental: bool = False,
) -> None:
    """Attempt to register an optional chunker."""
    try:
        module = importlib.import_module(module_path)
        chunker_class = getattr(module, attribute)
        registry.register(name, chunker_class)
        logger.debug(
            "chunker.optional_registered",
            name=name,
            module_path=module_path,
            experimental=experimental,
        )
    except (ImportError, AttributeError) as exc:
        logger.debug(
            "chunker.optional_failed",
            name=name,
            module_path=module_path,
            error=str(exc),
            experimental=experimental,
        )


def create_default_registry() -> ChunkerRegistry:
    """Create the default chunker registry with core chunkers."""
    registry = ChunkerRegistry()

    # Register core chunkers
    try:
        from .chunkers.section import SectionAwareChunker
        registry.register("section_aware", SectionAwareChunker)
    except ImportError:
        pass

    try:
        from .chunkers.sliding_window import SlidingWindowChunker
        registry.register("sliding_window", SlidingWindowChunker)
    except ImportError:
        pass

    try:
        from .chunkers.table import TableChunker
        registry.register("table", TableChunker)
    except ImportError:
        pass

    try:
        from .chunkers.clinical_role import ClinicalRoleChunker
        registry.register("clinical_role", ClinicalRoleChunker)
    except ImportError:
        pass

    try:
        from .chunkers.layout import LayoutHeuristicChunker
        registry.register("layout_heuristic", LayoutHeuristicChunker)
    except ImportError:
        pass

    try:
        from .chunkers.docling import DoclingChunker
        registry.register("docling", DoclingChunker)
    except ImportError:
        pass

    try:
        from .hybrid_chunker import HybridChunker
        registry.register("hybrid", HybridChunker)
    except ImportError:
        pass

    # Register optional chunkers
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


# Create default registry instance
default_registry = create_default_registry()
