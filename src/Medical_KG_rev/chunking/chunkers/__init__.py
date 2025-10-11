"""Lazy exports for built-in chunker implementations."""

from __future__ import annotations

from typing import Any

from importlib import import_module


__all__ = [
    "BayesSegChunker",
    "C99Chunker",
    "ClinicalRoleChunker",
    "DiscourseSegmenterChunker",
    "GraphPartitionChunker",
    "GraphRAGChunker",
    "GrobidSectionChunker",
    "LDATopicChunker",
    "LLMChapteringChunker",
    "LayoutAwareChunker",
    "LayoutHeuristicChunker",
    "SectionAwareChunker",
    "SemanticClusterChunker",
    "SemanticSplitterChunker",
    "SlidingWindowChunker",
    "TableChunker",
    "TextTilingChunker",
]

_CHUNKER_MAP = {
    "BayesSegChunker": ("Medical_KG_rev.chunking.chunkers.classical", "BayesSegChunker"),
    "C99Chunker": ("Medical_KG_rev.chunking.chunkers.classical", "C99Chunker"),
    "DiscourseSegmenterChunker": (
        "Medical_KG_rev.chunking.chunkers.advanced",
        "DiscourseSegmenterChunker",
    ),
    "ClinicalRoleChunker": (
        "Medical_KG_rev.chunking.chunkers.clinical_role",
        "ClinicalRoleChunker",
    ),
    "LDATopicChunker": ("Medical_KG_rev.chunking.chunkers.classical", "LDATopicChunker"),
    "GraphPartitionChunker": (
        "Medical_KG_rev.chunking.chunkers.semantic",
        "GraphPartitionChunker",
    ),
    "GraphRAGChunker": (
        "Medical_KG_rev.chunking.chunkers.advanced",
        "GraphRAGChunker",
    ),
    "GrobidSectionChunker": (
        "Medical_KG_rev.chunking.chunkers.advanced",
        "GrobidSectionChunker",
    ),
    "LayoutAwareChunker": (
        "Medical_KG_rev.chunking.chunkers.advanced",
        "LayoutAwareChunker",
    ),
    "LayoutHeuristicChunker": (
        "Medical_KG_rev.chunking.chunkers.layout",
        "LayoutHeuristicChunker",
    ),
    "SectionAwareChunker": (
        "Medical_KG_rev.chunking.chunkers.section",
        "SectionAwareChunker",
    ),
    "SemanticClusterChunker": (
        "Medical_KG_rev.chunking.chunkers.semantic",
        "SemanticClusterChunker",
    ),
    "SemanticSplitterChunker": (
        "Medical_KG_rev.chunking.chunkers.semantic",
        "SemanticSplitterChunker",
    ),
    "SlidingWindowChunker": (
        "Medical_KG_rev.chunking.chunkers.sliding_window",
        "SlidingWindowChunker",
    ),
    "TableChunker": ("Medical_KG_rev.chunking.chunkers.table", "TableChunker"),
    "TextTilingChunker": (
        "Medical_KG_rev.chunking.chunkers.classical",
        "TextTilingChunker",
    ),
    "LLMChapteringChunker": (
        "Medical_KG_rev.chunking.chunkers.llm",
        "LLMChapteringChunker",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_path, attribute = _CHUNKER_MAP[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise AttributeError(name) from exc
    module = import_module(module_path)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:  # pragma: no cover - optional dependency missing
        raise AttributeError(name) from exc


def __dir__() -> list[str]:  # pragma: no cover - tooling aid
    return sorted(set(__all__))
