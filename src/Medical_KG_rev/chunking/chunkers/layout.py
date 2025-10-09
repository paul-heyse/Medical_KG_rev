"""Chunker that relies on layout heuristics such as headings and font deltas."""

from __future__ import annotations

from collections.abc import Iterable

from ..base import ContextualChunker
from ..provenance import BlockContext
from ..segmentation import LayoutSegmenter, Segment
from ..tokenization import TokenCounter


class LayoutHeuristicChunker(ContextualChunker):
    name = "layout_heuristic"
    version = "v1"
    segment_type = "layout"

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        max_tokens: int = 600,
        heading_level_key: str = "heading_level",
        font_size_key: str = "font_size",
        whitespace_threshold: float = 0.25,
        font_delta_threshold: float = 2.0,
        segmenter: LayoutSegmenter | None = None,
    ) -> None:
        strategy = segmenter or LayoutSegmenter(
            max_tokens=max_tokens,
            heading_level_key=heading_level_key,
            font_size_key=font_size_key,
            whitespace_threshold=whitespace_threshold,
            font_delta_threshold=font_delta_threshold,
        )
        super().__init__(token_counter=token_counter, segmenter=strategy)
        self._segmenter = strategy
        self.max_tokens = strategy.max_tokens
        self.heading_level_key = strategy.heading_level_key
        self.font_size_key = strategy.font_size_key
        self.whitespace_threshold = strategy.whitespace_threshold
        self.font_delta_threshold = strategy.font_delta_threshold

    def segment_contexts(self, contexts: Iterable[BlockContext]) -> Iterable[Segment]:
        return self._segmenter.plan(list(contexts))

    def explain(self) -> dict[str, object]:
        return {
            "max_tokens": self.max_tokens,
            "heading_level_key": self.heading_level_key,
            "font_size_key": self.font_size_key,
            "whitespace_threshold": self.whitespace_threshold,
            "font_delta_threshold": self.font_delta_threshold,
        }
