"""Section aware chunker implementation."""

from __future__ import annotations

from typing import Iterable

from ..base import ContextualChunker
from ..provenance import BlockContext
from ..segmentation import Segment, SectionSegmenter
from ..tokenization import TokenCounter


class SectionAwareChunker(ContextualChunker):
    """Chunker that respects document sections with domain aware defaults."""

    name = "section_aware"
    version = "v1"
    default_granularity = "section"
    segment_type = "section"
    include_tables = True

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        target_tokens: int = 450,
        min_tokens: int = 180,
        max_tokens: int = 900,
        preserve_tables: bool = True,
        segmenter: SectionSegmenter | None = None,
    ) -> None:
        strategy = segmenter or SectionSegmenter(
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            preserve_tables=preserve_tables,
        )
        super().__init__(token_counter=token_counter, segmenter=strategy)
        self._segmenter = strategy
        self.target_tokens = strategy.target_tokens
        self.min_tokens = strategy.min_tokens
        self.max_tokens = strategy.max_tokens
        self.preserve_tables = strategy.preserve_tables

    def segment_contexts(self, contexts: Iterable[BlockContext]) -> Iterable[Segment]:
        return self._segmenter.plan(list(contexts))

    def explain(self) -> dict[str, object]:
        return {
            "target_tokens": self.target_tokens,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "preserve_tables": self.preserve_tables,
        }
