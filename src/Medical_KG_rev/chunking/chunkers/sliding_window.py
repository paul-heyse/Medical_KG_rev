"""Sliding window chunker with overlap support."""

from __future__ import annotations

from typing import Iterable

from ..base import ContextualChunker
from ..provenance import BlockContext
from ..segmentation import Segment, SlidingWindowSegmenter
from ..tokenization import TokenCounter


class SlidingWindowChunker(ContextualChunker):
    name = "sliding_window"
    version = "v1"
    default_granularity = "window"
    segment_type = "window"

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        target_tokens: int = 512,
        overlap_ratio: float = 0.25,
        min_tokens: int = 128,
        segmenter: SlidingWindowSegmenter | None = None,
    ) -> None:
        strategy = segmenter or SlidingWindowSegmenter(
            target_tokens=target_tokens,
            overlap_ratio=overlap_ratio,
            min_tokens=min_tokens,
        )
        super().__init__(token_counter=token_counter, segmenter=strategy)
        self._segmenter = strategy
        self.target_tokens = strategy.target_tokens
        self.overlap_ratio = strategy.overlap_ratio
        self.min_tokens = strategy.min_tokens

    def segment_contexts(self, contexts: Iterable[BlockContext]) -> Iterable[Segment]:
        return self._segmenter.plan(list(contexts))

    def explain(self) -> dict[str, object]:
        return {
            "target_tokens": self.target_tokens,
            "overlap_ratio": self.overlap_ratio,
            "min_tokens": self.min_tokens,
        }
