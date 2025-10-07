"""Sliding window chunker with overlap support."""

from __future__ import annotations

from typing import Iterable

from ..base import ContextualChunker, Segment
from ..provenance import BlockContext
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
    ) -> None:
        if not (0.0 <= overlap_ratio < 1.0):
            raise ValueError("overlap_ratio must be between 0 and 1")
        super().__init__(token_counter=token_counter)
        self.target_tokens = target_tokens
        self.overlap_ratio = overlap_ratio
        self.min_tokens = min_tokens

    def segment_contexts(self, contexts: Iterable[BlockContext]) -> Iterable[Segment]:
        contexts_list = list(contexts)
        if not contexts_list:
            return []
        segments: list[Segment] = []
        index = 0
        length = len(contexts_list)
        while index < length:
            window: list[BlockContext] = []
            token_total = 0
            j = index
            while j < length and token_total < self.target_tokens:
                window.append(contexts_list[j])
                token_total += contexts_list[j].token_count
                j += 1
            if not window:
                break
            segments.append(Segment(contexts=list(window)))
            step = max(1, int(len(window) * (1 - self.overlap_ratio)))
            index += step
        return self._merge_short_segments(segments)

    def explain(self) -> dict[str, object]:
        return {
            "target_tokens": self.target_tokens,
            "overlap_ratio": self.overlap_ratio,
            "min_tokens": self.min_tokens,
        }

    def _merge_short_segments(self, segments: list[Segment]) -> list[Segment]:
        if self.min_tokens <= 0 or len(segments) <= 1:
            return segments
        merged: list[Segment] = []
        for segment in segments:
            token_total = sum(ctx.token_count for ctx in segment.contexts)
            if merged and token_total < self.min_tokens:
                previous = merged.pop()
                merged.append(
                    Segment(
                        contexts=list(previous.contexts) + list(segment.contexts),
                        metadata=previous.metadata,
                    )
                )
            else:
                merged.append(segment)
        return merged
