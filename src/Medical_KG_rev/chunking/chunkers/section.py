"""Section aware chunker implementation."""

from __future__ import annotations

from collections.abc import Iterable

from ..base import ContextualChunker
from ..provenance import BlockContext
from ..segmentation import SectionSegmenter, Segment
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

    def _segment_section(self, contexts: list[BlockContext]) -> list[Segment]:
        buffer: list[BlockContext] = []
        token_budget = 0
        text_segments: list[list[BlockContext]] = []
        result: list[Segment] = []

        def flush_buffer() -> None:
            nonlocal buffer, token_budget
            if buffer:
                text_segments.append(buffer)
                buffer = []
                token_budget = 0

        def flush_text_segments() -> None:
            nonlocal text_segments
            if not text_segments:
                return
            for segment in self._merge_small_tail(text_segments):
                result.append(Segment(contexts=list(segment)))
            text_segments = []

        for ctx in contexts:
            if ctx.is_table and self.preserve_tables:
                flush_buffer()
                flush_text_segments()
                result.append(
                    Segment(
                        contexts=[ctx],
                        metadata={"segment_type": "table", "is_table": True},
                    )
                )
                continue
            buffer.append(ctx)
            token_budget += ctx.token_count
            if token_budget >= self.target_tokens:
                flush_buffer()
        flush_buffer()
        flush_text_segments()
        return result

    def _merge_small_tail(
        self, segments: list[list[BlockContext]]
    ) -> list[list[BlockContext]]:
        if not segments:
            return []
        if len(segments) == 1:
            return segments
        last = segments[-1]
        last_tokens = sum(ctx.token_count for ctx in last)
        if last_tokens < self.min_tokens:
            penultimate = segments[-2]
            combined = penultimate + last
            if sum(ctx.token_count for ctx in combined) <= self.max_tokens:
                segments = segments[:-2] + [combined]
        return segments
