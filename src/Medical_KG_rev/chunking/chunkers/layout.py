"""Chunker that relies on layout heuristics such as headings and font deltas."""

from __future__ import annotations

from typing import Iterable

from Medical_KG_rev.models.ir import BlockType

from ..base import ContextualChunker, Segment
from ..provenance import BlockContext
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
    ) -> None:
        super().__init__(token_counter=token_counter)
        self.max_tokens = max_tokens
        self.heading_level_key = heading_level_key
        self.font_size_key = font_size_key
        self.whitespace_threshold = whitespace_threshold
        self.font_delta_threshold = font_delta_threshold

    def segment_contexts(self, contexts: Iterable[BlockContext]) -> Iterable[Segment]:
        segments: list[Segment] = []
        buffer: list[BlockContext] = []
        token_total = 0
        last_heading_level: int | None = None
        last_font_size: float | None = None
        for ctx in contexts:
            metadata = ctx.block.metadata or {}
            heading_level = int(metadata.get(self.heading_level_key, 9))
            font_size = float(metadata.get(self.font_size_key, 0.0))
            whitespace_ratio = float(metadata.get("whitespace_ratio", 0.0))
            is_header = ctx.block.type == BlockType.HEADER
            should_flush = False
            if buffer:
                if heading_level <= (last_heading_level or heading_level):
                    should_flush = True
                if (
                    last_font_size is not None
                    and abs(font_size - last_font_size) >= self.font_delta_threshold
                ):
                    should_flush = True
                if whitespace_ratio >= self.whitespace_threshold:
                    should_flush = True
            if should_flush and buffer:
                segments.append(
                    Segment(
                        contexts=list(buffer),
                        metadata={"heading_level": last_heading_level},
                    )
                )
                buffer = []
                token_total = 0
            buffer.append(ctx)
            token_total += ctx.token_count
            last_heading_level = heading_level if not is_header else 0
            last_font_size = font_size or last_font_size
            if token_total >= self.max_tokens:
                segments.append(
                    Segment(
                        contexts=list(buffer),
                        metadata={
                            "heading_level": heading_level,
                            "token_budget_exhausted": True,
                        },
                    )
                )
                buffer = []
                token_total = 0
        if buffer:
            segments.append(
                Segment(
                    contexts=list(buffer),
                    metadata={"heading_level": last_heading_level},
                )
            )
        return segments

    def explain(self) -> dict[str, object]:
        return {
            "max_tokens": self.max_tokens,
            "heading_level_key": self.heading_level_key,
            "font_size_key": self.font_size_key,
            "whitespace_threshold": self.whitespace_threshold,
            "font_delta_threshold": self.font_delta_threshold,
        }
