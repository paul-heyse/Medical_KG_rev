"""Reusable segmentation strategies for contextual chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import groupby
from typing import Iterable, Sequence

from Medical_KG_rev.models.ir import BlockType

from .provenance import BlockContext


@dataclass(slots=True)
class Segment:
    """A contiguous span of block contexts to assemble into a chunk."""

    contexts: list[BlockContext]
    metadata: dict[str, object] | None = None


@dataclass(slots=True)
class SegmentAccumulator:
    """Utility to build and flush contiguous block spans."""

    contexts: list[BlockContext] = field(default_factory=list)
    metadata: dict[str, object] | None = None

    def add(self, context: BlockContext) -> None:
        self.contexts.append(context)

    def extend(self, contexts: Iterable[BlockContext]) -> None:
        self.contexts.extend(contexts)

    def token_total(self) -> int:
        return sum(context.token_count for context in self.contexts)

    def clear(self) -> None:
        self.contexts.clear()
        self.metadata = None

    def flush(self, *, metadata: dict[str, object] | None = None) -> Segment | None:
        if not self.contexts:
            return None
        merged: dict[str, object] = {}
        if self.metadata:
            merged.update(self.metadata)
        if metadata:
            merged.update(metadata)
        segment = Segment(contexts=list(self.contexts), metadata=merged or None)
        self.clear()
        return segment


class Segmenter(ABC):
    """Abstract segment planner producing contiguous spans."""

    @abstractmethod
    def plan(self, contexts: Sequence[BlockContext]) -> Iterable[Segment]:
        """Yield contiguous segments for the provided contexts."""


class SlidingWindowSegmenter(Segmenter):
    """Segment contexts with a sliding token window and overlap."""

    def __init__(
        self,
        *,
        target_tokens: int = 512,
        overlap_ratio: float = 0.25,
        min_tokens: int = 128,
    ) -> None:
        if not (0.0 <= overlap_ratio < 1.0):
            raise ValueError("overlap_ratio must be between 0 and 1")
        self.target_tokens = target_tokens
        self.overlap_ratio = overlap_ratio
        self.min_tokens = min_tokens

    def plan(self, contexts: Sequence[BlockContext]) -> Iterable[Segment]:
        if not contexts:
            return []
        windows: list[Segment] = []
        index = 0
        length = len(contexts)
        while index < length:
            accumulator = SegmentAccumulator()
            token_total = 0
            j = index
            while j < length and token_total < self.target_tokens:
                accumulator.add(contexts[j])
                token_total += contexts[j].token_count
                j += 1
            segment = accumulator.flush()
            if segment is None:
                break
            windows.append(segment)
            step = max(1, int(len(segment.contexts) * (1 - self.overlap_ratio)))
            index += step
        return self._merge_short_windows(windows)

    def _merge_short_windows(self, windows: list[Segment]) -> list[Segment]:
        if self.min_tokens <= 0 or len(windows) <= 1:
            return windows
        merged: list[Segment] = []
        for window in windows:
            token_total = sum(ctx.token_count for ctx in window.contexts)
            if merged and token_total < self.min_tokens:
                previous = merged.pop()
                combined_contexts = [*previous.contexts, *window.contexts]
                combined_metadata: dict[str, object] | None = None
                if previous.metadata or window.metadata:
                    combined_metadata = {**(previous.metadata or {})}
                    combined_metadata.update(window.metadata or {})
                merged.append(Segment(contexts=combined_contexts, metadata=combined_metadata))
            else:
                merged.append(window)
        return merged


class SectionSegmenter(Segmenter):
    """Segmenter that respects document sections and optional table preservation."""

    def __init__(
        self,
        *,
        target_tokens: int = 450,
        min_tokens: int = 180,
        max_tokens: int = 900,
        preserve_tables: bool = True,
    ) -> None:
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.preserve_tables = preserve_tables

    def plan(self, contexts: Sequence[BlockContext]) -> Iterable[Segment]:
        segments: list[Segment] = []
        for _, section_contexts in groupby(contexts, key=lambda ctx: ctx.section.id):
            segments.extend(self._segment_section(list(section_contexts)))
        return segments

    def _segment_section(self, contexts: list[BlockContext]) -> list[Segment]:
        text_segments: list[list[BlockContext]] = []
        accumulator = SegmentAccumulator()
        results: list[Segment] = []

        def flush_text_segments() -> None:
            nonlocal text_segments
            if not text_segments:
                return
            for span in self._merge_small_tail(text_segments):
                results.append(Segment(contexts=list(span)))
            text_segments = []

        for context in contexts:
            if context.is_table and self.preserve_tables:
                if accumulator.contexts:
                    text_segments.append(list(accumulator.contexts))
                    accumulator.clear()
                flush_text_segments()
                results.append(
                    Segment(
                        contexts=[context],
                        metadata={"segment_type": "table", "is_table": True},
                    )
                )
                continue
            accumulator.add(context)
            if accumulator.token_total() >= self.target_tokens:
                text_segments.append(list(accumulator.contexts))
                accumulator.clear()
        if accumulator.contexts:
            text_segments.append(list(accumulator.contexts))
        flush_text_segments()
        return results

    def _merge_small_tail(self, spans: list[list[BlockContext]]) -> list[list[BlockContext]]:
        if not spans:
            return []
        if len(spans) == 1:
            return spans
        last = spans[-1]
        last_tokens = sum(ctx.token_count for ctx in last)
        if last_tokens < self.min_tokens:
            penultimate = spans[-2]
            combined = [*penultimate, *last]
            if sum(ctx.token_count for ctx in combined) <= self.max_tokens:
                spans = [*spans[:-2], combined]
        return spans


class LayoutSegmenter(Segmenter):
    """Segmenter relying on layout heuristics such as headings and whitespace."""

    def __init__(
        self,
        *,
        max_tokens: int = 600,
        heading_level_key: str = "heading_level",
        font_size_key: str = "font_size",
        whitespace_threshold: float = 0.25,
        font_delta_threshold: float = 2.0,
    ) -> None:
        self.max_tokens = max_tokens
        self.heading_level_key = heading_level_key
        self.font_size_key = font_size_key
        self.whitespace_threshold = whitespace_threshold
        self.font_delta_threshold = font_delta_threshold

    def plan(self, contexts: Sequence[BlockContext]) -> Iterable[Segment]:
        segments: list[Segment] = []
        accumulator = SegmentAccumulator()
        last_heading_level: int | None = None
        last_font_size: float | None = None

        for context in contexts:
            metadata = context.block.metadata or {}
            heading_level = int(metadata.get(self.heading_level_key, 9))
            font_size = float(metadata.get(self.font_size_key, 0.0))
            whitespace_ratio = float(metadata.get("whitespace_ratio", 0.0))
            is_header = context.block.type == BlockType.HEADER
            should_flush = False
            if accumulator.contexts:
                if heading_level <= (last_heading_level or heading_level):
                    should_flush = True
                if (
                    last_font_size is not None
                    and abs(font_size - last_font_size) >= self.font_delta_threshold
                ):
                    should_flush = True
                if whitespace_ratio >= self.whitespace_threshold:
                    should_flush = True
            if should_flush and accumulator.contexts:
                segment = accumulator.flush(metadata={"heading_level": last_heading_level})
                if segment:
                    segments.append(segment)
            accumulator.add(context)
            last_heading_level = heading_level if not is_header else 0
            last_font_size = font_size or last_font_size
            if accumulator.token_total() >= self.max_tokens:
                segment = accumulator.flush(
                    metadata={
                        "heading_level": heading_level,
                        "token_budget_exhausted": True,
                    }
                )
                if segment:
                    segments.append(segment)
        tail = accumulator.flush(metadata={"heading_level": last_heading_level})
        if tail:
            segments.append(tail)
        return segments
