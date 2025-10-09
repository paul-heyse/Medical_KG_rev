"""Offset mapping utilities used by framework adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..provenance import BlockContext
from ..tokenization import TokenCounter, default_token_counter


@dataclass(slots=True)
class SegmentProjection:
    contexts: List[BlockContext]
    end_offset: int


def _clone_context(
    base: BlockContext,
    *,
    text: str,
    start_offset: int,
    end_offset: int,
    counter: TokenCounter,
) -> BlockContext:
    return BlockContext(
        block=base.block,
        section=base.section,
        title_path=base.title_path,
        text=text,
        start_char=base.start_char + start_offset,
        end_char=base.start_char + end_offset,
        token_count=counter.count(text),
        page_no=base.page_no,
    )


class OffsetMapper:
    """Maps string segments back to block contexts."""

    def __init__(
        self,
        contexts: Sequence[BlockContext],
        *,
        separator: str = "\n\n",
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.contexts = [ctx for ctx in contexts if ctx.text]
        self.separator = separator
        self.counter = token_counter or default_token_counter()
        self.aggregated_text = separator.join(ctx.text for ctx in self.contexts)
        self._spans: list[Tuple[int, int, BlockContext]] = []
        cursor = 0
        for ctx in self.contexts:
            start = cursor
            end = start + len(ctx.text)
            self._spans.append((start, end, ctx))
            cursor = end + len(separator)

    def project(self, text_segment: str, *, start_hint: int = 0) -> SegmentProjection:
        if not text_segment.strip():
            return SegmentProjection(contexts=[], end_offset=start_hint)
        text = text_segment.strip("\n")
        idx = self.aggregated_text.find(text, start_hint)
        if idx < 0:
            idx = self.aggregated_text.find(text.strip(), start_hint)
        if idx < 0:
            raise ValueError("Unable to map segment back to contexts")
        end = idx + len(text)
        sliced: list[BlockContext] = []
        for span_start, span_end, ctx in self._spans:
            if span_end <= idx or span_start >= end:
                continue
            local_start = max(idx, span_start)
            local_end = min(end, span_end)
            rel_start = local_start - span_start
            rel_end = local_end - span_start
            text_slice = ctx.text[rel_start:rel_end]
            sliced.append(
                _clone_context(
                    ctx,
                    text=text_slice,
                    start_offset=rel_start,
                    end_offset=rel_end,
                    counter=self.counter,
                )
            )
        return SegmentProjection(contexts=sliced, end_offset=end)
