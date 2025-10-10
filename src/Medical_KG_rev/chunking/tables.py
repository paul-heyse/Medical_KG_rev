"""Utilities for handling table preservation across chunkers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .provenance import BlockContext
from .tokenization import TokenCounter, default_token_counter


@dataclass(slots=True)
class TableSlice:
    """Represents a logical slice of a table context."""

    contexts: Sequence[BlockContext]
    metadata: dict[str, object]


def _clone_context(
    base: BlockContext,
    *,
    text: str,
    start_offset: int,
    end_offset: int,
    counter: TokenCounter,
) -> BlockContext:
    """Clone a block context with updated text and offsets."""
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


class TableHandler:
    """Utility that transforms table contexts into logical slices."""

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        mode: str = "row",
        rowgroup_size: int = 2,
    ) -> None:
        if mode not in {"row", "rowgroup", "summary"}:
            raise ValueError("Unsupported table handling mode")
        if rowgroup_size <= 0:
            raise ValueError("rowgroup_size must be positive")
        self.mode = mode
        self.counter = token_counter or default_token_counter()
        self.rowgroup_size = rowgroup_size

    def iter_slices(self, context: BlockContext) -> Iterable[TableSlice]:
        """Yield logical slices for a table block."""
        if not context.text:
            return []
        rows = self._extract_rows(context)
        if not rows:
            return [TableSlice(contexts=[context], metadata={"mode": self.mode})]
        if self.mode == "summary":
            summary_text = rows[0]["text"]
            if caption := context.block.metadata.get("caption"):
                summary_text = f"{caption}\n{summary_text}"
            summary_context = _clone_context(
                context,
                text=summary_text,
                start_offset=rows[0]["start"],
                end_offset=rows[0]["end"],
                counter=self.counter,
            )
            return [
                TableSlice(
                    contexts=[summary_context],
                    metadata={
                        "mode": self.mode,
                        "segment_type": "table",
                        "row_indices": [rows[0]["index"]],
                        "is_summary": True,
                    },
                )
            ]

        slices: list[TableSlice] = []
        buffer: list[BlockContext] = []
        row_indices: list[int] = []
        for row in rows:
            row_ctx = _clone_context(
                context,
                text=row["text"],
                start_offset=row["start"],
                end_offset=row["end"],
                counter=self.counter,
            )
            buffer.append(row_ctx)
            row_indices.append(row["index"])
            if self.mode == "row" or len(buffer) >= self.rowgroup_size:
                slices.append(
                    TableSlice(
                        contexts=list(buffer),
                        metadata={
                            "mode": self.mode,
                            "segment_type": "table",
                            "row_indices": list(row_indices),
                        },
                    )
                )
                buffer.clear()
                row_indices.clear()
        if buffer:
            slices.append(
                TableSlice(
                    contexts=list(buffer),
                    metadata={
                        "mode": self.mode,
                        "segment_type": "table",
                        "row_indices": list(row_indices),
                        "is_partial": True,
                    },
                )
            )
        return slices

    def _extract_rows(self, context: BlockContext) -> list[dict[str, object]]:
        """Parse raw table text into logical rows with offsets."""
        metadata_rows: Sequence[str] | None = None
        meta = context.block.metadata
        if isinstance(meta, dict):
            metadata_rows = meta.get("table_rows") or meta.get("rows")  # type: ignore[assignment]
        if metadata_rows:
            rows = [str(row) for row in metadata_rows if str(row).strip()]
        else:
            rows = [line.strip() for line in context.text.splitlines() if line.strip()]
        cursor = 0
        normalized_rows: list[dict[str, object]] = []
        for index, row in enumerate(rows):
            start = context.text.find(row, cursor)
            if start < 0:
                start = cursor
            end = start + len(row)
            cursor = end
            normalized_rows.append({"index": index, "text": row, "start": start, "end": end})
        return normalized_rows
