"""Utilities for provenance tracking and title path construction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from Medical_KG_rev.models.ir import Block, BlockType, Document, Section

from .tokenization import TokenCounter, default_token_counter


@dataclass(slots=True, frozen=True)
class BlockContext:
    """Normalized view over IR blocks with positional metadata."""

    block: Block
    section: Section
    title_path: tuple[str, ...]
    text: str
    start_char: int
    end_char: int
    token_count: int
    page_no: int | None

    @property
    def section_title(self) -> str:
        return self.section.title or "Untitled"

    @property
    def is_table(self) -> bool:
        return self.block.type == BlockType.TABLE or self.block.metadata.get("is_table", False)


def _extract_page_number(metadata: dict[str, object]) -> int | None:
    """Best-effort extraction of page number from block metadata."""
    for key in ("page", "page_number", "page_no", "pageIndex"):
        value = metadata.get(key)
        if isinstance(value, int) and value >= 1:
            return value
        if isinstance(value, str) and value.isdigit():
            page = int(value)
            if page >= 1:
                return page
    return None


class ProvenanceNormalizer:
    """Transforms documents into block contexts used by chunkers."""

    def __init__(self, *, token_counter: TokenCounter | None = None) -> None:
        self.counter = token_counter or default_token_counter()

    def iter_block_contexts(self, document: Document) -> Iterable[BlockContext]:
        cursor = 0
        for section in document.sections:
            section_title = section.title or "Untitled"
            title_path = tuple(filter(None, (document.title, section_title))) or (section_title,)
            for block in section.blocks:
                text = (block.text or "").strip()
                token_count = self.counter.count(text)
                block_length = len(text)
                start = cursor
                end = start + block_length
                if block_length:
                    cursor = end + 1  # account for newline separators
                page = _extract_page_number(block.metadata)
                yield BlockContext(
                    block=block,
                    section=section,
                    title_path=title_path,
                    text=text,
                    start_char=start,
                    end_char=max(end, start + 1),
                    token_count=token_count,
                    page_no=page,
                )

    def total_tokens(self, contexts: Sequence[BlockContext]) -> int:
        return sum(context.token_count for context in contexts)


def make_chunk_id(doc_id: str, chunker: str, granularity: str, index: int) -> str:
    return f"{doc_id}:{chunker}:{granularity}:{index}".replace(" ", "_")
