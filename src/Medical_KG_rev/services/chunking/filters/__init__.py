"""Filter chain utilities for preprocessing documents prior to chunking."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
import re

from Medical_KG_rev.models.ir import Block, BlockType, Document, Section


FilterFunc = Callable[[Document], Document]


_HEADER_FOOTER_PATTERN = re.compile(r"page\s+\d+\s+of\s+\d+", re.IGNORECASE)


def _clone_block(block: Block, *, text: str | None = None, metadata: dict | None = None) -> Block:
    data = block.model_dump()
    if text is not None:
        data["text"] = text
    if metadata is not None:
        data["metadata"] = metadata
    return Block(**data)


def _clone_section(section: Section, *, blocks: Iterable[Block]) -> Section:
    data = section.model_dump()
    data["blocks"] = list(blocks)
    return Section(**data)


def drop_boilerplate(document: Document) -> Document:
    """Remove repeated headers/footers and empty blocks."""
    sections: list[Section] = []
    for section in document.sections:
        filtered_blocks: list[Block] = []
        for block in section.blocks:
            text = (block.text or "").strip()
            metadata = block.metadata or {}
            role = str(metadata.get("role", ""))
            if not text:
                continue
            if role.lower() in {"header", "footer"}:
                continue
            if _HEADER_FOOTER_PATTERN.search(text):
                continue
            filtered_blocks.append(block)
        if filtered_blocks:
            sections.append(_clone_section(section, blocks=filtered_blocks))
    if not sections:
        return document
    return document.model_copy(update={"sections": sections})


def exclude_references(document: Document) -> Document:
    """Drop sections that correspond to references/bibliographies."""
    retained: list[Section] = []
    for section in document.sections:
        title = (section.title or "").strip().lower()
        if title and "reference" in title:
            continue
        retained.append(section)
    if not retained:
        return document
    return document.model_copy(update={"sections": retained})


def deduplicate_page_furniture(document: Document) -> Document:
    """Remove duplicate short blocks that typically represent running headers."""
    seen: Counter[str] = Counter()
    for section in document.sections:
        for block in section.blocks:
            text = (block.text or "").strip()
            if not text:
                continue
            if len(text) <= 80:
                seen[text] += 1
    sections: list[Section] = []
    for section in document.sections:
        filtered_blocks: list[Block] = []
        for block in section.blocks:
            text = (block.text or "").strip()
            if text and len(text) <= 80 and seen[text] > 1:
                continue
            filtered_blocks.append(block)
        if filtered_blocks:
            sections.append(_clone_section(section, blocks=filtered_blocks))
    if not sections:
        return document
    return document.model_copy(update={"sections": sections})


def preserve_tables_html(document: Document) -> Document:
    """Annotate low-confidence tables so downstream chunkers retain HTML."""
    sections: list[Section] = []
    for section in document.sections:
        new_blocks: list[Block] = []
        for block in section.blocks:
            if block.type != BlockType.TABLE:
                new_blocks.append(block)
                continue
            metadata = dict(block.metadata)
            confidence = metadata.get("rectangularize_confidence")
            if confidence is not None and confidence < 0.8:
                metadata["is_unparsed_table"] = True
                new_blocks.append(_clone_block(block, metadata=metadata))
            else:
                new_blocks.append(block)
        if new_blocks:
            sections.append(_clone_section(section, blocks=new_blocks))
    if not sections:
        return document
    return document.model_copy(update={"sections": sections})


FILTERS: dict[str, FilterFunc] = {
    "drop_boilerplate": drop_boilerplate,
    "exclude_references": exclude_references,
    "deduplicate_page_furniture": deduplicate_page_furniture,
    "preserve_tables_html": preserve_tables_html,
}


def apply_filter_chain(document: Document, filter_names: Iterable[str]) -> Document:
    """Apply configured filters to *document* in order."""
    current = document
    for name in filter_names:
        func = FILTERS.get(name)
        if func is None:
            continue
        current = func(current)
    return current


__all__ = ["FILTERS", "apply_filter_chain"]
