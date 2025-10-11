"""Docling wrapper with explicit PDF guard."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from docling import partition

from Medical_KG_rev.models.figure import Figure
from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.models.table import Table, TableCell
from Medical_KG_rev.services.parsing.docling_vlm_service import DoclingVLMResult


def partition_to_document(doc_id: str, fmt: str, partitioned: Iterable[Any]) -> Document:
    """Convert Docling partition output into the internal document model."""

    blocks: list[Block] = []
    for idx, element in enumerate(partitioned):
        text = getattr(element, "text", "") or ""
        metadata = getattr(element, "metadata", {}) or {}
        blocks.append(
            Block(
                id=f"{doc_id}-block-{idx}",
                type=BlockType.PARAGRAPH,
                text=text,
                metadata=dict(metadata),
            )
        )

    sections = [Section(id=f"{doc_id}-section-0", title=None, blocks=blocks)]
    return Document(id=doc_id, source=f"docling-{fmt}", sections=sections)


def parse_document(content: bytes, *, doc_id: str, file_format: str = "pdf") -> Document:
    """Parse raw bytes with Docling and convert to the internal document representation."""
    partitioned = partition(content=content, file_format=file_format)
    return partition_to_document(doc_id, file_format, partitioned)
