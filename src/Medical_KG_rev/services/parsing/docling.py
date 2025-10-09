"""Docling wrapper with explicit PDF guard."""

from __future__ import annotations

from typing import Any

from Medical_KG_rev.models.ir import Document


class DoclingParser:
    """Thin adapter around docling partitioners that enforces PDF guard rails."""

    SUPPORTED_FORMATS = {"html", "xml", "text"}

    def __init__(self) -> None:
        try:
            from docling import partition
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("docling is not installed") from exc
        self._partition = partition

    def parse(self, *, content: bytes, fmt: str, doc_id: str) -> Document:
        fmt_normalized = fmt.lower()
        if fmt_normalized == "pdf":
            raise ValueError(
                "Docling cannot be used for PDF parsing in production. Use MinerU for PDF OCR (GPU-only policy)."
            )
        if fmt_normalized not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{fmt}'. Allowed formats: {sorted(self.SUPPORTED_FORMATS)}"
            )
        partitioned = self._partition(content=content, format=fmt_normalized)
        return _map_to_ir(doc_id=doc_id, partitioned=partitioned, fmt=fmt_normalized)


def _map_to_ir(*, doc_id: str, partitioned: Any, fmt: str) -> Document:
    from Medical_KG_rev.models.ir import Block, BlockType, Document, Section

    sections = []
    blocks = []
    for idx, element in enumerate(partitioned):
        text = getattr(element, "text", "") or ""
        metadata = getattr(element, "metadata", {})
        block = Block(
            id=f"{doc_id}-block-{idx}",
            type=BlockType.PARAGRAPH,
            text=text,
            metadata=dict(metadata),
        )
        blocks.append(block)
    sections.append(Section(id=f"{doc_id}-section-0", title=None, blocks=blocks))
    return Document(id=doc_id, source=f"docling-{fmt}", sections=sections)
