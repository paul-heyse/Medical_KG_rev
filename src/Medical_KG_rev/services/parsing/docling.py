"""Docling wrapper with explicit PDF guard."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from Medical_KG_rev.models.figure import Figure
from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.models.table import Table, TableCell
from Medical_KG_rev.services.parsing.docling_vlm_service import DoclingVLMResult


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


class DoclingVLMOutputParser:
    """Convert :class:`DoclingVLMResult` payloads into the Document IR."""

    def __init__(self, *, include_tables: bool = True, include_figures: bool = True) -> None:
        self._include_tables = include_tables
        self._include_figures = include_figures

    def parse(self, result: DoclingVLMResult) -> Document:
        """Map a Docling VLM result into a :class:`Document` instance."""

        blocks: list[Block] = []
        blocks.extend(self._text_blocks(result))

        if self._include_tables:
            blocks.extend(self._table_blocks(result))

        if self._include_figures:
            blocks.extend(self._figure_blocks(result))

        section = Section(
            id=f"{result.document_id}-section-0",
            title=result.metadata.get("title"),
            blocks=blocks,
        )
        metadata = dict(result.metadata)
        metadata.setdefault("backend", "docling_vlm")
        provenance = metadata.get("provenance", {})
        metadata.setdefault(
            "docling_vlm",
            {
                "model_name": provenance.get("model_name"),
                "processing_time_seconds": provenance.get("processing_time_seconds"),
                "tables": len(result.tables),
                "figures": len(result.figures),
                "text_length": len(result.text or ""),
            },
        )
        return Document(
            id=result.document_id,
            source="docling_vlm",
            title=result.metadata.get("title"),
            sections=[section],
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------
    def _text_blocks(self, result: DoclingVLMResult) -> list[Block]:
        paragraphs = list(self._iter_paragraphs(result.text))
        blocks: list[Block] = []
        for index, paragraph in enumerate(paragraphs):
            block = Block(
                id=f"{result.document_id}-text-{index}",
                type=BlockType.PARAGRAPH,
                text=paragraph,
                metadata={"source": "docling_vlm"},
            )
            blocks.append(block)
        return blocks

    def _table_blocks(self, result: DoclingVLMResult) -> list[Block]:
        blocks: list[Block] = []
        for index, payload in enumerate(result.tables):
            table = self._build_table(result.document_id, index, payload)
            block = Block(
                id=f"{result.document_id}-table-{index}",
                type=BlockType.TABLE,
                text=table.to_markdown(),
                table=table,
                metadata={"source": "docling_vlm", **table.metadata},
            )
            blocks.append(block)
        return blocks

    def _figure_blocks(self, result: DoclingVLMResult) -> list[Block]:
        blocks: list[Block] = []
        for index, payload in enumerate(result.figures):
            figure = self._build_figure(result.document_id, index, payload)
            block = Block(
                id=f"{result.document_id}-figure-{index}",
                type=BlockType.FIGURE,
                text=figure.caption,
                figure=figure,
                metadata={"source": "docling_vlm", **figure.metadata},
            )
            blocks.append(block)
        return blocks

    @staticmethod
    def _iter_paragraphs(text: str | None) -> Iterable[str]:
        if not text:
            return []
        normalised = text.replace("\r\n", "\n").replace("\r", "\n")
        segments = [segment.strip() for segment in normalised.split("\n\n")]
        return [segment for segment in segments if segment]

    @staticmethod
    def _build_table(doc_id: str, index: int, payload: Any) -> Table:
        metadata = DoclingVLMOutputParser._extract_metadata(payload, {"cells", "headers", "caption"})
        table_id = str(payload.get("id", f"{doc_id}-table-{index}"))
        page = int(payload.get("page", 1) or 1)
        headers = tuple(str(item) for item in payload.get("headers", []) if item is not None)
        cells = []
        for cell in payload.get("cells", []):
            cells.append(
                TableCell(
                    row=int(cell.get("row", 0) or 0),
                    column=int(cell.get("column", 0) or 0),
                    content=str(cell.get("text", "")),
                    rowspan=int(cell.get("rowspan", 1) or 1),
                    colspan=int(cell.get("colspan", 1) or 1),
                    bbox=DoclingVLMOutputParser._ensure_bbox(cell.get("bbox")),
                    confidence=(
                        float(cell.get("confidence"))
                        if cell.get("confidence") is not None
                        else None
                    ),
                )
            )
        return Table(
            id=table_id,
            page=page if page > 0 else 1,
            cells=tuple(cells),
            headers=headers,
            caption=str(payload.get("caption")) if payload.get("caption") is not None else None,
            metadata=metadata,
        )

    @staticmethod
    def _build_figure(doc_id: str, index: int, payload: Any) -> Figure:
        metadata = DoclingVLMOutputParser._extract_metadata(
            payload,
            {"image_path", "caption", "type", "mime_type", "width", "height"},
        )
        figure_id = str(payload.get("id", f"{doc_id}-figure-{index}"))
        page = int(payload.get("page", 1) or 1)
        image_path = str(
            payload.get("image_path")
            or payload.get("uri")
            or f"{doc_id}-figure-{index}.png"
        )
        return Figure(
            id=figure_id,
            page=page if page > 0 else 1,
            image_path=image_path,
            caption=(
                str(payload.get("caption")) if payload.get("caption") is not None else None
            ),
            figure_type=str(payload.get("type")) if payload.get("type") is not None else None,
            mime_type=str(payload.get("mime_type")) if payload.get("mime_type") is not None else None,
            width=int(payload.get("width")) if payload.get("width") is not None else None,
            height=int(payload.get("height")) if payload.get("height") is not None else None,
            metadata=metadata,
        )

    @staticmethod
    def _extract_metadata(payload: Any, excluded: set[str]) -> dict[str, Any]:
        if isinstance(payload, dict):
            return {k: v for k, v in payload.items() if k not in excluded}
        metadata: dict[str, Any] = {}
        for key in dir(payload):
            if key.startswith("_") or key in excluded:
                continue
            value = getattr(payload, key)
            metadata[key] = value
        return metadata

    @staticmethod
    def _ensure_bbox(value: Any) -> tuple[float, float, float, float] | None:
        if not value:
            return None
        try:
            x0, y0, x1, y1 = value
            return float(x0), float(y0), float(x1), float(y1)
        except Exception:  # pragma: no cover - defensive fallback
            return None


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
