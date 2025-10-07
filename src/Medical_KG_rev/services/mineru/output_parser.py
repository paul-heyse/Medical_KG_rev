from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from Medical_KG_rev.models.equation import Equation
from Medical_KG_rev.models.figure import Figure
from Medical_KG_rev.models.table import Table, TableCell

logger = structlog.get_logger(__name__)


class MineruOutputParserError(RuntimeError):
    """Raised when MinerU CLI output cannot be parsed."""


@dataclass(slots=True)
class ParsedBlock:
    id: str
    page: int
    type: str
    text: str | None
    bbox: tuple[float, float, float, float] | None
    confidence: float | None
    reading_order: int | None
    metadata: dict[str, Any]
    table_id: str | None = None
    figure_id: str | None = None
    equation_id: str | None = None


@dataclass(slots=True)
class ParsedDocument:
    document_id: str
    blocks: list[ParsedBlock]
    tables: list[Table]
    figures: list[Figure]
    equations: list[Equation]
    metadata: dict[str, Any]


class MineruOutputParser:
    """Parses structured JSON output emitted by MinerU CLI."""

    def parse_path(self, path: Path) -> ParsedDocument:
        if not path.exists():
            raise MineruOutputParserError(f"MinerU output file not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MineruOutputParserError(f"Invalid MinerU JSON output: {exc}") from exc
        return self.parse_dict(payload)

    def parse_dict(self, payload: dict[str, Any]) -> ParsedDocument:
        document_id = str(payload.get("document_id", ""))
        if not document_id:
            raise MineruOutputParserError("MinerU output missing 'document_id'")
        blocks = [self._parse_block(entry) for entry in payload.get("blocks", [])]
        tables = [self._parse_table(entry) for entry in payload.get("tables", [])]
        figures = [self._parse_figure(entry) for entry in payload.get("figures", [])]
        equations = [self._parse_equation(entry) for entry in payload.get("equations", [])]
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"value": metadata}
        parsed = ParsedDocument(
            document_id=document_id,
            blocks=blocks,
            tables=tables,
            figures=figures,
            equations=equations,
            metadata=metadata,
        )
        logger.debug(
            "mineru.output.parsed",
            document_id=document_id,
            blocks=len(blocks),
            tables=len(tables),
            figures=len(figures),
            equations=len(equations),
        )
        return parsed

    def parse_markdown(self, document_id: str, markdown: str) -> ParsedDocument:
        blocks: list[ParsedBlock] = []
        for idx, line in enumerate(markdown.splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            blocks.append(
                ParsedBlock(
                    id=f"blk-md-{idx}",
                    page=1,
                    type="paragraph",
                    text=text,
                    bbox=None,
                    confidence=1.0,
                    reading_order=idx,
                    metadata={"source": "markdown"},
                )
            )
        return ParsedDocument(
            document_id=document_id,
            blocks=blocks,
            tables=[],
            figures=[],
            equations=[],
            metadata={"format": "markdown"},
        )

    def _parse_block(self, payload: dict[str, Any]) -> ParsedBlock:
        bbox_value = payload.get("bbox")
        bbox = tuple(float(value) for value in bbox_value) if bbox_value else None
        cleaned_bbox = bbox if bbox and len(bbox) == 4 else None
        return ParsedBlock(
            id=str(payload.get("id")),
            page=int(payload.get("page", 1)),
            type=str(payload.get("type", "paragraph")),
            text=payload.get("text"),
            bbox=cleaned_bbox,
            confidence=self._maybe_float(payload.get("confidence")),
            reading_order=self._maybe_int(payload.get("reading_order")),
            metadata=self._ensure_dict(payload.get("metadata", {})),
            table_id=payload.get("table_id"),
            figure_id=payload.get("figure_id"),
            equation_id=payload.get("equation_id"),
        )

    def _parse_table(self, payload: dict[str, Any]) -> Table:
        cells = tuple(self._parse_table_cell(cell) for cell in payload.get("cells", []))
        headers = tuple(str(value) for value in payload.get("headers", []))
        bbox_value = payload.get("bbox")
        bbox = tuple(float(value) for value in bbox_value) if bbox_value else None
        return Table(
            id=str(payload.get("id")),
            page=int(payload.get("page", 1)),
            caption=payload.get("caption"),
            cells=cells,
            headers=headers,
            bbox=bbox if bbox and len(bbox) == 4 else None,
            metadata=self._ensure_dict(payload.get("metadata", {})),
        )

    def _parse_table_cell(self, payload: dict[str, Any]) -> TableCell:
        bbox_value = payload.get("bbox")
        bbox = tuple(float(value) for value in bbox_value) if bbox_value else None
        return TableCell(
            row=int(payload.get("row", 0)),
            column=int(payload.get("column", 0)),
            content=str(payload.get("content", "")),
            rowspan=int(payload.get("rowspan", 1)),
            colspan=int(payload.get("colspan", 1)),
            bbox=bbox if bbox and len(bbox) == 4 else None,
            confidence=self._maybe_float(payload.get("confidence")),
        )

    def _parse_figure(self, payload: dict[str, Any]) -> Figure:
        bbox_value = payload.get("bbox")
        bbox = tuple(float(value) for value in bbox_value) if bbox_value else None
        return Figure(
            id=str(payload.get("id")),
            page=int(payload.get("page", 1)),
            image_path=str(payload.get("image_path", "")),
            caption=payload.get("caption"),
            bbox=bbox if bbox and len(bbox) == 4 else None,
            figure_type=payload.get("figure_type"),
            mime_type=payload.get("mime_type"),
            width=self._maybe_int(payload.get("width")),
            height=self._maybe_int(payload.get("height")),
            metadata=self._ensure_dict(payload.get("metadata", {})),
        )

    def _parse_equation(self, payload: dict[str, Any]) -> Equation:
        bbox_value = payload.get("bbox")
        bbox = tuple(float(value) for value in bbox_value) if bbox_value else None
        return Equation(
            id=str(payload.get("id")),
            page=int(payload.get("page", 1)),
            latex=str(payload.get("latex", "")),
            mathml=payload.get("mathml"),
            bbox=bbox if bbox and len(bbox) == 4 else None,
            display=bool(payload.get("display", True)),
            metadata=self._ensure_dict(payload.get("metadata", {})),
        )

    def _ensure_dict(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {"value": value}

    def _maybe_int(self, value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _maybe_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


__all__ = [
    "MineruOutputParser",
    "MineruOutputParserError",
    "ParsedBlock",
    "ParsedDocument",
]
