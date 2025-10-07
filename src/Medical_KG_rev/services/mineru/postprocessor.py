from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from .output_parser import ParsedBlock, ParsedDocument
from .types import Block, Document, MineruRequest

logger = structlog.get_logger(__name__)


class MineruPostProcessor:
    """Transforms parsed MinerU output into service-level structures."""

    def __init__(self, *, preserve_layout: bool = True) -> None:
        self._preserve_layout = preserve_layout

    def build_document(
        self,
        parsed: ParsedDocument,
        request: MineruRequest,
        provenance: dict[str, Any],
    ) -> Document:
        table_index = {table.id: table for table in parsed.tables}
        figure_index = {figure.id: figure for figure in parsed.figures}
        equation_index = {equation.id: equation for equation in parsed.equations}
        blocks: list[Block] = []
        for raw in parsed.blocks:
            blocks.append(self._build_block(raw, table_index, figure_index, equation_index))
        metadata = dict(parsed.metadata)
        metadata.setdefault("source", "mineru-cli")
        document = Document(
            document_id=request.document_id,
            tenant_id=request.tenant_id,
            blocks=blocks,
            tables=list(parsed.tables),
            figures=list(parsed.figures),
            equations=list(parsed.equations),
            metadata=metadata,
            provenance=self._build_provenance(parsed, provenance),
        )
        logger.info(
            "mineru.postprocess.completed",
            document_id=request.document_id,
            blocks=len(blocks),
            tables=len(parsed.tables),
            figures=len(parsed.figures),
            equations=len(parsed.equations),
        )
        return document

    def _build_block(
        self,
        block: ParsedBlock,
        tables: dict[str, Any],
        figures: dict[str, Any],
        equations: dict[str, Any],
    ) -> Block:
        attached_table = tables.get(block.table_id) if block.table_id else None
        attached_figure = figures.get(block.figure_id) if block.figure_id else None
        attached_equation = equations.get(block.equation_id) if block.equation_id else None
        return Block(
            id=block.id,
            page=block.page,
            kind=block.type,
            text=block.text,
            bbox=block.bbox,
            confidence=block.confidence,
            reading_order=block.reading_order,
            metadata=block.metadata,
            table=attached_table,
            figure=attached_figure,
            equation=attached_equation,
        )

    def _build_provenance(
        self, parsed: ParsedDocument, provenance: dict[str, Any]
    ) -> dict[str, Any]:
        result = dict(provenance)
        result.setdefault("document_id", parsed.document_id)
        result.setdefault("processed_at", datetime.now(timezone.utc).isoformat())
        return result


__all__ = ["MineruPostProcessor"]
