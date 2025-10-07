from __future__ import annotations

import csv
import json
import mimetypes
from collections import defaultdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

import structlog

from Medical_KG_rev.models.equation import Equation
from Medical_KG_rev.models.figure import Figure
from Medical_KG_rev.models.ir import Block as IrBlock
from Medical_KG_rev.models.ir import BlockType, Document as IrDocument, Section
from Medical_KG_rev.models.table import Table
from Medical_KG_rev.storage.object_store import FigureStorageClient

from .artifacts import MineruArtifacts, build_artifacts
from .output_parser import ParsedBlock, ParsedDocument
from .types import Block, Document, MineruRequest

logger = structlog.get_logger(__name__)


class MineruPostProcessor:
    """Transforms parsed MinerU output into service-level structures."""

    def __init__(
        self,
        *,
        preserve_layout: bool = True,
        figure_storage: FigureStorageClient | None = None,
        inline_equation_character_limit: int = 120,
    ) -> None:
        self._preserve_layout = preserve_layout
        self._figure_storage = figure_storage
        self._inline_equation_character_limit = max(40, inline_equation_character_limit)

    def build_document(
        self,
        parsed: ParsedDocument,
        request: MineruRequest,
        provenance: dict[str, Any],
    ) -> Document:
        artifacts = self._prepare_artifacts(parsed, request)

        blocks: list[Block] = []
        ir_blocks: list[IrBlock] = []
        for raw in parsed.blocks:
            block = self._build_block(raw, artifacts)
            blocks.append(block)
            if block.ir_block is not None:
                ir_blocks.append(block.ir_block)

        metadata = dict(parsed.metadata)
        metadata.setdefault("source", "mineru-cli")
        for key, value in artifacts.metadata_payload().items():
            metadata.setdefault(key, value)

        ir_document = self._build_ir_document(
            request=request,
            parsed=parsed,
            ir_blocks=ir_blocks,
            metadata=metadata,
        )

        document = Document(
            document_id=request.document_id,
            tenant_id=request.tenant_id,
            blocks=blocks,
            tables=list(artifacts.tables),
            figures=list(artifacts.figures),
            equations=list(artifacts.equations),
            metadata=metadata,
            provenance=self._build_provenance(parsed, provenance),
            ir_document=ir_document,
        )
        logger.info(
            "mineru.postprocess.completed",
            document_id=request.document_id,
            blocks=len(blocks),
            tables=len(artifacts.tables),
            figures=len(artifacts.figures),
            equations=len(artifacts.equations),
        )
        return document

    def _prepare_artifacts(self, parsed: ParsedDocument, request: MineruRequest) -> MineruArtifacts:
        tables, table_exports = self._prepare_tables(parsed.tables)
        figures, figure_assets = self._prepare_figures(
            request.tenant_id, request.document_id, parsed.figures
        )
        equations, equation_rendering = self._prepare_equations(parsed.equations)
        return build_artifacts(
            tables=tables,
            figures=figures,
            equations=equations,
            table_exports=table_exports,
            figure_assets=figure_assets,
            equation_rendering=equation_rendering,
        )

    def _prepare_tables(
        self, tables: Iterable[Table]
    ) -> tuple[list[Table], dict[str, dict[str, str]]]:
        prepared: list[Table] = []
        exports: dict[str, dict[str, str]] = {}
        for table in tables:
            serialisations = self._render_table_serialisations(table)
            metadata_updates: dict[str, Any] = {}
            if "headers" not in table.metadata:
                metadata_updates["headers"] = list(table.headers)
            if "exports" not in table.metadata:
                metadata_updates["exports"] = serialisations
            enriched = table.enrich(metadata=metadata_updates) if metadata_updates else table
            prepared.append(enriched)
            exports[table.id] = serialisations
        return prepared, exports

    def _prepare_figures(
        self,
        tenant_id: str,
        document_id: str,
        figures: Iterable[Figure],
    ) -> tuple[list[Figure], dict[str, dict[str, str]]]:
        prepared: list[Figure] = []
        assets: dict[str, dict[str, str]] = {}
        for figure in figures:
            storage_info: dict[str, str] = {}
            if self._figure_storage:
                stored = self._store_figure(tenant_id, document_id, figure)
                if stored:
                    storage_info.update(stored)
            updates: dict[str, Any] = {}
            if storage_info.get("url"):
                updates["image_path"] = storage_info["url"]
            figure_copy = (
                figure.enrich(metadata=storage_info, **updates)
                if storage_info or updates
                else figure
            )
            prepared.append(figure_copy)
            if storage_info:
                assets[figure.id] = storage_info
        return prepared, assets

    def _prepare_equations(
        self, equations: Iterable[Equation]
    ) -> tuple[list[Equation], dict[str, str]]:
        prepared: list[Equation] = []
        rendering: dict[str, str] = {}
        for equation in equations:
            render_mode = self._infer_equation_render_mode(equation)
            metadata_updates: dict[str, Any] = {}
            if "render_mode" not in equation.metadata:
                metadata_updates["render_mode"] = render_mode
            rendering[equation.id] = render_mode
            enriched = (
                equation.enrich(metadata=metadata_updates)
                if metadata_updates
                else equation
            )
            prepared.append(enriched)
        return prepared, rendering

    def _render_table_serialisations(self, table: Table) -> dict[str, str]:
        cells_payload = [cell.model_dump() for cell in table.cells]
        payload = {
            "id": table.id,
            "headers": list(table.headers),
            "caption": table.caption,
            "cells": cells_payload,
        }
        json_blob = json.dumps(payload, ensure_ascii=False)
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        headers = list(table.headers) or [f"column_{idx}" for idx in range(self._infer_column_count(table))]
        writer.writerow(headers)
        grid = self._build_table_grid(table)
        for row in grid:
            writer.writerow(row)
        csv_blob = csv_buffer.getvalue()
        markdown = table.to_markdown()
        return {"json": json_blob, "csv": csv_blob, "markdown": markdown}

    def _infer_column_count(self, table: Table) -> int:
        if table.headers:
            return len(table.headers)
        return max((cell.column + cell.colspan for cell in table.cells), default=0)

    def _build_table_grid(self, table: Table) -> list[list[str]]:
        rows = max((cell.row + cell.rowspan for cell in table.cells), default=0)
        header_count = len(table.headers)
        cols = max((cell.column + cell.colspan for cell in table.cells), default=header_count)
        if cols == 0:
            cols = header_count or 0
        grid: list[list[str]] = [["" for _ in range(cols)] for _ in range(rows)]
        for cell in table.cells:
            for row_offset in range(cell.rowspan):
                for col_offset in range(cell.colspan):
                    grid[cell.row + row_offset][cell.column + col_offset] = cell.content
        return grid

    def _store_figure(
        self,
        tenant_id: str,
        document_id: str,
        figure: Figure,
    ) -> dict[str, str] | None:
        path = Path(figure.image_path)
        if not path.exists():
            logger.warning(
                "mineru.postprocess.figure_missing",
                figure_id=figure.id,
                path=str(path),
            )
            return None
        content_type = figure.mime_type or mimetypes.guess_type(path.name)[0] or "image/png"
        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.bind(error=str(exc), figure_id=figure.id).warning(
                "mineru.postprocess.figure_read_failed"
            )
            return None
        key = self._figure_storage.store_figure(
            tenant_id,
            document_id,
            figure.id,
            data,
            content_type=content_type,
            metadata={"source": "mineru-cli"},
        )
        url = self._figure_storage.generate_figure_url(
            tenant_id,
            document_id,
            figure.id,
            key=key,
        )
        return {
            "storage_key": key,
            "url": url,
            "content_type": content_type,
        }

    def _infer_equation_render_mode(self, equation) -> str:
        if not getattr(equation, "latex", ""):
            return "link"
        if len(equation.latex) <= self._inline_equation_character_limit and getattr(equation, "mathml", None) is None:
            return "inline"
        return "link"

    def _build_block(self, block: ParsedBlock, artifacts: MineruArtifacts) -> Block:
        attached_table, attached_figure, attached_equation = artifacts.attachments_for(
            block.table_id, block.figure_id, block.equation_id
        )
        metadata = dict(block.metadata)
        metadata.setdefault("page", block.page)
        metadata.setdefault("reading_order", block.reading_order)
        if attached_table:
            metadata.setdefault("table_id", attached_table.id)
            metadata.setdefault("is_table", True)
            metadata.setdefault("table_caption", attached_table.caption)
            exports = artifacts.table_exports.get(attached_table.id, {})
            if exports:
                metadata.setdefault("table_markdown", exports.get("markdown"))
                metadata.setdefault("table_csv", exports.get("csv"))
        if attached_figure:
            metadata.setdefault("figure_id", attached_figure.id)
            assets = artifacts.figure_assets.get(attached_figure.id, {})
            if assets:
                metadata.setdefault("figure_url", assets.get("url"))
        if attached_equation:
            metadata.setdefault("equation_id", attached_equation.id)
            render_mode = artifacts.equation_rendering.get(attached_equation.id)
            if render_mode:
                metadata.setdefault("equation_render_mode", render_mode)

        ir_block = IrBlock(
            id=block.id,
            type=self._resolve_block_type(block.type, attached_table, attached_figure, attached_equation),
            text=block.text,
            metadata=metadata,
            layout_bbox=block.bbox,
            reading_order=block.reading_order,
            confidence_score=block.confidence,
            table=attached_table,
            figure=attached_figure,
            equation=attached_equation,
        )

        return Block(
            id=block.id,
            page=block.page,
            kind=block.type,
            text=block.text,
            bbox=block.bbox,
            confidence=block.confidence,
            reading_order=block.reading_order,
            metadata=metadata,
            table=attached_table,
            figure=attached_figure,
            equation=attached_equation,
            ir_block=ir_block,
        )

    def _resolve_block_type(
        self,
        raw_type: str,
        table: Any,
        figure: Any,
        equation: Any,
    ) -> BlockType:
        if table:
            return BlockType.TABLE
        if figure:
            return BlockType.FIGURE
        if equation:
            return BlockType.EQUATION
        try:
            return BlockType(raw_type)
        except Exception:
            return BlockType.PARAGRAPH

    def _build_ir_document(
        self,
        *,
        request: MineruRequest,
        parsed: ParsedDocument,
        ir_blocks: list[IrBlock],
        metadata: dict[str, Any],
    ) -> IrDocument:
        document_id = request.document_id
        tenant_id = request.tenant_id
        sections_by_page: dict[int, list[IrBlock]] = defaultdict(list)
        for raw_block, ir_block in zip(parsed.blocks, ir_blocks):
            sections_by_page[raw_block.page].append(ir_block)
        sections = [
            Section(
                id=f"{document_id}:page:{page}",
                title=f"Page {page}",
                blocks=tuple(page_blocks),
            )
            for page, page_blocks in sorted(sections_by_page.items())
        ]
        doc_metadata = dict(metadata)
        doc_metadata.setdefault("tenant_id", tenant_id)
        return IrDocument(
            id=document_id,
            source=str(metadata.get("source", "mineru")),
            title=metadata.get("title"),
            sections=tuple(sections),
            metadata=doc_metadata,
        )

    def _build_provenance(
        self, parsed: ParsedDocument, provenance: dict[str, Any]
    ) -> dict[str, Any]:
        result = dict(provenance)
        result.setdefault("document_id", parsed.document_id)
        result.setdefault("processed_at", datetime.now(timezone.utc).isoformat())
        return result
__all__ = ["MineruPostProcessor"]
