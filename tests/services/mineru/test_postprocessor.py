from __future__ import annotations

from pathlib import Path

import pytest

from Medical_KG_rev.models.figure import Figure
from Medical_KG_rev.models.table import Table, TableCell
from Medical_KG_rev.services.mineru.output_parser import ParsedBlock, ParsedDocument
from Medical_KG_rev.services.mineru.postprocessor import MineruPostProcessor
from Medical_KG_rev.services.mineru.types import MineruRequest
from Medical_KG_rev.storage.object_store import FigureStorageClient, InMemoryObjectStore


pytest.importorskip("pydantic")


def test_postprocessor_enriches_outputs(tmp_path: Path):
    table = Table(
        id="tbl-1",
        page=1,
        headers=("A", "B"),
        cells=(
            TableCell(row=0, column=0, content="A"),
            TableCell(row=0, column=1, content="B"),
        ),
    )
    figure_path = tmp_path / "fig-1.png"
    figure_path.write_bytes(b"fakepng")
    figure = Figure(id="fig-1", page=1, image_path=str(figure_path), mime_type="image/png")
    parsed = ParsedDocument(
        document_id="doc-1",
        blocks=[
            ParsedBlock(
                id="blk-1",
                page=1,
                type="table",
                text="A|B",
                bbox=(0.0, 0.0, 1.0, 1.0),
                confidence=0.9,
                reading_order=1,
                metadata={},
                table_id="tbl-1",
            ),
            ParsedBlock(
                id="blk-2",
                page=1,
                type="figure",
                text=None,
                bbox=(0.1, 0.1, 0.9, 0.9),
                confidence=0.8,
                reading_order=2,
                metadata={},
                figure_id="fig-1",
            ),
        ],
        tables=[table],
        figures=[figure],
        equations=[],
        metadata={},
    )
    storage = FigureStorageClient(InMemoryObjectStore())
    processor = MineruPostProcessor(figure_storage=storage)
    request = MineruRequest(tenant_id="tenant", document_id="doc-1", content=b"")
    document = processor.build_document(parsed, request, provenance={})

    assert document.blocks[0].table is not None
    assert document.blocks[0].ir_block is not None
    assert "table_csv" in document.blocks[0].metadata
    assert document.metadata["source"] == "mineru-cli"
    table_exports = document.metadata["table_exports"]["tbl-1"]
    assert "csv" in table_exports and "json" in table_exports
    figure_meta = document.metadata["figure_assets"]["fig-1"]
    assert figure_meta["storage_key"].endswith("fig-1.png")
    assert document.ir_document is not None
