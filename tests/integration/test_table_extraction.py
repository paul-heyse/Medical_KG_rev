from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from Medical_KG_rev.models.table import Table, TableCell
from Medical_KG_rev.services.mineru.output_parser import ParsedBlock, ParsedDocument
from Medical_KG_rev.services.mineru.postprocessor import MineruPostProcessor
from Medical_KG_rev.services.mineru.types import MineruRequest


def test_table_extraction_preserves_structure_and_exports():
    table = Table(
        id="tbl-1",
        page=1,
        headers=("Dose", "Response", "Notes"),
        cells=(
            TableCell(row=0, column=0, content="5mg"),
            TableCell(row=0, column=1, content="Positive"),
            TableCell(row=0, column=2, content="Stable"),
        ),
    )
    block = ParsedBlock(
        id="blk-1",
        page=1,
        type="table",
        text="Dose | Response | Notes",
        bbox=(0.1, 0.1, 0.9, 0.2),
        confidence=0.95,
        reading_order=1,
        metadata={},
        table_id="tbl-1",
    )
    parsed = ParsedDocument(
        document_id="doc-table",
        blocks=[block],
        tables=[table],
        figures=[],
        equations=[],
        metadata={"layout_model": "simulated"},
    )
    request = MineruRequest("tenant", "doc-table", b"raw")
    processor = MineruPostProcessor()

    document = processor.build_document(parsed, request, {"source": "integration-test"})

    assert document.tables, "expected table to be materialised"
    stored_table = document.tables[0]
    assert stored_table.metadata["exports"]["markdown"].startswith("Dose | Response | Notes")
    assert stored_table.dimensions() == (1, 3)
    assert document.blocks[0].table == stored_table
    assert "table_exports" in document.metadata
    assert "tbl-1" in document.metadata["table_exports"]
