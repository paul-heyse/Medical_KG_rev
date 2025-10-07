from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from Medical_KG_rev.services.mineru.output_parser import ParsedBlock, ParsedDocument
from Medical_KG_rev.services.mineru.postprocessor import MineruPostProcessor
from Medical_KG_rev.services.mineru.types import MineruRequest


def test_postprocessor_links_tables_to_blocks():
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
            )
        ],
        tables=[
            type("TableStub", (), {"id": "tbl-1", "headers": ("A", "B")})()
        ],
        figures=[],
        equations=[],
        metadata={},
    )
    processor = MineruPostProcessor()
    request = MineruRequest(tenant_id="tenant", document_id="doc-1", content=b"")
    document = processor.build_document(parsed, request, provenance={})
    assert document.blocks[0].table is not None
    assert document.blocks[0].table.headers == ("A", "B")
    assert document.metadata["source"] == "mineru-cli"
