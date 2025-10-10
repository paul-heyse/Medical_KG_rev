from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from Medical_KG_rev.models.equation import Equation
from Medical_KG_rev.models.figure import Figure
from Medical_KG_rev.models.table import Table, TableCell
from Medical_KG_rev.services.mineru.output_parser import ParsedBlock, ParsedDocument
from Medical_KG_rev.services.mineru.postprocessor import MineruPostProcessor
from Medical_KG_rev.services.mineru.types import MineruRequest


class _FigureStorageStub:
    def __init__(self) -> None:
        self.keys: list[str] = []
        self.urls: list[str] = []

    def store_figure(
        self,
        tenant_id: str,
        document_id: str,
        figure_id: str,
        data: bytes,
        *,
        content_type: str,
        metadata,
    ):
        key = f"mineru/{tenant_id}/{document_id}/{figure_id}.png"
        self.keys.append(key)
        return key

    def generate_figure_url(self, tenant_id: str, document_id: str, figure_id: str, *, key: str):
        url = f"https://assets.invalid/{tenant_id}/{document_id}/{figure_id}"
        self.urls.append(url)
        return url


@pytest.fixture
def mineru_document(tmp_path):
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
    figure_path = tmp_path / "figure.png"
    figure_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    figure = Figure(
        id="fig-1",
        page=1,
        image_path=str(figure_path),
        caption="Microscopy image",
        mime_type="image/png",
    )
    equation = Equation(id="eq-1", page=1, latex="E=mc^2", mathml=None, display=True)
    blocks = [
        ParsedBlock(
            id="blk-1",
            page=1,
            type="paragraph",
            text="Study overview",
            bbox=(0.1, 0.1, 0.9, 0.2),
            confidence=0.9,
            reading_order=1,
            metadata={},
        ),
        ParsedBlock(
            id="blk-2",
            page=1,
            type="table",
            text="Dose | Response | Notes",
            bbox=(0.1, 0.25, 0.9, 0.4),
            confidence=0.95,
            reading_order=2,
            metadata={},
            table_id="tbl-1",
        ),
        ParsedBlock(
            id="blk-3",
            page=1,
            type="figure",
            text="",
            bbox=(0.1, 0.45, 0.9, 0.75),
            confidence=0.92,
            reading_order=3,
            metadata={},
            figure_id="fig-1",
        ),
        ParsedBlock(
            id="blk-4",
            page=1,
            type="equation",
            text="E=mc^2",
            bbox=(0.1, 0.8, 0.6, 0.9),
            confidence=0.9,
            reading_order=4,
            metadata={},
            equation_id="eq-1",
        ),
    ]
    parsed = ParsedDocument(
        document_id="doc-quality",
        blocks=blocks,
        tables=[table],
        figures=[figure],
        equations=[equation],
        metadata={"layout_model": "simulated", "vision_model": "simulated"},
    )
    storage = _FigureStorageStub()
    processor = MineruPostProcessor(figure_storage=storage)
    document = processor.build_document(
        parsed, MineruRequest("tenant", "doc-quality", b"raw"), {"source": "quality-test"}
    )
    return document, storage


def test_mineru_tables_exceed_stub(mineru_document):
    document, _ = mineru_document
    legacy_tables = 0  # legacy stub had no structured tables
    assert len(document.tables) > legacy_tables
    assert document.tables[0].metadata["headers"] == ["Dose", "Response", "Notes"]


def test_table_structure_preserved(mineru_document):
    document, _ = mineru_document
    table = document.tables[0]
    assert table.dimensions() == (1, 3)
    assert "Dose" in table.metadata["exports"]["markdown"]


def test_figure_metadata_complete(mineru_document):
    document, storage = mineru_document
    assert storage.keys, "figure storage should be invoked"
    figure = document.figures[0]
    assert "url" in figure.metadata
    assert figure.metadata["url"].startswith("https://assets.invalid")


def test_reading_order_monotonic(mineru_document):
    document, _ = mineru_document
    orders = [block.metadata["reading_order"] for block in document.blocks]
    assert orders == sorted(orders), "blocks should retain ascending reading order"
