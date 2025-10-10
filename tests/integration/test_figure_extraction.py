from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from Medical_KG_rev.models.figure import Figure
from Medical_KG_rev.services.mineru.output_parser import ParsedBlock, ParsedDocument
from Medical_KG_rev.services.mineru.postprocessor import MineruPostProcessor
from Medical_KG_rev.services.mineru.types import MineruRequest


class _RecordingFigureStorage:
    def __init__(self) -> None:
        self.stored: list[tuple[str, str, str, bytes]] = []
        self.generated: list[tuple[str, str, str]] = []

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
        self.stored.append((tenant_id, document_id, figure_id, data))
        return f"mineru/{tenant_id}/{document_id}/{figure_id}.png"

    def generate_figure_url(self, tenant_id: str, document_id: str, figure_id: str, *, key: str):
        self.generated.append((tenant_id, document_id, figure_id))
        return f"https://example.invalid/{tenant_id}/{document_id}/{figure_id}"


def test_figure_extraction_records_storage(tmp_path):
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    figure = Figure(
        id="fig-1",
        page=2,
        image_path=str(image_path),
        caption="Example figure",
        mime_type="image/png",
    )
    block = ParsedBlock(
        id="blk-fig",
        page=2,
        type="figure",
        text="",
        bbox=(0.1, 0.2, 0.9, 0.8),
        confidence=0.9,
        reading_order=5,
        metadata={},
        figure_id="fig-1",
    )
    parsed = ParsedDocument(
        document_id="doc-fig",
        blocks=[block],
        tables=[],
        figures=[figure],
        equations=[],
        metadata={"vision_model": "simulated"},
    )
    storage = _RecordingFigureStorage()
    processor = MineruPostProcessor(figure_storage=storage)

    document = processor.build_document(
        parsed, MineruRequest("tenant", "doc-fig", b"raw"), {"source": "integration-test"}
    )

    assert storage.stored, "figure bytes should be stored"
    assert storage.generated, "figure URLs should be generated"
    stored_figure = document.figures[0]
    assert stored_figure.metadata["storage_key"].endswith("fig-1.png")
    assert stored_figure.metadata["url"].endswith("fig-1")
    assert "fig-1" in document.metadata["figure_assets"]
