"""Regression tests comparing MinerU archive output with Docling payloads."""

from __future__ import annotations

from Medical_KG_rev.gateway.models import DoclingProcessingPayload


def test_docling_payload_preserves_core_fields_from_mineru_archive() -> None:
    """Ensure Docling payload retains compatibility with MinerU document metadata."""

    mineru_reference = {
        "document_id": "doc-legacy",
        "tables": 2,
    }
    payload = DoclingProcessingPayload(
        document_id="doc-legacy",
        text="Sample text",
        tables=[{"id": "t1"}, {"id": "t2"}],
        figures=[],
        metadata={"source": "docling"},
    )
    assert payload.document_id == mineru_reference["document_id"]
    assert len(payload.tables) >= mineru_reference["tables"]
    assert payload.metadata["source"] == "docling"
