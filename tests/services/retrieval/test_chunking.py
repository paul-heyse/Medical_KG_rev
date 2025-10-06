from __future__ import annotations

from Medical_KG_rev.services.retrieval.chunking import ChunkingOptions, ChunkingService


def _sample_document() -> str:
    return (
        "Introduction\nThis is the introduction.\n\n"
        "Methods\nFirst paragraph.\n\n"
        "Results\nTable Header|Column\nRow|Value\n\n"
        "Conclusion\nFinal thoughts."
    )


def test_section_chunking_preserves_headers():
    service = ChunkingService()
    chunks = service.chunk("doc-1", _sample_document(), ChunkingOptions(strategy="section", max_tokens=50))
    assert any("Introduction" in chunk.text for chunk in chunks)
    assert any("Methods" in chunk.text for chunk in chunks)


def test_paragraph_chunking_respects_boundaries():
    service = ChunkingService()
    chunks = service.chunk("doc-1", _sample_document(), ChunkingOptions(strategy="paragraph", max_tokens=20))
    assert all("\n\n" not in chunk.text for chunk in chunks)


def test_table_chunking_keeps_table_intact():
    service = ChunkingService()
    chunks = service.chunk("doc-1", _sample_document(), ChunkingOptions(strategy="table", max_tokens=50))
    table_chunks = [chunk for chunk in chunks if "|" in chunk.text]
    assert len(table_chunks) == 1
    assert "Row|Value" in table_chunks[0].text


def test_sliding_window_overlap():
    service = ChunkingService()
    text = " ".join(f"token{i}" for i in range(50))
    chunks = service.sliding_window("doc-1", text, max_tokens=10, overlap=0.5)
    assert len(chunks) > 1
    first = chunks[0].metadata["end_token"]
    second = chunks[1].metadata["start_token"]
    assert second < first
