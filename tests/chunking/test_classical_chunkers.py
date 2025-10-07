import pytest

from Medical_KG_rev.chunking.chunkers import C99Chunker, TextTilingChunker
from Medical_KG_rev.models.ir import Block, Document, Section


def _document() -> Document:
    blocks = [
        Block(id=f"b-{idx}", text=f"Sentence {idx} about study outcomes and methods.")
        for idx in range(1, 8)
    ]
    section = Section(id="s-1", title="Study", blocks=blocks)
    return Document(id="doc-classical", source="pmc", title="Classical", sections=[section])


def test_text_tiling_chunker_handles_document() -> None:
    nltk = pytest.importorskip("nltk")
    nltk.download("punkt", quiet=True)
    chunker = TextTilingChunker(w=10, k=5)
    chunks = chunker.chunk(_document(), tenant_id="tenant", granularity="paragraph")
    assert chunks


def test_c99_chunker_builds_similarity_matrix() -> None:
    pytest.importorskip("sklearn")
    chunker = C99Chunker()
    chunks = chunker.chunk(_document(), tenant_id="tenant", granularity="paragraph")
    assert chunks
