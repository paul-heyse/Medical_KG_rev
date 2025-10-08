from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.services.chunking.wrappers.llamaindex_parser import (
    LlamaIndexChunker,
)


def _document() -> Document:
    section = Section(
        id="sec-1",
        title="Introduction",
        blocks=[
            Block(
                id="b1",
                type=BlockType.PARAGRAPH,
                text="Sentence one. Sentence two. Sentence three.",
                metadata={},
            )
        ],
    )
    return Document(id="doc-1", source="unit-test", sections=[section])


def test_llamaindex_fallback_chunks():
    profile = {
        "name": "fallback",
        "respect_boundaries": ["section"],
        "metadata": {"window_size": 2},
        "sentence_splitter": "simple",
    }
    chunker = LlamaIndexChunker(profile=profile)
    chunks = chunker.chunk(_document(), profile="fallback")
    assert len(chunks) == 3
    assert all(chunk.section_label == "Introduction" for chunk in chunks)
    assert all(chunk.metadata["chunking_profile"] == "fallback" for chunk in chunks)
