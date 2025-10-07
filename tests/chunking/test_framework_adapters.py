import pytest

from Medical_KG_rev.chunking.adapters.langchain import LangChainSplitterChunker
from Medical_KG_rev.chunking.adapters.unstructured_adapter import UnstructuredChunker
from Medical_KG_rev.models.ir import Block, Document, Section


def _document() -> Document:
    blocks = [
        Block(id="b-1", text="Heading\n\nThis is a body of text."),
        Block(id="b-2", text="Another paragraph with additional insight."),
    ]
    return Document(
        id="doc-adapter",
        source="pmc",
        title="Adapter",
        sections=[Section(id="s-1", title="Adapter", blocks=blocks)],
    )


def test_langchain_splitter_chunker_runs() -> None:
    pytest.importorskip("langchain")
    chunker = LangChainSplitterChunker(chunk_size=50, chunk_overlap=0)
    chunks = chunker.chunk(_document(), tenant_id="tenant", granularity="paragraph")
    assert chunks


def test_unstructured_chunker_handles_document() -> None:
    pytest.importorskip("unstructured")
    chunker = UnstructuredChunker(mode="chunk_by_title")
    chunks = chunker.chunk(_document(), tenant_id="tenant", granularity="section")
    assert isinstance(chunks, list)
