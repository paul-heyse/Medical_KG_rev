from Medical_KG_rev.chunking.adapters.table_aware import TableAwareChunker
from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.models.table import Table, TableCell


class DummyCounter:
    def count(self, text: str) -> int:
        return len(text.split())


def test_table_aware_chunker_creates_table_chunks():
    table = Table(
        id="tbl-1",
        page=1,
        caption="Example",
        headers=("A", "B"),
        cells=(
            TableCell(row=0, column=0, content="1"),
            TableCell(row=0, column=1, content="2"),
        ),
        metadata={"exports": {"markdown": "|A|B|", "csv": "A,B", "json": "{}"}},
    )
    table_block = Block(
        id="blk-table",
        type=BlockType.TABLE,
        text="A|B",
        metadata={"is_table": True},
        layout_bbox=(0.0, 0.0, 1.0, 1.0),
        table=table,
    )
    text_block = Block(
        id="blk-text",
        type=BlockType.PARAGRAPH,
        text="Intro paragraph",
        metadata={},
    )
    section = Section(id="sec-1", blocks=[text_block, table_block])
    document = Document(id="doc-1", source="test", sections=[section])

    chunker = TableAwareChunker(max_paragraphs=1, token_counter=DummyCounter())
    chunks = chunker.chunk(document, tenant_id="tenant")

    assert len(chunks) == 2
    table_chunk = next(chunk for chunk in chunks if chunk.meta.get("segment_type") == "table")
    assert table_chunk.meta["table_id"] == "tbl-1"
    assert table_chunk.meta["table_markdown"] == "|A|B|"
