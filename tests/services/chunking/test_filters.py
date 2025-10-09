from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.services.chunking.filters import apply_filter_chain


def _document_with_blocks(blocks: list[Block]) -> Document:
    section = Section(id="sec-1", title="Introduction", blocks=blocks)
    return Document(id="doc-1", source="unit-test", sections=[section])


def test_drop_boilerplate_removes_headers():
    header = Block(
        id="b1", type=BlockType.PARAGRAPH, text="Page 1 of 20", metadata={"role": "header"}
    )
    body = Block(id="b2", type=BlockType.PARAGRAPH, text="Valid content")
    document = _document_with_blocks([header, body])

    filtered = apply_filter_chain(document, ["drop_boilerplate"])

    assert len(filtered.sections[0].blocks) == 1
    assert filtered.sections[0].blocks[0].id == "b2"


def test_preserve_tables_html_marks_low_confidence():
    table = Block(
        id="tbl-1",
        type=BlockType.TABLE,
        text="<table></table>",
        metadata={"rectangularize_confidence": 0.5},
    )
    document = _document_with_blocks([table])

    filtered = apply_filter_chain(document, ["preserve_tables_html"])

    block = filtered.sections[0].blocks[0]
    assert block.metadata["is_unparsed_table"] is True


def test_deduplicate_page_furniture_removes_repeated_short_lines():
    block1 = Block(id="b1", type=BlockType.PARAGRAPH, text="Clinical Trial")
    block2 = Block(id="b2", type=BlockType.PARAGRAPH, text="Clinical Trial")
    block3 = Block(
        id="b3",
        type=BlockType.PARAGRAPH,
        text="Important content that should remain unique across the page.",
    )
    document = _document_with_blocks([block1, block2, block3])

    filtered = apply_filter_chain(document, ["deduplicate_page_furniture"])

    texts = [block.text for block in filtered.sections[0].blocks]
    assert "Clinical Trial" not in texts
    assert any("Important" in text for text in texts)
