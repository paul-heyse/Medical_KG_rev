import pytest

from Medical_KG_rev.models.ir import Block, BlockType, Document, Section, Span, Table, TableCell


def test_document_validates_spans():
    span = Span(start=0, end=4, text="test")
    block = Block(id="b1", text="test", spans=[span])
    section = Section(id="s1", title="Section", blocks=[block])
    document = Document(id="doc1", source="clinicaltrials", sections=[section])
    assert document.sections[0].blocks[0].spans[0].start == 0


def test_invalid_span_raises():
    span = Span(start=0, end=4)
    block = Block(id="b1", text="hi", spans=[span])
    section = Section(id="s1", blocks=[block])
    with pytest.raises(ValueError):
        Document(id="doc1", source="clinicaltrials", sections=[section])


def test_block_type_enum():
    assert BlockType.PARAGRAPH.value == "paragraph"


def test_table_validation_and_iteration():
    table = Table(cells=[TableCell(row=0, column=0, content="A")])
    block = Block(id="b1", type=BlockType.TABLE, text="A", spans=[Span(start=0, end=1, text="A")], metadata={"table": table.model_dump()})
    section = Section(id="s1", blocks=[block])
    document = Document(id="doc2", source="clinicaltrials", sections=[section])
    blocks = list(document.iter_blocks())
    assert blocks[0].id == "b1"
    spans = document.find_spans(lambda s: s.end == 1)
    assert spans[0].end == 1

    with pytest.raises(ValueError):
        Table(cells=[TableCell(row=0, column=0, content="A"), TableCell(row=0, column=0, content="B")])
