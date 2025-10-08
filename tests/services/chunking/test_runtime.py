from types import SimpleNamespace

from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.services.chunking import runtime
from Medical_KG_rev.services.chunking.runtime import _BlockContext


def _make_document() -> Document:
    return Document(
        id="doc-1",
        source="unit-test",
        sections=[
            Section(
                id="s1",
                title="Introduction",
                blocks=[
                    Block(id="b1", type=BlockType.PARAGRAPH, text="Alpha"),
                    Block(id="b2", type=BlockType.TABLE, text="Beta"),
                ],
            ),
            Section(
                id="s2",
                title="Methods",
                blocks=[Block(id="b3", type=BlockType.PARAGRAPH, text="Gamma")],
            ),
        ],
    )


def test_iter_block_contexts_and_group_contexts_boundaries():
    document = _make_document()
    contexts = list(runtime.iter_block_contexts(document))

    assert [(ctx.block.id, ctx.start, ctx.end) for ctx in contexts] == [
        ("b1", 0, 5),
        ("b2", 5, 9),
        ("b3", 9, 14),
    ]

    grouped = runtime.group_contexts(contexts, respect_boundaries=["section", "table"])
    assert [[ctx.block.id for ctx in group] for group in grouped] == [["b1"], ["b2"], ["b3"]]

    # Table-only boundary check to ensure the fast path is exercised
    grouped_tables = runtime.group_contexts(contexts, respect_boundaries=["table"])
    assert [[ctx.block.id for ctx in group] for group in grouped_tables] == [["b1"], ["b2"], ["b3"]]


def test_assemble_chunks_handles_alignment_mismatch_and_metadata():
    document = _make_document()
    contexts = list(runtime.iter_block_contexts(document))
    groups = [[contexts[0]], [contexts[2]]]

    chunks = runtime.assemble_chunks(
        document=document,
        profile_name="pmc-imrad",
        groups=groups,
        chunk_texts=["Alpha", "Delta"],
        chunk_to_group_index=[0, 1],
        intent_hint_provider=lambda section: section.title.lower() if section else None,
        metadata_provider=lambda _: {"chunker_version": "v2"},
    )

    assert [chunk.text for chunk in chunks] == ["Alpha", "Delta"]
    assert [chunk.char_offsets for chunk in chunks] == [(0, 5), (9, 14)]
    assert all(chunk.chunk_id.startswith("doc-1:") for chunk in chunks)
    assert chunks[0].metadata["chunker_version"] == "v2"
    assert chunks[0].metadata["source_system"] == "unit-test"
    assert chunks[0].metadata["chunking_profile"] == "pmc-imrad"
    assert "created_at" in chunks[0].metadata
    assert chunks[1].intent_hint == "methods"


def test_group_contexts_table_boundary_manual():
    section = Section(id="s-manual", title="Manual", blocks=[])
    para_block = Block(id="p", type=BlockType.PARAGRAPH, text="Alpha")
    table_block = Block(id="t", type=BlockType.TABLE, text="Beta")
    contexts = [
        _BlockContext(block=para_block, section=section, text="Alpha", start=0, end=5),
        _BlockContext(block=table_block, section=section, text="Beta", start=5, end=9),
    ]

    grouped = runtime.group_contexts(contexts, respect_boundaries=["table"])
    assert [[ctx.block.id for ctx in group] for group in grouped] == [["p"], ["t"]]


def test_assemble_chunks_skips_empty_text_contexts():
    document = Document(
        id="doc-2",
        source="unit-test",
        sections=[
            Section(
                id="s1",
                title="Abstract",
                blocks=[
                    Block(id="b0", type=BlockType.PARAGRAPH, text=None),
                    Block(id="b1", type=BlockType.PARAGRAPH, text="Actual content"),
                ],
            )
        ],
    )
    contexts = list(runtime.iter_block_contexts(document))
    groups = [contexts]

    chunks = runtime.assemble_chunks(
        document=document,
        profile_name="pmc-imrad",
        groups=groups,
        chunk_texts=["Actual content"],
        chunk_to_group_index=[0],
        intent_hint_provider=lambda section: section.title if section else None,
        metadata_provider=None,
    )

    assert chunks[0].text == "Actual content"
    assert chunks[0].char_offsets == (0, 14)
    assert chunks[0].metadata["chunker_version"] == "unknown"


def test_build_chunk_includes_section_metadata():
    document = _make_document()
    section = SimpleNamespace(title="  Results  ", metadata={"intent": "outcome"})
    chunk = runtime.build_chunk(
        document=document,
        profile_name="guideline",
        text="Result text",
        mapping=[0, 1, 2],
        section=section,
        intent_hint="summary",
        metadata={"source_system": "override"},
    )

    assert chunk.section_label == "Results"
    assert chunk.metadata["source_system"] == "override"
    assert chunk.metadata["section_metadata"] == {"intent": "outcome"}


def test_build_chunk_defaults_with_empty_mapping():
    document = _make_document()
    chunk = runtime.build_chunk(
        document=document,
        profile_name="default",
        text="Snippet",
        mapping=[None, None],
        section=None,
        intent_hint=None,
        metadata=None,
    )

    assert chunk.char_offsets == (0, 0)
    assert chunk.intent_hint == ""
    assert chunk.metadata["chunker_version"] == "unknown"


def test_intent_providers():
    section = SimpleNamespace(metadata={"intent": "ae", "intent_hint": "safety"})
    assert runtime.identity_intent_provider(section) == "ae"
    assert runtime.default_intent_provider(section) == "safety"
    assert runtime.identity_intent_provider(SimpleNamespace(metadata="oops")) is None
    assert runtime.default_intent_provider(SimpleNamespace(metadata=None)) is None
    assert runtime.identity_intent_provider(None) is None
    assert runtime.default_intent_provider(None) is None
