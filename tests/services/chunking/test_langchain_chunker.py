from __future__ import annotations

from typing import Any

import pytest

from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.services.chunking.wrappers import langchain_splitter


class _FakeSplitter:
    def __init__(self, *, chunk_size: int, chunk_overlap: int, length_function):
        self.params = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
        self.length_function = length_function

    def split_text(self, text: str) -> list[str]:
        return [part for part in text.split("|") if part]


class _FakeTokenizer:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def encode(self, text: str) -> list[int]:
        return text.split()


def _document() -> Document:
    return Document(
        id="doc-1",
        source="pmc",
        sections=[
            Section(
                id="s1",
                title="Introduction",
                blocks=[
                    Block(
                        id="b1",
                        type=BlockType.PARAGRAPH,
                        text="Alpha|Beta",
                    )
                ],
            ),
            Section(
                id="s2",
                title="Methods",
                blocks=[
                    Block(
                        id="b2",
                        type=BlockType.PARAGRAPH,
                        text="Gamma",
                    )
                ],
            ),
        ],
    )


@pytest.fixture(autouse=True)
def _stub_dependencies(monkeypatch):
    def fake_loader() -> tuple[type[_FakeSplitter], type[_FakeTokenizer]]:
        return _FakeSplitter, _FakeTokenizer

    monkeypatch.setattr(langchain_splitter, "_ensure_langchain_dependencies", fake_loader)
    yield


def test_langchain_chunker_respects_boundaries():
    profile = {
        "name": "pmc-imrad",
        "chunker_type": "langchain_recursive",
        "target_tokens": 8,
        "overlap_tokens": 1,
        "respect_boundaries": ["section"],
        "filters": [],
        "metadata": {"chunker_version": "langchain-v0.2.0"},
    }

    chunker = langchain_splitter.LangChainChunker(profile=profile)
    chunks = chunker.chunk(_document(), profile="pmc-imrad")

    assert [chunk.text for chunk in chunks] == ["Alpha", "Beta", "Gamma"]
    assert [chunk.section_label for chunk in chunks] == [
        "Introduction",
        "Introduction",
        "Methods",
    ]
    assert [chunk.char_offsets for chunk in chunks] == [(0, 5), (6, 10), (10, 15)]
    assert all(chunk.metadata["chunking_profile"] == "pmc-imrad" for chunk in chunks)


def test_langchain_chunker_uses_token_length(monkeypatch):
    profile = {
        "name": "pmc-imrad",
        "chunker_type": "langchain_recursive",
        "target_tokens": 4,
        "overlap_tokens": 0,
        "respect_boundaries": ["section"],
        "filters": [],
    }

    captured: dict[str, Any] = {}

    class _ObservingSplitter(_FakeSplitter):
        def __init__(self, *, chunk_size: int, chunk_overlap: int, length_function):
            super().__init__(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=length_function,
            )
            captured["chunk_size"] = chunk_size
            captured["chunk_overlap"] = chunk_overlap
            captured["token_count"] = length_function("one two three")

    def fake_loader() -> tuple[type[_ObservingSplitter], type[_FakeTokenizer]]:
        return _ObservingSplitter, _FakeTokenizer

    monkeypatch.setattr(langchain_splitter, "_ensure_langchain_dependencies", fake_loader)

    chunker = langchain_splitter.LangChainChunker(profile=profile)
    chunker.chunk(_document(), profile="pmc-imrad")

    assert captured["chunk_size"] == profile["target_tokens"] * 4
    assert captured["chunk_overlap"] == profile["overlap_tokens"] * 4
    assert captured["token_count"] == 3


@pytest.mark.parametrize(
    ("section_title", "text", "expected_chunks"),
    [
        ("Background", "Alpha|Beta", 2),
        ("Methods", "Gamma", 1),
        ("Results", "Delta|Epsilon|Zeta", 3),
        ("Discussion", "Eta", 1),
        ("Conclusion", "Theta|Iota", 2),
    ],
)
def test_langchain_chunker_handles_multiple_documents(section_title, text, expected_chunks):
    profile = {
        "name": "pmc-imrad",
        "chunker_type": "langchain_recursive",
        "target_tokens": 6,
        "overlap_tokens": 0,
        "respect_boundaries": ["section"],
        "filters": [],
    }

    document = Document(
        id="doc-param",
        source="pmc",
        sections=[
            Section(
                id="section-1",
                title=section_title,
                blocks=[
                    Block(
                        id="block-1",
                        type=BlockType.PARAGRAPH,
                        text=text,
                    )
                ],
            )
        ],
    )

    chunker = langchain_splitter.LangChainChunker(profile=profile)
    chunks = chunker.chunk(document, profile="pmc-imrad")

    assert len(chunks) == expected_chunks
    assert {chunk.section_label for chunk in chunks} == {section_title}
