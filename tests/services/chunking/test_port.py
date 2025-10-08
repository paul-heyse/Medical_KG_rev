from __future__ import annotations

from pathlib import Path

import pytest

from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.services.chunking import (
    Chunk,
    UnknownChunkerError,
    chunk_document,
    reset_registry,
)
from Medical_KG_rev.services.chunking.port import register_chunker
from Medical_KG_rev.services.chunking.profiles.loader import ProfileRepository
from Medical_KG_rev.services.chunking.wrappers import simple


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_registry()
    simple.register()
    yield
    reset_registry()


@pytest.fixture
def document() -> Document:
    section = Section(
        id="sec-1",
        title="Introduction",
        blocks=[
            Block(
                id="b1",
                type=BlockType.PARAGRAPH,
                text="Sentence one. Sentence two.",
                metadata={"intent_hint": "narrative"},
            ),
            Block(id="b2", type=BlockType.PARAGRAPH, text="Sentence three."),
        ],
    )
    return Document(id="doc-1", source="unit-test", sections=[section])


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    profile = tmp_path / "default.yaml"
    profile.write_text(
        """
name: default
domain: test
chunker_type: simple
target_tokens: 5
overlap_tokens: 0
respect_boundaries:
  - section
sentence_splitter: simple
metadata:
  intent_hints:
    Introduction: narrative
        """.strip()
    )
    return tmp_path


def profile_loader_factory(profile_dir: Path):
    repo = ProfileRepository(directory=profile_dir)

    def loader(name: str) -> dict[str, str]:
        profile = repo.get(name)
        return profile.model_dump()

    return loader


def test_chunk_document(document: Document, profile_dir: Path) -> None:
    loader = profile_loader_factory(profile_dir)
    chunks = chunk_document(document, profile_name="default", profile_loader=loader)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.doc_id == document.id for chunk in chunks)
    assert {chunk.intent_hint for chunk in chunks} == {"narrative"}
    for chunk in chunks:
        assert chunk.metadata["chunking_profile"] == "default"
        assert chunk.metadata["source_system"] == document.source
        assert "chunker_version" in chunk.metadata
        assert "created_at" in chunk.metadata


def test_unknown_chunker(document: Document, profile_dir: Path) -> None:
    loader = profile_loader_factory(profile_dir)
    reset_registry()
    with pytest.raises(UnknownChunkerError):
        chunk_document(document, profile_name="default", profile_loader=loader)


def test_custom_registration(document: Document, profile_dir: Path) -> None:
    class DummyChunker:
        def __init__(self, *, profile: dict[str, str]) -> None:
            self.profile = profile

        def chunk(self, document: Document, *, profile: str):
            return []

    register_chunker("dummy", lambda *, profile: DummyChunker(profile=profile))
    base_loader = profile_loader_factory(profile_dir)

    def loader(name: str) -> dict[str, str]:
        profile = base_loader(name)
        profile["chunker_type"] = "dummy"
        return profile

    chunks = chunk_document(document, profile_name="default", profile_loader=loader)
    assert chunks == []
