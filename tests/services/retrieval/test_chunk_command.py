import sys
from types import SimpleNamespace

import pytest


class _YamlStub(SimpleNamespace):
    YAMLError = Exception

    @staticmethod
    def safe_load(*_args, **_kwargs):
        return {}


sys.modules.setdefault("yaml", _YamlStub())

from Medical_KG_rev.chunking.exceptions import InvalidDocumentError
from Medical_KG_rev.services.retrieval.chunking import ChunkCommand


def test_chunk_command_requires_text() -> None:
    with pytest.raises(InvalidDocumentError):
        ChunkCommand(
            tenant_id="tenant-a",
            document_id="doc-1",
            text=" ",
        )


def test_chunk_command_context_metadata_includes_options() -> None:
    command = ChunkCommand(
        tenant_id="tenant-a",
        document_id="doc-1",
        text="hello",
        strategy="section",
        chunk_size=256,
        overlap=0.15,
        options={"profile": "fast", "section_title": "Intro"},
    )

    metadata = command.context_metadata()

    assert metadata["strategy"] == "section"
    assert metadata["max_tokens"] == 256
    assert metadata["overlap"] == 0.15
    assert metadata["profile"] == "fast"
    assert "text" not in metadata


def test_chunk_command_build_document_generates_blocks() -> None:
    command = ChunkCommand(
        tenant_id="tenant-a",
        document_id="doc-1",
        text="para one\n\npara two",
        options={"title": "Doc", "source": "gateway"},
    )

    document = command.build_document()

    assert document.id == "doc-1"
    assert document.metadata["title"] == "Doc"
    assert len(document.sections[0].blocks) == 2
