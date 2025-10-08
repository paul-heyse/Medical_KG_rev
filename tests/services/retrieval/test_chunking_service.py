import sys
from types import SimpleNamespace

import pytest


class _YamlStub(SimpleNamespace):
    YAMLError = Exception

    @staticmethod
    def safe_load(*_args, **_kwargs):
        return {}


sys.modules.setdefault("yaml", _YamlStub())

from Medical_KG_rev.chunking.exceptions import ChunkerConfigurationError
from Medical_KG_rev.chunking.models import Chunk
from Medical_KG_rev.services.retrieval.chunking import ChunkCommand, ChunkingService


class _StubStage:
    def execute(self, context, document):
        return [
            Chunk(
                chunk_id="chunk-1",
                doc_id=document.id,
                tenant_id=context.tenant_id,
                body="hello",
                granularity="paragraph",
                chunker="stub",
                chunker_version="1.0",
                start_char=0,
                end_char=5,
            )
        ]


def test_chunking_service_invokes_stage_with_command() -> None:
    service = ChunkingService(chunk_stage=_StubStage())
    command = ChunkCommand(tenant_id="tenant-a", document_id="doc-1", text="hello")

    chunks = service.chunk(command)

    assert len(chunks) == 1
    assert chunks[0].doc_id == "doc-1"
    assert chunks[0].tenant_id == "tenant-a"


def test_chunking_service_rejects_unknown_strategy() -> None:
    service = ChunkingService(chunk_stage=_StubStage())
    command = ChunkCommand(
        tenant_id="tenant-a",
        document_id="doc-1",
        text="hello",
        strategy="unknown",
    )

    with pytest.raises(ChunkerConfigurationError):
        service.chunk(command)
