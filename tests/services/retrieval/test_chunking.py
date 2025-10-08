from __future__ import annotations

import pytest

from Medical_KG_rev.chunking.exceptions import InvalidDocumentError
from Medical_KG_rev.chunking.models import Chunk
from Medical_KG_rev.orchestration.stages.contracts import StageContext
from Medical_KG_rev.services.retrieval.chunking import ChunkingService
from Medical_KG_rev.services.retrieval.chunking_command import ChunkCommand


class _StubChunkStage:
    def __init__(self) -> None:
        self.executions: list[tuple[StageContext, object]] = []

    def execute(self, ctx: StageContext, document) -> list[Chunk]:  # noqa: ANN001 - interface contract
        self.executions.append((ctx, document))
        text = document.sections[0].blocks[0].text or ""
        return [
            Chunk(
                chunk_id=f"{document.id}:chunk:0",
                doc_id=document.id,
                tenant_id=ctx.tenant_id,
                body=text,
                title_path=tuple(filter(None, (document.title,))),
                section=document.sections[0].id,
                start_char=0,
                end_char=len(text),
                granularity="paragraph",
                chunker="stub",
                chunker_version="1.0.0",
                meta={},
            )
        ]


def test_chunking_service_builds_document() -> None:
    stage = _StubChunkStage()
    service = ChunkingService(chunk_stage=stage)
    command = ChunkCommand(
        tenant_id="tenant-1",
        document_id="doc-1",
        text="Heading\n\nBody paragraph.",
        strategy="semantic",
        metadata={"title": "Sample"},
    )
    chunks = service.chunk(command)
    assert len(chunks) == 1
    ctx, document = stage.executions[0]
    assert ctx.tenant_id == "tenant-1"
    assert document.id == "doc-1"
    assert document.metadata["title"] == "Sample"
    assert ctx.metadata["strategy"] == "semantic"
    assert chunks[0].body == "Heading\n\nBody paragraph."


def test_chunking_service_requires_text() -> None:
    with pytest.raises(InvalidDocumentError):
        ChunkCommand(tenant_id="tenant", document_id="doc", text="  ")


def test_available_strategies_exposed() -> None:
    service = ChunkingService(chunk_stage=_StubChunkStage())
    strategies = service.available_strategies()
    assert "semantic" in strategies
    assert all(isinstance(strategy, str) for strategy in strategies)
