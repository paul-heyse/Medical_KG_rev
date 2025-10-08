from dataclasses import dataclass

import pytest

from Medical_KG_rev.chunking.exceptions import InvalidDocumentError
from Medical_KG_rev.services.retrieval.chunking_command import (
    ChunkCommand,
    ChunkRequestProtocol,
)


@dataclass
class _StubRequest(ChunkRequestProtocol):
    tenant_id: str
    document_id: str
    strategy: str
    chunk_size: int | None
    overlap: float


def test_chunk_command_from_request_normalises_metadata() -> None:
    request = _StubRequest(
        tenant_id="tenant-x",
        document_id="doc-42",
        strategy="semantic",
        chunk_size=256,
        overlap=0.15,
    )
    command = ChunkCommand.from_request(
        request,
        text="Body",
        metadata={"profile": "pdf", "title": "Sample"},
        correlation_id="corr-1",
        context={"job_id": "job-7"},
    )
    assert command.profile == "pdf"
    assert command.metadata["title"] == "Sample"
    assert command.chunk_size == 256
    assert command.context["job_id"] == "job-7"
    payload = command.asdict()
    assert payload["profile"] == "pdf"
    assert "text" not in payload


def test_chunk_command_with_context_merges_values() -> None:
    command = ChunkCommand(
        tenant_id="tenant-x",
        document_id="doc-42",
        text="body",
    )
    enriched = command.with_context(job_id="job-1", endpoint="rest")
    assert command.context == {}
    assert enriched.context["job_id"] == "job-1"
    assert enriched.context["endpoint"] == "rest"


def test_chunk_command_validates_overlap_range() -> None:
    with pytest.raises(InvalidDocumentError):
        ChunkCommand(
            tenant_id="tenant-x",
            document_id="doc-42",
            text="body",
            overlap=1.5,
        )
