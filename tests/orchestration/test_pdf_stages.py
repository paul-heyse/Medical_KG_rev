from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("pydantic")

from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.orchestration.stage_plugins import (
    MineruStage,
    PdfDownloadRecord,
    PdfDownloadStage,
)
from Medical_KG_rev.orchestration.stages.contracts import StageContext
from Medical_KG_rev.services.mineru.processing_service import (
    MineruProcessingError,
    MineruProcessingResult,
)
from Medical_KG_rev.services.pdf import PdfDownloadError, PdfDownloadResult
from Medical_KG_rev.services.pdf.storage import PdfStorageClient
from Medical_KG_rev.storage.object_store import InMemoryObjectStore


@dataclass(slots=True)
class _FakeDownloadService:
    result: PdfDownloadResult

    def download(self, url: str, *, correlation_id: str | None = None) -> PdfDownloadResult:
        return self.result


@dataclass(slots=True)
class _FailingDownloadService:
    def download(self, url: str, *, correlation_id: str | None = None) -> PdfDownloadResult:
        raise PdfDownloadError("boom", error_type="network-error")


@dataclass(slots=True)
class _RecordingStorage(PdfStorageClient):
    cleaned: list[tuple[str, str]] = field(default_factory=list)

    def __init__(self) -> None:
        super().__init__(InMemoryObjectStore())

    def cleanup_document(self, tenant_id: str, document_id: str) -> int:
        self.cleaned.append((tenant_id, document_id))
        return super().cleanup_document(tenant_id, document_id)


@dataclass(slots=True)
class _FakeMineruService:
    result: MineruProcessingResult

    def process_pdf(
        self,
        *,
        tenant_id: str,
        document_id: str,
        pdf_bytes: bytes,
        correlation_id: str | None = None,
        source_label: str | None = None,
    ) -> MineruProcessingResult:
        return self.result


def _build_document() -> Document:
    section = Section(
        id="abstract",
        title="Abstract",
        blocks=[Block(id="b1", type=BlockType.PARAGRAPH, text="content", spans=[])],
    )
    return Document(
        id="doc-1",
        source="openalex",
        title="Test",
        sections=[section],
        metadata={},
        pdf_url="https://example.com/test.pdf",
    )


def test_pdf_download_stage_success() -> None:
    storage = PdfStorageClient(InMemoryObjectStore())
    result = PdfDownloadResult(
        url="https://example.com/test.pdf",
        data=b"%PDF-1.4",
        size_bytes=8,
        content_type="application/pdf",
        checksum="abc",
        duration_seconds=0.1,
        headers={},
        resumed=False,
    )
    stage = PdfDownloadStage(name="download", service=_FakeDownloadService(result), storage=storage)
    ctx = StageContext(tenant_id="tenant", job_id="job", doc_id="doc-1")

    records = stage.execute(ctx, _build_document())

    assert len(records) == 1
    record = records[0]
    assert record.status == "success"
    assert record.storage_key is not None
    assert record.size_bytes == 8
    assert storage.fetch_pdf(record.storage_key) == b"%PDF-1.4"


def test_pdf_download_stage_failure_records_error() -> None:
    storage = _RecordingStorage()
    stage = PdfDownloadStage(name="download", service=_FailingDownloadService(), storage=storage)
    ctx = StageContext(tenant_id="tenant", job_id="job", doc_id="doc-1")

    records = stage.execute(ctx, _build_document())

    assert len(records) == 1
    assert records[0].status == "failed"
    assert records[0].error is not None
    assert storage.cleaned == [("tenant", "doc-1")]


def test_mineru_stage_success() -> None:
    storage = PdfStorageClient(InMemoryObjectStore())
    # preload storage with PDF
    stored = storage.store_pdf(
        tenant_id="tenant",
        document_id="doc-1",
        payload=b"%PDF-1.4",
        checksum="abc",
        content_type="application/pdf",
    )
    ir_document = Document(
        id="doc-1",
        source="mineru",
        sections=[],
        metadata={},
    )
    mineru_result = MineruProcessingResult(
        ir_document=ir_document,
        duration_seconds=1.0,
        metadata={"gpu": "cuda:0"},
    )
    stage = MineruStage(name="mineru", service=_FakeMineruService(mineru_result), storage=storage)
    ctx = StageContext(tenant_id="tenant", job_id="job", doc_id="doc-1")

    output = stage.execute(
        ctx,
        [
            PdfDownloadRecord(
                tenant_id="tenant",
                document_id="doc-1",
                url="https://example.com/test.pdf",
                storage_key=stored.key,
                size_bytes=8,
                content_type="application/pdf",
                checksum="abc",
                duration_seconds=0.1,
                resumed=False,
                status="success",
            )
        ],
    )

    assert output is mineru_result
    assert output.ir_document.id == "doc-1"


def test_mineru_stage_raises_when_missing_download() -> None:
    storage = PdfStorageClient(InMemoryObjectStore())
    fallback_result = MineruProcessingResult(
        ir_document=_build_document(),
        duration_seconds=0.5,
        metadata={},
    )
    stage = MineruStage(name="mineru", service=_FakeMineruService(fallback_result), storage=storage)
    ctx = StageContext(tenant_id="tenant", job_id="job", doc_id="doc-1")

    with pytest.raises(MineruProcessingError):
        stage.execute(ctx, [])
