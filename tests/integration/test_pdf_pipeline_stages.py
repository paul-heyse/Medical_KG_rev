from __future__ import annotations

import httpx
import pytest

from Medical_KG_rev.models.ir import Document
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.stage_plugins import DownloadStage, MineruStage
from Medical_KG_rev.orchestration.stages.contracts import StageContext
from Medical_KG_rev.orchestration.kafka import KafkaClient
from Medical_KG_rev.services.pdf import (
    MineruProcessingService,
    PdfDownloadError,
    PdfDownloadRequest,
    PdfDownloadService,
    PdfMetadata,
    PdfStorageClient,
)
from Medical_KG_rev.services.pdf.gpu_manager import GpuResourceManager
from Medical_KG_rev.services.pdf.storage import PdfStorageConfig


class _Validator:
    def validate(self, url: str) -> PdfMetadata:
        return PdfMetadata(
            url=url,
            content_type="application/pdf",
            size=None,
            last_modified=None,
            accessible=True,
            headers={"Accept-Ranges": "bytes"},
        )


class _ImmediateProcessor:
    def process(self, request):  # pragma: no cover - interface alignment
        from datetime import datetime, timezone

        from Medical_KG_rev.services.mineru.types import MineruResponse, ProcessingMetadata
        from Medical_KG_rev.services.mineru.types import Document as MineruDocument

        now = datetime.now(timezone.utc)
        metadata = ProcessingMetadata(
            document_id=request.document_id,
            mineru_version="1.0",
            model_names={"primary": "test"},
            gpu_id="gpu0",
            worker_id="worker",
            started_at=now,
            completed_at=now,
            duration_seconds=0.01,
            cli_stdout="",
            cli_stderr="",
            cli_descriptor="mineru",
        )
        return MineruResponse(
            document=MineruDocument(document_id=request.document_id, tenant_id=request.tenant_id),
            processed_at=now,
            duration_seconds=0.01,
            metadata=metadata,
        )


def _build_download_stage(storage: PdfStorageClient) -> DownloadStage:
    service = PdfDownloadService(
        storage=storage,
        validator=_Validator(),
        timeout=2.0,
        max_attempts=1,
    )
    return DownloadStage(name="download", download_service=service, storage=storage)


def _build_mineru_stage(storage: PdfStorageClient) -> MineruStage:
    service = MineruProcessingService(
        processor=_ImmediateProcessor(),
        storage=storage,
        gpu_manager=GpuResourceManager(max_concurrent=1),
        timeout=2.0,
        sla_seconds=0.2,
    )
    return MineruStage(name="mineru", processing_service=service)


def _mock_http(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, content=payload)
        return httpx.Response(200)

    transport = httpx.MockTransport(_handler)
    original_stream = httpx.stream

    def _stream(method: str, url: str, **kwargs):
        kwargs.setdefault("transport", transport)
        return original_stream(method, url, **kwargs)

    monkeypatch.setattr(httpx, "stream", _stream)


def test_pdf_pipeline_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_http(monkeypatch, b"%PDF")
    storage = PdfStorageClient(
        config=PdfStorageConfig(base_prefix="pdf-test", enable_access_logging=False)
    )
    ledger = JobLedger()
    ledger.create(job_id="job-1", doc_key="doc-1", tenant_id="tenant", pipeline="pipeline")
    kafka = KafkaClient()

    download_stage = _build_download_stage(storage)
    download_stage.bind_runtime(ledger=ledger, kafka=kafka)
    mineru_stage = _build_mineru_stage(storage)
    mineru_stage.bind_runtime(ledger=ledger, kafka=kafka)

    ctx = StageContext(
        tenant_id="tenant",
        job_id="job-1",
        doc_id="doc-1",
        correlation_id="corr",
        metadata={},
        pipeline_name="pipeline",
        pipeline_version="v1",
    )
    document = Document(id="doc-1", source="openalex", pdf_url="https://example.org/sample.pdf")

    download_result = download_stage.execute(ctx, document)
    assert download_result is not None

    entry = ledger.get("job-1")
    assert entry is not None
    assert entry.pdf_downloaded is True
    assert entry.pdf_failure_code is None

    mineru_result = mineru_stage.execute(ctx, download_result)
    assert mineru_result is not None
    assert mineru_result.response.document.document_id == "doc-1"

    updated = ledger.get("job-1")
    assert updated is not None
    assert updated.pdf_ir_ready is True
    assert updated.pdf_failure_code is None
    assert kafka.pending("pdf.deadletter.v1") == 0


def test_pdf_pipeline_handles_multiple_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_http(monkeypatch, b"%PDF multi")
    storage = PdfStorageClient(
        config=PdfStorageConfig(base_prefix="pdf-test", enable_access_logging=False)
    )
    ledger = JobLedger()
    kafka = KafkaClient()

    download_stage = _build_download_stage(storage)
    download_stage.bind_runtime(ledger=ledger, kafka=kafka)
    mineru_stage = _build_mineru_stage(storage)
    mineru_stage.bind_runtime(ledger=ledger, kafka=kafka)

    for index in range(3):
        job_id = f"job-{index}"
        doc_id = f"doc-{index}"
        ledger.create(job_id=job_id, doc_key=doc_id, tenant_id="tenant", pipeline="pipeline")
        ctx = StageContext(
            tenant_id="tenant",
            job_id=job_id,
            doc_id=doc_id,
            correlation_id=f"corr-{index}",
            metadata={},
            pipeline_name="pipeline",
            pipeline_version="v1",
        )
        document = Document(id=doc_id, source="openalex", pdf_url="https://example.org/sample.pdf")
        download_result = download_stage.execute(ctx, document)
        assert download_result is not None
        mineru_stage.execute(ctx, download_result)
        entry = ledger.get(job_id)
        assert entry is not None
        assert entry.pdf_ir_ready is True

    assert kafka.pending("pdf.deadletter.v1") == 0


def test_pdf_download_failure_routes_to_dlq(monkeypatch: pytest.MonkeyPatch) -> None:
    storage = PdfStorageClient(
        config=PdfStorageConfig(base_prefix="pdf-test", enable_access_logging=False)
    )
    ledger = JobLedger()
    ledger.create(job_id="job-2", doc_key="doc-2", tenant_id="tenant", pipeline="pipeline")
    kafka = KafkaClient()

    download_stage = _build_download_stage(storage)
    download_stage.bind_runtime(ledger=ledger, kafka=kafka)

    def _failing_download(request: PdfDownloadRequest):
        raise PdfDownloadError("validation error", retryable=False, code="validation")

    download_stage.download_service.download = _failing_download  # type: ignore[assignment]

    ctx = StageContext(
        tenant_id="tenant",
        job_id="job-2",
        doc_id="doc-2",
        correlation_id="corr",
        metadata={},
        pipeline_name="pipeline",
        pipeline_version="v1",
    )
    document = Document(id="doc-2", source="openalex", pdf_url="https://example.org/sample.pdf")

    with pytest.raises(PdfDownloadError):
        download_stage.execute(ctx, document)

    entry = ledger.get("job-2")
    assert entry is not None
    assert entry.pdf_downloaded is False
    assert entry.pdf_failure_code == "validation"
    assert kafka.pending("pdf.deadletter.v1") == 1
