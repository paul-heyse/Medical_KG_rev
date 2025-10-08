from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from Medical_KG_rev.services.pdf import MineruProcessingError, MineruProcessingService, PdfStorageClient
from Medical_KG_rev.services.pdf.gpu_manager import GpuResourceManager
from Medical_KG_rev.services.pdf.storage import PdfStorageConfig
from Medical_KG_rev.services.mineru.types import Document as MineruDocument
from Medical_KG_rev.services.mineru.types import MineruResponse, ProcessingMetadata


class _ImmediateProcessor:
    def process(self, request):  # pragma: no cover - interface match
        now = datetime.now(timezone.utc)
        document = MineruDocument(document_id=request.document_id, tenant_id=request.tenant_id)
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
            document=document,
            processed_at=now,
            duration_seconds=0.01,
            metadata=metadata,
        )


class _SlowProcessor:
    def __init__(self, delay: float) -> None:
        self.delay = delay

    def process(self, request):  # pragma: no cover - interface match
        time.sleep(self.delay)
        return _ImmediateProcessor().process(request)


def _make_storage() -> PdfStorageClient:
    return PdfStorageClient(
        config=PdfStorageConfig(base_prefix="pdf-test", enable_access_logging=False)
    )


def _store_pdf(storage: PdfStorageClient, tenant_id: str, document_id: str, content: bytes) -> tuple[str, str]:
    return storage.run(
        storage.store(
            tenant_id=tenant_id,
            document_id=document_id,
            data=content,
            content_type="application/pdf",
            metadata=None,
        )
    )


def test_processing_service_success_stores_state() -> None:
    storage = _make_storage()
    key, checksum = _store_pdf(storage, "tenant-a", "doc-1", b"%PDF")
    service = MineruProcessingService(
        processor=_ImmediateProcessor(),
        storage=storage,
        gpu_manager=GpuResourceManager(max_concurrent=1),
        timeout=2.0,
        sla_seconds=0.05,
    )

    result = service.process(
        tenant_id="tenant-a",
        document_id="doc-1",
        storage_key=key,
        checksum=checksum,
        correlation_id="corr",
    )

    assert result.checksum == checksum
    state = storage.run(storage.fetch_processing_state("tenant-a", "doc-1"))
    assert state is not None
    assert state["status"] == "completed"
    assert state["checksum"] == checksum


def test_processing_service_timeout_records_partial_state() -> None:
    storage = _make_storage()
    key, checksum = _store_pdf(storage, "tenant-a", "doc-2", b"%PDF")
    service = MineruProcessingService(
        processor=_SlowProcessor(delay=0.2),
        storage=storage,
        gpu_manager=GpuResourceManager(max_concurrent=1),
        timeout=0.05,
        sla_seconds=0.5,
    )

    with pytest.raises(MineruProcessingError) as excinfo:
        service.process(
            tenant_id="tenant-a",
            document_id="doc-2",
            storage_key=key,
            checksum=checksum,
            correlation_id="corr",
        )

    error = excinfo.value
    assert error.retryable is True
    assert error.code == "timeout"

    state = storage.run(storage.fetch_processing_state("tenant-a", "doc-2"))
    assert state is not None
    assert state["status"] == "partial"
    assert state["code"] == "timeout"


def test_processing_service_enforces_gpu_concurrency() -> None:
    storage = _make_storage()
    key1, checksum1 = _store_pdf(storage, "tenant-a", "doc-3", b"%PDF")
    key2, checksum2 = _store_pdf(storage, "tenant-a", "doc-4", b"%PDF")
    gpu_manager = GpuResourceManager(max_concurrent=1)
    service = MineruProcessingService(
        processor=_SlowProcessor(delay=0.1),
        storage=storage,
        gpu_manager=gpu_manager,
        timeout=1.0,
        sla_seconds=0.5,
    )

    def _run(doc_id: str, key: str, checksum: str):
        return service.process(
            tenant_id="tenant-a",
            document_id=doc_id,
            storage_key=key,
            checksum=checksum,
            correlation_id="corr",
        )

    start = time.perf_counter()
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut1 = executor.submit(_run, "doc-3", key1, checksum1)
        fut2 = executor.submit(_run, "doc-4", key2, checksum2)
        fut1.result()
        fut2.result()

    duration = time.perf_counter() - start
    assert duration >= 0.2
