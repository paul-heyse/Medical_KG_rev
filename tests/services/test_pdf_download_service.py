from __future__ import annotations

import httpx
import pytest

from Medical_KG_rev.services.pdf import (
    DownloadCircuitBreaker,
    PdfDownloadError,
    PdfDownloadRequest,
    PdfDownloadService,
    PdfMetadata,
    PdfStorageClient,
)
from Medical_KG_rev.services.pdf.storage import PdfStorageConfig
from Medical_KG_rev.storage.object_store import InMemoryObjectStore


class _FakeValidator:
    def __init__(self, headers: dict[str, str]) -> None:
        self._headers = headers

    def validate(self, url: str) -> PdfMetadata:
        return PdfMetadata(
            url=url,
            content_type="application/pdf",
            size=None,
            last_modified=None,
            accessible=True,
            headers=self._headers,
        )


def test_pdf_download_service_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"%PDF-1.4 test document"

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

    storage = PdfStorageClient(
        backend=InMemoryObjectStore(),
        config=PdfStorageConfig(base_prefix="pdf-test", enable_access_logging=False),
    )
    service = PdfDownloadService(
        storage=storage,
        validator=_FakeValidator({"Accept-Ranges": "bytes"}),
        timeout=5.0,
        max_attempts=1,
    )

    request = PdfDownloadRequest(
        tenant_id="tenant-a",
        document_id="doc-1",
        url="https://example.org/sample.pdf",
    )

    result = service.download(request)
    assert result.size == len(payload)
    stored = storage.run(storage.fetch(result.storage_key))
    assert stored == payload


def test_pdf_download_service_circuit_breaker_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    transport = httpx.MockTransport(_handler)
    original_stream = httpx.stream

    def _stream(method: str, url: str, **kwargs):
        kwargs.setdefault("transport", transport)
        return original_stream(method, url, **kwargs)

    monkeypatch.setattr(httpx, "stream", _stream)

    storage = PdfStorageClient(
        backend=InMemoryObjectStore(),
        config=PdfStorageConfig(base_prefix="pdf-test", enable_access_logging=False),
    )
    circuit_breaker = DownloadCircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
    service = PdfDownloadService(
        storage=storage,
        validator=_FakeValidator({}),
        timeout=1.0,
        max_attempts=1,
        circuit_breaker=circuit_breaker,
    )

    request = PdfDownloadRequest(
        tenant_id="tenant-a",
        document_id="doc-1",
        url="https://example.org/sample.pdf",
    )

    with pytest.raises(PdfDownloadError) as first:
        service.download(request)
    assert first.value.retryable is True

    with pytest.raises(PdfDownloadError) as second:
        service.download(request)
    assert second.value.code == "circuit_open"
    assert second.value.retryable is False
