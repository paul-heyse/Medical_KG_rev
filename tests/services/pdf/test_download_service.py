import pytest

pytest.importorskip("httpx")
import httpx

pytest.importorskip("respx")
import respx
pytest.importorskip("pydantic")
from httpx import Response

from Medical_KG_rev.services.pdf import PdfDownloadError, PdfDownloadService
from Medical_KG_rev.utils.http_client import BackoffStrategy, HttpClient, RetryConfig


@pytest.fixture()
def download_service() -> PdfDownloadService:
    client = HttpClient(
        retry=RetryConfig(
            attempts=1,
            backoff_strategy=BackoffStrategy.NONE,
            jitter=False,
            timeout=5.0,
        )
    )
    service = PdfDownloadService(client)
    try:
        yield service
    finally:
        client.close()


@respx.mock
def test_pdf_download_success(download_service: PdfDownloadService) -> None:
    url = "https://example.com/test.pdf"
    respx.get(url).mock(
        return_value=Response(
            200,
            headers={"Content-Type": "application/pdf", "Content-Length": "8"},
            content=b"%PDF-1.4",
        )
    )

    result = download_service.download(url)

    assert result.url == url
    assert result.size_bytes == 8
    assert result.content_type == "application/pdf"
    assert result.checksum
    assert result.duration_seconds >= 0.0


@respx.mock
def test_pdf_download_invalid_content_type(download_service: PdfDownloadService) -> None:
    url = "https://example.com/not-pdf"
    respx.get(url).mock(
        return_value=Response(
            200,
            headers={"Content-Type": "text/html"},
            content=b"<html></html>",
        )
    )

    with pytest.raises(PdfDownloadError) as excinfo:
        download_service.download(url)
    assert excinfo.value.error_type == "invalid-content-type"


def test_pdf_download_timeout_error_type() -> None:
    class _TimeoutClient:
        def request(self, method: str, url: str, **kwargs):
            raise httpx.TimeoutException("timeout", request=httpx.Request(method, url))

        def close(self) -> None:
            return None

    service = PdfDownloadService(_TimeoutClient())

    with pytest.raises(PdfDownloadError) as excinfo:
        service.download("https://example.com/timeout.pdf")
    assert excinfo.value.error_type == "timeout"
