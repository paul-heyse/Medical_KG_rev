"""Robust PDF download service used by orchestration stages."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from tempfile import SpooledTemporaryFile
from typing import Mapping
from urllib.parse import urlparse

import httpx
import structlog
from pybreaker import CircuitBreakerError

from Medical_KG_rev.observability.metrics import (
    record_pdf_download_failure,
    record_pdf_download_success,
)
from Medical_KG_rev.utils.http_client import HttpClient

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class PdfDownloadResult:
    """Represents a completed PDF download."""

    url: str
    data: bytes
    size_bytes: int
    content_type: str | None
    checksum: str
    duration_seconds: float
    headers: Mapping[str, str]
    resumed: bool


class PdfDownloadError(RuntimeError):
    """Raised when a PDF cannot be downloaded or validated."""

    def __init__(self, message: str, *, error_type: str = "download-error") -> None:
        super().__init__(message)
        self.error_type = error_type


class PdfDownloadService:
    """Service responsible for retrieving PDFs over HTTP with resilience."""

    def __init__(
        self,
        client: HttpClient,
        *,
        chunk_size: int = 1024 * 128,
        spool_threshold: int = 8 * 1024 * 1024,
        max_resume_attempts: int = 3,
        accept_content_types: tuple[str, ...] = (
            "application/pdf",
            "application/octet-stream",
        ),
    ) -> None:
        self._client = client
        self._chunk_size = max(1024, chunk_size)
        self._spool_threshold = max(1024 * 1024, spool_threshold)
        self._max_resume_attempts = max(1, max_resume_attempts)
        self._accept_content_types = accept_content_types

    def download(
        self,
        url: str,
        *,
        correlation_id: str | None = None,
    ) -> PdfDownloadResult:
        """Download a PDF from the given URL and return the payload."""

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise PdfDownloadError(
                f"Unsupported URL scheme for PDF download: {url}", error_type="invalid-url"
            )

        attempt = 0
        total_downloaded = 0
        resumed = False
        sha256 = hashlib.sha256()
        start_time = time.monotonic()
        headers: dict[str, str] = {
            "Accept": "application/pdf,application/octet-stream;q=0.8",
        }

        with SpooledTemporaryFile(max_size=self._spool_threshold) as buffer:
            while True:
                attempt += 1
                range_headers = dict(headers)
                if total_downloaded:
                    range_headers["Range"] = f"bytes={total_downloaded}-"
                logger.info(
                    "pdf.download.attempt",
                    url=url,
                    attempt=attempt,
                    resumed=total_downloaded > 0,
                    correlation_id=correlation_id,
                )
                try:
                    response = self._client.request(
                        "GET",
                        url,
                        headers=range_headers,
                        stream=True,
                    )
                except httpx.TimeoutException as exc:
                    record_pdf_download_failure(source="http", error_type="timeout")
                    raise PdfDownloadError("PDF download timed out", error_type="timeout") from exc
                except httpx.NetworkError as exc:
                    record_pdf_download_failure(source="http", error_type="network-error")
                    raise PdfDownloadError(
                        "Network error while downloading PDF",
                        error_type="network-error",
                    ) from exc
                except CircuitBreakerError as exc:
                    record_pdf_download_failure(source="http", error_type="circuit-open")
                    raise PdfDownloadError(
                        "Circuit breaker open for PDF download",
                        error_type="circuit-open",
                    ) from exc
                try:
                    response.raise_for_status()
                except httpx.HTTPError as exc:  # pragma: no cover - handled by HttpClient retries
                    raise PdfDownloadError(str(exc), error_type="http-error") from exc

                if total_downloaded and response.status_code != httpx.codes.PARTIAL_CONTENT:
                    logger.warning(
                        "pdf.download.resume_unsupported",
                        url=url,
                        status=response.status_code,
                    )
                    buffer.seek(0)
                    buffer.truncate(0)
                    sha256 = hashlib.sha256()
                    total_downloaded = 0

                try:
                    for chunk in response.iter_bytes(chunk_size=self._chunk_size):
                        if not chunk:
                            continue
                        buffer.write(chunk)
                        sha256.update(chunk)
                        total_downloaded += len(chunk)
                except httpx.HTTPError as exc:
                    if attempt >= self._max_resume_attempts:
                        record_pdf_download_failure(source="http", error_type="stream-error")
                        raise PdfDownloadError(
                            "PDF download interrupted repeatedly",
                            error_type="stream-error",
                        ) from exc
                    resumed = True
                    logger.warning(
                        "pdf.download.resume",
                        url=url,
                        attempt=attempt,
                        downloaded=total_downloaded,
                        error=str(exc),
                    )
                    continue
                break

            duration = time.monotonic() - start_time
            buffer.seek(0)
            payload = buffer.read()

        content_type = response.headers.get("Content-Type")
        if content_type:
            content_type = content_type.split(";")[0].strip().lower()
        if content_type and not any(
            content_type.startswith(prefix) for prefix in self._accept_content_types
        ):
            record_pdf_download_failure(source="http", error_type="invalid-content-type")
            raise PdfDownloadError(
                f"Remote resource is not a PDF (content-type={content_type})",
                error_type="invalid-content-type",
            )

        checksum = sha256.hexdigest()
        logger.info(
            "pdf.download.completed",
            url=url,
            bytes=total_downloaded,
            checksum=checksum,
            duration_seconds=round(duration, 3),
            resumed=resumed,
            correlation_id=correlation_id,
        )
        record_pdf_download_success(source="http", size_bytes=total_downloaded, duration_seconds=duration)

        return PdfDownloadResult(
            url=url,
            data=payload,
            size_bytes=total_downloaded,
            content_type=content_type,
            checksum=checksum,
            duration_seconds=duration,
            headers=dict(response.headers),
            resumed=resumed,
        )
