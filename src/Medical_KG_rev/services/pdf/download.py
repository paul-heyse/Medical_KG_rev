from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog
from opentelemetry import trace

from Medical_KG_rev.observability.alerts import get_alert_manager
from Medical_KG_rev.observability.metrics import (
    PDF_DOWNLOAD_ATTEMPTS,
    PDF_DOWNLOAD_BYTES,
    PDF_DOWNLOAD_FAILURES,
    PDF_DOWNLOAD_LATENCY,
    set_pdf_download_circuit_state,
    set_pdf_download_failure_rate,
)

from .storage import PdfStorageClient
from .validation import PdfMetadata, PdfUrlValidator

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("Medical_KG_rev.services.pdf.download")


@dataclass(slots=True)
class PdfDownloadRequest:
    tenant_id: str
    document_id: str
    url: str
    correlation_id: str | None = None
    expected_size: int | None = None
    expected_content_type: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class PdfDownloadResult:
    request: PdfDownloadRequest
    storage_key: str
    checksum: str
    content_type: str | None
    size: int
    duration_seconds: float
    attempts: int
    resumed: bool
    pdf_metadata: PdfMetadata | None


class PdfDownloadError(RuntimeError):
    """Exception raised when the download service fails."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        code: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.code = code
        self.metadata = metadata or {}


class DownloadCircuitBreaker:
    """Simple in-memory circuit breaker tracking failures per host."""

    def __init__(self, *, failure_threshold: int = 5, recovery_timeout: float = 300.0) -> None:
        self._failure_threshold = max(1, failure_threshold)
        self._recovery_timeout = max(1.0, recovery_timeout)
        self._failures: dict[str, tuple[int, float]] = {}
        self._lock = Lock()

    def is_open(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            count, opened_at = self._failures.get(key, (0, 0.0))
            if count < self._failure_threshold:
                return False
            if opened_at and now - opened_at >= self._recovery_timeout:
                self._failures.pop(key, None)
                return False
            if not opened_at:
                self._failures[key] = (count, now)
            return True

    def record_failure(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            count, opened_at = self._failures.get(key, (0, 0.0))
            count += 1
            if count >= self._failure_threshold and not opened_at:
                opened_at = now
            self._failures[key] = (count, opened_at)
            return count >= self._failure_threshold

    def record_success(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)


class FailureRateTracker:
    """Rolling failure rate calculator using a bounded deque."""

    def __init__(self, *, window_seconds: float = 300.0, max_samples: int = 50) -> None:
        self._window = max(1.0, window_seconds)
        self._samples: deque[tuple[float, bool]] = deque(maxlen=max(1, max_samples))

    def record(self, success: bool) -> float:
        now = time.time()
        self._samples.append((now, success))
        while self._samples and now - self._samples[0][0] > self._window:
            self._samples.popleft()
        if not self._samples:
            return 0.0
        failures = sum(1 for _, ok in self._samples if not ok)
        return failures / len(self._samples)


class PdfDownloadService:
    """Download PDF documents with retry, tracing, and failure safeguards."""

    def __init__(
        self,
        *,
        storage: PdfStorageClient,
        validator: PdfUrlValidator | None = None,
        timeout: float = 60.0,
        max_attempts: int = 3,
        chunk_size: int = 1024 * 512,
        circuit_breaker: DownloadCircuitBreaker | None = None,
        circuit_failure_threshold: int = 5,
        circuit_recovery_seconds: float = 300.0,
        failure_alert_threshold: float = 0.5,
        failure_window_seconds: float = 300.0,
        backoff_initial: float = 1.0,
        backoff_max: float = 10.0,
    ) -> None:
        self._storage = storage
        self._validator = validator or PdfUrlValidator()
        self._timeout = timeout
        self._max_attempts = max(1, max_attempts)
        self._chunk_size = max(4096, chunk_size)
        self._circuit_breaker = circuit_breaker or DownloadCircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            recovery_timeout=circuit_recovery_seconds,
        )
        self._failure_alert_threshold = max(0.0, min(1.0, failure_alert_threshold))
        self._failure_window = max(1.0, failure_window_seconds)
        self._backoff_initial = max(0.1, backoff_initial)
        self._backoff_max = max(self._backoff_initial, backoff_max)
        self._failure_trackers: dict[str, FailureRateTracker] = {}
        self._open_circuits: set[str] = set()
        self._alert_manager = get_alert_manager()

    def download(self, request: PdfDownloadRequest) -> PdfDownloadResult:
        host = self._extract_host(request.url)
        logger.info(
            "pdf.download.start",
            tenant_id=request.tenant_id,
            document_id=request.document_id,
            url=request.url,
            correlation_id=request.correlation_id,
        )

        with tracer.start_as_current_span(
            "pdf.download",
            attributes={
                "tenant_id": request.tenant_id,
                "document_id": request.document_id,
                "pdf.url": request.url,
            },
        ):
            if self._circuit_breaker.is_open(host):
                PDF_DOWNLOAD_FAILURES.labels("circuit_open").inc()
                set_pdf_download_circuit_state(request.tenant_id, host, True)
                if host not in self._open_circuits:
                    self._open_circuits.add(host)
                    self._alert_manager.circuit_state_changed(
                        f"pdf-download:{host}", "open"
                    )
                raise PdfDownloadError(
                    f"Circuit breaker open for host '{host}'",
                    retryable=False,
                    code="circuit_open",
                )

            metadata: PdfMetadata | None = None
            try:
                metadata = self._validator.validate(request.url)
            except Exception as exc:
                PDF_DOWNLOAD_FAILURES.labels("validation").inc()
                self._record_failure(request.tenant_id, host, success=False)
                self._alert_manager.error_observed("pdf.download", "validation")
                raise PdfDownloadError(
                    f"PDF URL validation failed: {exc}",
                    retryable=False,
                    code="validation",
                ) from exc

            if request.expected_content_type and metadata.content_type:
                if request.expected_content_type.lower() not in metadata.content_type.lower():
                    PDF_DOWNLOAD_FAILURES.labels("content_type_mismatch").inc()
                    self._record_failure(request.tenant_id, host, success=False)
                    raise PdfDownloadError(
                        "PDF content type mismatch: "
                        f"expected {request.expected_content_type}, got {metadata.content_type}",
                        retryable=False,
                        code="content_type_mismatch",
                    )

            attempts = 0
            start_time = time.perf_counter()
            data = bytearray()
            resumed = False
            accept_ranges = metadata.headers.get("Accept-Ranges", "").lower() == "bytes" if metadata else False
            last_error: Exception | None = None

            for attempt in range(1, self._max_attempts + 1):
                attempts = attempt
                try:
                    chunk_count = 0
                    for chunk in self._stream(
                        request.url,
                        offset=len(data) if resumed else 0,
                        accept_ranges=accept_ranges,
                    ):
                        data.extend(chunk)
                        chunk_count += 1
                        if chunk_count % 50 == 0:
                            logger.debug(
                                "pdf.download.progress",
                                tenant_id=request.tenant_id,
                                document_id=request.document_id,
                                bytes=len(data),
                            )
                    break
                except httpx.HTTPError as exc:
                    last_error = exc
                    retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                        isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code >= 500
                    )
                    reason = "http_error"
                    PDF_DOWNLOAD_ATTEMPTS.labels(reason).inc()
                    logger.warning(
                        "pdf.download.retry",
                        attempt=attempt,
                        tenant_id=request.tenant_id,
                        document_id=request.document_id,
                        error=str(exc),
                        retryable=retryable,
                    )
                    if not accept_ranges:
                        data.clear()
                    else:
                        resumed = True
                    if attempt == self._max_attempts:
                        PDF_DOWNLOAD_FAILURES.labels(reason).inc()
                        self._record_failure(request.tenant_id, host, success=False)
                        raise PdfDownloadError(
                            f"Failed to download PDF after {attempt} attempts", retryable=retryable, code=reason
                        ) from exc
                    sleep_for = min(
                        self._backoff_initial * (2 ** (attempt - 1)),
                        self._backoff_max,
                    )
                    time.sleep(sleep_for)
                    continue

            if last_error:
                logger.debug(
                    "pdf.download.recovered",
                    tenant_id=request.tenant_id,
                    document_id=request.document_id,
                    attempts=attempts,
                    resumed=resumed,
                )

            total_bytes = len(data)
            if request.expected_size and total_bytes < request.expected_size:
                PDF_DOWNLOAD_FAILURES.labels("size_mismatch").inc()
                self._record_failure(request.tenant_id, host, success=False)
                raise PdfDownloadError(
                    "Downloaded PDF smaller than expected size "
                    f"({total_bytes} < {request.expected_size})",
                    retryable=False,
                    code="size_mismatch",
                )

            duration = time.perf_counter() - start_time
            key, checksum = self._storage.run(
                self._storage.store(
                    tenant_id=request.tenant_id,
                    document_id=request.document_id,
                    data=bytes(data),
                    content_type=metadata.content_type if metadata else None,
                    metadata={"correlation-id": request.correlation_id or ""} | (request.metadata or {}),
                )
            )

            PDF_DOWNLOAD_LATENCY.observe(duration)
            PDF_DOWNLOAD_BYTES.observe(total_bytes)
            PDF_DOWNLOAD_ATTEMPTS.labels("success").inc()
            self._record_failure(request.tenant_id, host, success=True)
            self._alert_manager.latency_breach("pdf.download", duration * 1000)

            if host in self._open_circuits:
                self._open_circuits.remove(host)
                self._alert_manager.circuit_state_changed(f"pdf-download:{host}", "closed")
            set_pdf_download_circuit_state(request.tenant_id, host, False)

            logger.info(
                "pdf.download.completed",
                stage="download",
                tenant_id=request.tenant_id,
                document_id=request.document_id,
                bytes=total_bytes,
                duration=round(duration, 3),
                key=key,
                attempts=attempts,
                resumed=resumed,
            )

            return PdfDownloadResult(
                request=request,
                storage_key=key,
                checksum=checksum,
                content_type=metadata.content_type if metadata else None,
                size=total_bytes,
                duration_seconds=duration,
                attempts=attempts,
                resumed=resumed,
                pdf_metadata=metadata,
            )

    def _stream(self, url: str, *, offset: int, accept_ranges: bool):
        headers = {
            "User-Agent": "Medical-KG-PdfDownloader/1.0",
            "Accept": "application/pdf,application/octet-stream",
        }
        if offset and accept_ranges:
            headers["Range"] = f"bytes={offset}-"
        with httpx.stream("GET", url, headers=headers, timeout=self._timeout) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes(self._chunk_size):
                if not chunk:
                    continue
                yield chunk

    def _record_failure(self, tenant_id: str, host: str, *, success: bool) -> None:
        tracker = self._failure_trackers.setdefault(
            tenant_id, FailureRateTracker(window_seconds=self._failure_window)
        )
        failure_rate = tracker.record(success)
        set_pdf_download_failure_rate(tenant_id, failure_rate)
        if success:
            self._circuit_breaker.record_success(host)
            return

        if self._circuit_breaker.record_failure(host):
            set_pdf_download_circuit_state(tenant_id, host, True)
            if host not in self._open_circuits:
                self._open_circuits.add(host)
                self._alert_manager.circuit_state_changed(
                    f"pdf-download:{host}", "open"
                )
        if failure_rate >= self._failure_alert_threshold:
            self._alert_manager.download_failure_rate(tenant_id, failure_rate, self._failure_window)

    def _extract_host(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            if parsed.netloc:
                return parsed.netloc.lower()
        except Exception:  # pragma: no cover - defensive guard
            pass
        return "unknown"


__all__ = [
    "DownloadCircuitBreaker",
    "PdfDownloadError",
    "PdfDownloadRequest",
    "PdfDownloadResult",
    "PdfDownloadService",
]
