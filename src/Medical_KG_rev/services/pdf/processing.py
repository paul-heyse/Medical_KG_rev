from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass

import structlog
from opentelemetry import trace

from Medical_KG_rev.observability.alerts import get_alert_manager
from Medical_KG_rev.observability.metrics import (
    PDF_PROCESSING_FAILURES,
    PDF_PROCESSING_LATENCY,
    observe_pdf_processing_duration,
    record_pdf_processing_sla_breach,
)
from Medical_KG_rev.services.mineru.service import MineruGpuUnavailableError, MineruOutOfMemoryError, MineruProcessor
from Medical_KG_rev.services.mineru.types import MineruRequest, MineruResponse

from .gpu_manager import GpuResourceManager
from .storage import PdfStorageClient

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("Medical_KG_rev.services.pdf.mineru")


class MineruProcessingError(RuntimeError):
    """Raised when MinerU processing fails."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        code: str = "unexpected",
        state: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.code = code
        self.state = state or {}


@dataclass(slots=True)
class MineruProcessingResult:
    response: MineruResponse
    checksum: str
    duration_seconds: float


class MineruProcessingService:
    """Execute MinerU processing for downloaded PDF artifacts."""

    def __init__(
        self,
        *,
        processor: MineruProcessor,
        storage: PdfStorageClient,
        gpu_manager: GpuResourceManager | None = None,
        timeout: float = 300.0,
        sla_seconds: float = 240.0,
        enable_state_recovery: bool = True,
    ) -> None:
        self._processor = processor
        self._storage = storage
        self._gpu_manager = gpu_manager or GpuResourceManager()
        self._timeout = max(1.0, timeout)
        self._sla_seconds = max(1.0, sla_seconds)
        self._enable_state_recovery = enable_state_recovery
        self._alert_manager = get_alert_manager()

    def process(
        self,
        *,
        tenant_id: str,
        document_id: str,
        storage_key: str,
        checksum: str,
        correlation_id: str | None = None,
    ) -> MineruProcessingResult:
        logger.info(
            "pdf.mineru.start",
            tenant_id=tenant_id,
            document_id=document_id,
            storage_key=storage_key,
            correlation_id=correlation_id,
        )
        with tracer.start_as_current_span(
            "pdf.mineru.process",
            attributes={
                "tenant_id": tenant_id,
                "document_id": document_id,
                "storage_key": storage_key,
            },
        ):
            pdf_bytes = self._storage.run(self._storage.fetch(storage_key))
            actual_checksum = hashlib.sha256(pdf_bytes).hexdigest()
            if checksum and actual_checksum != checksum:
                PDF_PROCESSING_FAILURES.labels("checksum_mismatch").inc()
                raise MineruProcessingError(
                    f"Checksum mismatch for {storage_key}: expected {checksum}, got {actual_checksum}",
                    retryable=False,
                    code="checksum_mismatch",
                )

            prior_state: dict[str, object] | None = None
            if self._enable_state_recovery:
                prior_state = self._storage.run(
                    self._storage.fetch_processing_state(tenant_id, document_id)
                )
                if prior_state and prior_state.get("status") == "partial":
                    logger.info(
                        "pdf.mineru.recover.partial",
                        tenant_id=tenant_id,
                        document_id=document_id,
                        reason=prior_state.get("reason"),
                    )

            start_time = time.perf_counter()
            try:
                response = self._execute_with_timeout(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    request=MineruRequest(
                        tenant_id=tenant_id, document_id=document_id, content=pdf_bytes
                    ),
                )
            except MineruProcessingError as exc:
                PDF_PROCESSING_FAILURES.labels(exc.code).inc()
                state = {
                    "status": "partial",
                    "reason": str(exc),
                    "retryable": exc.retryable,
                    "code": exc.code,
                    "checksum": actual_checksum,
                }
                state.update(exc.state)
                self._storage.run(
                    self._storage.store_processing_state(tenant_id, document_id, state)
                )
                if exc.code == "timeout":
                    elapsed = time.perf_counter() - start_time
                    self._alert_manager.mineru_timeout(tenant_id, document_id, elapsed)
                self._alert_manager.error_observed("pdf.mineru", exc.code)
                raise
            except (MineruOutOfMemoryError, MineruGpuUnavailableError) as exc:
                PDF_PROCESSING_FAILURES.labels(type(exc).__name__).inc()
                state = {
                    "status": "partial",
                    "reason": str(exc),
                    "retryable": True,
                    "code": type(exc).__name__,
                    "checksum": actual_checksum,
                }
                self._storage.run(
                    self._storage.store_processing_state(tenant_id, document_id, state)
                )
                self._alert_manager.error_observed("pdf.mineru", type(exc).__name__)
                raise
            except Exception as exc:
                PDF_PROCESSING_FAILURES.labels("unexpected").inc()
                self._storage.run(
                    self._storage.store_processing_state(
                        tenant_id,
                        document_id,
                        {
                            "status": "partial",
                            "reason": str(exc),
                            "retryable": False,
                            "code": "unexpected",
                            "checksum": actual_checksum,
                        },
                    )
                )
                self._alert_manager.error_observed("pdf.mineru", "unexpected")
                raise MineruProcessingError(str(exc), retryable=False, code="unexpected") from exc

            duration = time.perf_counter() - start_time
            PDF_PROCESSING_LATENCY.observe(duration)
            size_bucket = self._size_bucket(len(pdf_bytes))
            observe_pdf_processing_duration(size_bucket, duration)
            if duration >= self._sla_seconds:
                record_pdf_processing_sla_breach("mineru")
                self._alert_manager.processing_sla_breach("mineru", duration)

            state = {
                "status": "completed",
                "duration": duration,
                "checksum": actual_checksum,
                "size_bytes": len(pdf_bytes),
                "prior_state": prior_state,
            }
            self._storage.run(
                self._storage.store_processing_state(tenant_id, document_id, state)
            )

            logger.info(
                "pdf.mineru.completed",
                tenant_id=tenant_id,
                document_id=document_id,
                duration=round(duration, 3),
                size=len(pdf_bytes),
            )

            return MineruProcessingResult(
                response=response, checksum=actual_checksum, duration_seconds=duration
            )

    def _execute_with_timeout(
        self,
        *,
        tenant_id: str,
        document_id: str,
        request: MineruRequest,
    ) -> MineruResponse:
        def _run() -> MineruResponse:
            with self._gpu_manager.reserve(tenant_id, document_id):
                return self._processor.process(request)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            try:
                return future.result(timeout=self._timeout)
            except FuturesTimeoutError as exc:
                future.cancel()
                raise MineruProcessingError(
                    f"MinerU processing exceeded timeout of {self._timeout} seconds",
                    retryable=True,
                    code="timeout",
                ) from exc

    def _size_bucket(self, size_bytes: int) -> str:
        mb = size_bytes / (1024 * 1024)
        if mb < 1:
            return "<1mb"
        if mb < 5:
            return "1-5mb"
        if mb < 10:
            return "5-10mb"
        if mb < 20:
            return "10-20mb"
        return ">=20mb"


__all__ = [
    "MineruProcessingError",
    "MineruProcessingResult",
    "MineruProcessingService",
]
