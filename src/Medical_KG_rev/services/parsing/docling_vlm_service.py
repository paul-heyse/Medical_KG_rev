"""Docling Gemma3 VLM integration for PDF processing."""

from __future__ import annotations

import random
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import structlog

from Medical_KG_rev.config.docling_config import DoclingVLMConfig
from Medical_KG_rev.services.gpu.manager import GpuManager, GpuNotAvailableError
from Medical_KG_rev.services.health import CheckResult, failure, success
from Medical_KG_rev.services.mineru.types import (
    Block,
    Document,
    MineruBatchResponse,
    MineruRequest,
    MineruResponse,
    ProcessingMetadata,
)

from .exceptions import (
    DoclingCircuitBreakerOpenError,
    DoclingModelLoadError,
    DoclingProcessingError,
    DoclingVLMError,
)
from .metrics import (
    DOCLING_VLM_GPU_MEMORY_MB,
    DOCLING_VLM_MODEL_LOAD_SECONDS,
    DOCLING_VLM_PROCESSING_SECONDS,
    DOCLING_VLM_REQUESTS_TOTAL,
    DOCLING_VLM_RETRY_ATTEMPTS_TOTAL,
)

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class DoclingVLMResult:
    """Single document response produced by the Docling VLM backend."""

    document: Document
    processed_at: datetime
    duration_seconds: float
    metadata: ProcessingMetadata


class DoclingVLMService:
    """High-level façade over the Docling Gemma3 VLM pipeline."""

    CIRCUIT_FAILURE_THRESHOLD = 3
    CIRCUIT_RESET_SECONDS = 180

    def __init__(
        self,
        config: DoclingVLMConfig | None = None,
        *,
        gpu_manager: GpuManager | None = None,
    ) -> None:
        self._config = config or DoclingVLMConfig()
        self._gpu_manager = gpu_manager or GpuManager(
            min_memory_mb=self._config.required_gpu_memory_mb()
        )
        self._converter = None
        self._lock = threading.Lock()
        self._failure_count = 0
        self._circuit_open_until: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, request: MineruRequest) -> MineruResponse:
        """Process a single PDF using the Docling backend."""

        self._assert_circuit_closed()
        pdf_path, cleanup = self._resolve_request_path(request)
        try:
            result = self._process_path(
                tenant_id=request.tenant_id,
                document_id=request.document_id,
                pdf_path=pdf_path,
            )
        except DoclingVLMError:
            self._record_failure()
            raise
        else:
            self._record_success()
        finally:
            if cleanup:
                pdf_path.unlink(missing_ok=True)
        return MineruResponse(
            document=result.document,
            processed_at=result.processed_at,
            duration_seconds=result.duration_seconds,
            metadata=result.metadata,
        )

    def process_batch(self, batch: Sequence[MineruRequest]) -> MineruBatchResponse:
        """Process a batch of PDFs using the Docling backend."""

        started_at = datetime.now(tz=UTC)
        documents: list[Document] = []
        metadata: list[ProcessingMetadata] = []
        for request in batch:
            try:
                response = process_pdf_with_retries(
                    self,
                    request,
                    attempts=max(1, self._config.retry_attempts),
                )
            except DoclingCircuitBreakerOpenError as exc:
                DOCLING_VLM_REQUESTS_TOTAL.labels(outcome="error").inc()
                logger.error(
                    "docling.vlm.batch.circuit_open",
                    document_id=request.document_id,
                    tenant_id=request.tenant_id,
                    error=str(exc),
                )
                break
            except DoclingVLMError as exc:
                DOCLING_VLM_REQUESTS_TOTAL.labels(outcome="error").inc()
                logger.error(
                    "docling.vlm.batch.error",
                    document_id=request.document_id,
                    tenant_id=request.tenant_id,
                    error=str(exc),
                )
                continue
            documents.append(response.document)
            metadata.append(response.metadata)
        completed_at = datetime.now(tz=UTC)
        duration = (completed_at - started_at).total_seconds()
        return MineruBatchResponse(
            documents=documents,
            processed_at=completed_at,
            duration_seconds=duration,
            metadata=metadata,
        )

    def health(self) -> CheckResult:
        """Expose a readiness check compatible with the gateway health service."""

        model_path = self._config.resolved_model_path
        if not model_path.exists():
            return failure(f"Gemma3 checkpoint missing at {model_path}")
        if self._circuit_open_until is not None:
            remaining = self._circuit_open_until - time.monotonic()
            if remaining > 0:
                return failure(f"Circuit breaker open for another {remaining:.0f}s")
        ok, snapshot, detail = self._gpu_manager.health_status(
            self._config.required_gpu_memory_mb()
        )
        if not ok:
            return failure(detail or "GPU unavailable for Docling")
        extra = (
            f"Docling ready on cuda device with {snapshot.get('free_mb', 0)} MB free"
        )
        return success(extra)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_request_path(self, request: MineruRequest) -> tuple[Path, bool]:
        if request.content:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(request.content)
            tmp.flush()
            tmp.close()
            return Path(tmp.name), True
        if not request.storage_uri:
            msg = "DoclingVLMService requires a storage URI when content bytes are absent"
            raise DoclingProcessingError(msg)
        return Path(request.storage_uri), False

    def _ensure_converter(self):
        if self._converter is not None:
            return self._converter
        with self._lock:
            if self._converter is not None:
                return self._converter
            start = time.perf_counter()
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import (
                    EasyOcrOptions,
                    PdfPipelineOptions,
                )
                from docling.document_converter import (
                    DocumentConverter,
                    PdfFormatOption,
                )
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise DoclingModelLoadError("docling is not installed") from exc

            model_path = self._config.resolved_model_path
            if not model_path.exists():
                raise DoclingModelLoadError(
                    f"Docling Gemma3 checkpoint not found at {model_path}"
                )

            options = PdfPipelineOptions(
                artifacts_path=str(self._config.resolved_model_path),
                do_table_structure=True,
                do_ocr=True,
                ocr_options=EasyOcrOptions(use_gpu=True),
            )
            pdf_option = PdfFormatOption(pipeline_options=options)
            converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                format_options={InputFormat.PDF: pdf_option},
            )
            elapsed = time.perf_counter() - start
            DOCLING_VLM_MODEL_LOAD_SECONDS.labels(
                self._config.revision or "local"
            ).observe(elapsed)
            self._converter = converter
            return converter

    def _process_path(self, *, tenant_id: str, document_id: str, pdf_path: Path) -> DoclingVLMResult:
        converter = self._ensure_converter()
        started_at = datetime.now(tz=UTC)
        timer = time.perf_counter()
        try:
            with self._gpu_manager.device_session(
                "docling_vlm",
                required_memory_mb=self._config.required_gpu_memory_mb(),
                warmup=True,
            ) as device:
                snapshot_before = self._gpu_manager.memory_snapshot(device)
                logger.info(
                    "docling.vlm.processing.start",
                    document_id=document_id,
                    tenant_id=tenant_id,
                    device=f"cuda:{device.index}",
                    required_memory_mb=self._config.required_gpu_memory_mb(),
                    free_memory_mb=snapshot_before.get("free_mb"),
                )
                conversion = converter.convert(str(pdf_path))
                snapshot_after = self._gpu_manager.memory_snapshot(device)
                DOCLING_VLM_GPU_MEMORY_MB.labels(
                    device=f"cuda:{device.index}",
                    state="reserved",
                ).set(self._config.required_gpu_memory_mb())
                DOCLING_VLM_GPU_MEMORY_MB.labels(
                    device=f"cuda:{device.index}",
                    state="free",
                ).set(snapshot_after.get("free_mb", 0))
        except GpuNotAvailableError as exc:
            DOCLING_VLM_REQUESTS_TOTAL.labels(outcome="error").inc()
            raise DoclingProcessingError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - docling runtime errors
            DOCLING_VLM_REQUESTS_TOTAL.labels(outcome="error").inc()
            raise DoclingProcessingError(
                f"Docling failed to process {pdf_path}: {exc}"
            ) from exc

        duration = time.perf_counter() - timer
        completed_at = datetime.now(tz=UTC)
        metadata = self._build_metadata(
            tenant_id=tenant_id,
            document_id=document_id,
            duration=duration,
            started_at=started_at,
            completed_at=completed_at,
        )
        document = self._map_docling_to_ir(
            tenant_id=tenant_id,
            document_id=document_id,
            conversion=conversion,
        )

        DOCLING_VLM_PROCESSING_SECONDS.labels(
            tenant_id=tenant_id,
            outcome="success",
        ).observe(duration)
        DOCLING_VLM_REQUESTS_TOTAL.labels(outcome="success").inc()
        return DoclingVLMResult(
            document=document,
            processed_at=completed_at,
            duration_seconds=duration,
            metadata=metadata,
        )

    def _assert_circuit_closed(self) -> None:
        window = self._circuit_open_until
        if window is None:
            return
        remaining = window - time.monotonic()
        if remaining <= 0:
            self._circuit_open_until = None
            self._failure_count = 0
            return
        raise DoclingCircuitBreakerOpenError(
            f"Docling circuit breaker open for another {remaining:.0f}s"
        )

    def _record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= self.CIRCUIT_FAILURE_THRESHOLD:
                self._circuit_open_until = time.monotonic() + self.CIRCUIT_RESET_SECONDS
                logger.error(
                    "docling.vlm.circuit.opened",
                    failure_count=self._failure_count,
                    reset_seconds=self.CIRCUIT_RESET_SECONDS,
                )

    def _record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._circuit_open_until = None

    def _build_metadata(
        self,
        *,
        tenant_id: str,
        document_id: str,
        duration: float,
        started_at: datetime,
        completed_at: datetime,
    ) -> ProcessingMetadata:
        return ProcessingMetadata(
            document_id=document_id,
            mineru_version="docling-vlm",
            model_names={
                "vlm": "gemma3-12b",
                "revision": self._config.revision or "local",
            },
            gpu_id=None,
            worker_id=None,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            cli_stdout="docling_vlm",
            cli_stderr="",
            cli_descriptor=f"docling-vlm:{tenant_id}",
            planned_memory_mb=self._config.required_gpu_memory_mb(),
        )

    def _map_docling_to_ir(
        self,
        *,
        tenant_id: str,
        document_id: str,
        conversion,
    ) -> Document:
        doc = Document(document_id=document_id, tenant_id=tenant_id)
        doc.metadata["docling_status"] = getattr(conversion.status, "value", "unknown")
        output = getattr(conversion, "output", None)
        pages = getattr(output, "pages", []) if output else []
        doc.metadata["page_count"] = len(pages)
        for index, page in enumerate(pages, start=1):
            elements = getattr(page, "elements", []) or []
            texts = []
            for element in elements:
                text = getattr(element, "text", None)
                if text:
                    texts.append(text)
            if not texts:
                continue
            doc.blocks.append(
                Block(
                    id=f"{document_id}-page-{index}",
                    page=index,
                    kind="text",
                    text="\n".join(texts),
                    bbox=None,
                    confidence=None,
                    reading_order=index,
                    metadata={"source": "docling"},
                )
            )
        return doc


def process_pdf_with_retries(
    service: DoclingVLMService,
    request: MineruRequest,
    *,
    attempts: int,
    base_backoff_seconds: float = 2.0,
    max_backoff_seconds: float = 30.0,
) -> MineruResponse:
    """Execute Docling processing with retry semantics."""

    last_error: DoclingVLMError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return service.process(request)
        except DoclingCircuitBreakerOpenError:
            raise
        except DoclingVLMError as exc:
            last_error = exc
            DOCLING_VLM_RETRY_ATTEMPTS_TOTAL.labels(reason=exc.__class__.__name__).inc()
            logger.warning(
                "docling.vlm.retry",
                attempt=attempt,
                remaining=attempts - attempt,
                error=str(exc),
                document_id=request.document_id,
                tenant_id=request.tenant_id,
            )
            if attempt < attempts:
                sleep_for = min(
                    max_backoff_seconds,
                    base_backoff_seconds * (2 ** (attempt - 1)),
                )
                jitter = random.uniform(0.1, 0.3)
                delay = sleep_for * (1 + jitter)
                logger.debug(
                    "docling.vlm.retry.sleep",
                    attempt=attempt,
                    sleep_seconds=round(delay, 2),
                    document_id=request.document_id,
                )
                time.sleep(delay)
    assert last_error is not None  # pragma: no cover - defensive assertion
    raise last_error

