"""High-level MinerU processing service used by orchestration stages."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import structlog

from Medical_KG_rev.config.settings import MineruSettings, get_settings
from Medical_KG_rev.models.ir import Document as IrDocument
from Medical_KG_rev.observability.metrics import (
    get_gpu_utilisation_snapshot,
    record_pdf_processing_failure,
    record_pdf_processing_success,
)
from Medical_KG_rev.services.mineru.cli_wrapper import MineruCliError, SimulatedMineruCli
from Medical_KG_rev.services.mineru.output_parser import MineruOutputParserError
from Medical_KG_rev.services.mineru.service import (
    MineruGpuUnavailableError,
    MineruOutOfMemoryError,
    MineruProcessor,
)
from Medical_KG_rev.services.mineru.types import MineruRequest

logger = structlog.get_logger(__name__)


class MineruProcessingError(RuntimeError):
    """Raised when MinerU processing fails."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "processing-error",
        recorded: bool = False,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.recorded = recorded


@dataclass(slots=True)
class MineruProcessingResult:
    """Structured response returned after MinerU processing completes."""

    ir_document: IrDocument
    duration_seconds: float
    metadata: Mapping[str, Any]


class MineruProcessingService:
    """Wrapper around :class:`MineruProcessor` with metric emission."""

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        simulate = bool(config.get("simulate", True))
        source_label = config.get("source") or "mineru"
        settings = config.get("settings")
        if settings is None:
            settings = get_settings().mineru
        elif isinstance(settings, dict):
            settings = MineruSettings(**settings)
        self._settings = settings
        self._default_source = str(source_label)
        timeout_override = config.get("timeout_seconds")
        self._timeout_seconds = (
            float(timeout_override)
            if timeout_override is not None
            else float(self._settings.cli_timeout_seconds())
        )
        self._require_gpu = bool(config.get("require_gpu", False))
        if simulate:
            cli = SimulatedMineruCli(self._settings)
            self._processor = MineruProcessor(settings=self._settings, cli=cli)
        else:
            self._processor = MineruProcessor(settings=self._settings)
        logger.info(
            "mineru.processing_service.initialised",
            simulate=simulate,
            source_label=self._default_source,
        )

    def process_pdf(
        self,
        *,
        tenant_id: str,
        document_id: str,
        pdf_bytes: bytes,
        correlation_id: str | None = None,
        source_label: str | None = None,
    ) -> MineruProcessingResult:
        label = source_label or self._default_source
        request = MineruRequest(tenant_id=tenant_id, document_id=document_id, content=pdf_bytes)
        pdf_size = len(pdf_bytes)
        start = perf_counter()
        try:
            response = self._run_with_timeout(request)
        except MineruProcessingError as exc:
            if not getattr(exc, "recorded", False):
                record_pdf_processing_failure(source=label, error_type=exc.error_type)
            logger.error(
                "mineru.processing.failed",
                document_id=document_id,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
                error=str(exc),
                error_type=exc.error_type,
            )
            raise
        except MineruOutOfMemoryError as exc:
            record_pdf_processing_failure(source=label, error_type="gpu-oom")
            logger.error(
                "mineru.processing.gpu_oom",
                document_id=document_id,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
                error=str(exc),
            )
            raise MineruProcessingError(str(exc), error_type="gpu-oom", recorded=True) from exc
        except MineruGpuUnavailableError as exc:
            record_pdf_processing_failure(source=label, error_type="gpu-unavailable")
            logger.error(
                "mineru.processing.gpu_unavailable",
                document_id=document_id,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
                error=str(exc),
            )
            raise MineruProcessingError(str(exc), error_type="gpu-unavailable", recorded=True) from exc
        except MineruCliError as exc:
            error_type = self._classify_cli_error(str(exc))
            record_pdf_processing_failure(source=label, error_type=error_type)
            logger.error(
                "mineru.processing.cli_error",
                document_id=document_id,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
                error=str(exc),
                error_type=error_type,
            )
            raise MineruProcessingError(str(exc), error_type=error_type, recorded=True) from exc
        except MineruOutputParserError as exc:
            record_pdf_processing_failure(source=label, error_type="parse-error")
            logger.error(
                "mineru.processing.parse_error",
                document_id=document_id,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
                error=str(exc),
            )
            raise MineruProcessingError(str(exc), recorded=True) from exc
        except Exception as exc:  # pragma: no cover - unexpected guard
            record_pdf_processing_failure(source=label, error_type="unexpected")
            logger.exception(
                "mineru.processing.unexpected_error",
                document_id=document_id,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
            )
            raise MineruProcessingError(str(exc)) from exc
        ir_document = response.document.ir_document
        if ir_document is None:
            record_pdf_processing_failure(source=label, error_type="missing-ir")
            raise MineruProcessingError(
                "MinerU response did not include an IR document",
                error_type="missing-ir",
                recorded=True,
            )
        duration_seconds = response.duration_seconds or (perf_counter() - start)
        gpu_snapshot = get_gpu_utilisation_snapshot()
        if self._require_gpu and not gpu_snapshot:
            record_pdf_processing_failure(source=label, error_type="gpu-unavailable")
            raise MineruProcessingError(
                "GPU was required but unavailable during MinerU processing",
                error_type="gpu-unavailable",
                recorded=True,
            )
        record_pdf_processing_success(
            source=label,
            duration_seconds=duration_seconds,
            size_bytes=pdf_size,
            gpu_snapshot=gpu_snapshot,
        )
        metadata = response.metadata.as_dict() if hasattr(response.metadata, "as_dict") else {}
        return MineruProcessingResult(
            ir_document=ir_document,
            duration_seconds=duration_seconds,
            metadata=metadata,
        )


    def _run_with_timeout(self, request: MineruRequest):
        if self._timeout_seconds <= 0:
            return self._processor.process(request)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._processor.process, request)
            try:
                return future.result(timeout=self._timeout_seconds)
            except FuturesTimeout as exc:
                future.cancel()
                raise MineruProcessingError(
                    f"MinerU processing exceeded {self._timeout_seconds:.1f}s timeout",
                    error_type="timeout",
                ) from exc

    @staticmethod
    def _classify_cli_error(message: str) -> str:
        lowered = message.lower()
        if "timeout" in lowered:
            return "timeout"
        if "out of memory" in lowered or "oom" in lowered:
            return "gpu-oom"
        if "gpu" in lowered and "available" in lowered:
            return "gpu-unavailable"
        return "cli-error"


__all__ = [
    "MineruProcessingError",
    "MineruProcessingResult",
    "MineruProcessingService",
]
