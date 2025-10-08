"""Custom orchestration stage implementations for PDF processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import structlog

from Medical_KG_rev.models.ir import Document as IrDocument
from Medical_KG_rev.observability.metrics import (
    record_pdf_download_failure,
    record_pdf_download_success,
    record_pdf_processing_failure,
)
from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition
from Medical_KG_rev.orchestration.dagster.stage_registry import (
    StageMetadata,
    StageRegistration,
)
from Medical_KG_rev.orchestration.stages.contracts import StageContext
from Medical_KG_rev.services.mineru.processing_service import (
    MineruProcessingError,
    MineruProcessingService,
)
from Medical_KG_rev.services.pdf import PdfDownloadError, PdfDownloadService, PdfStorageClient
from Medical_KG_rev.storage.base import StorageError
from Medical_KG_rev.storage.object_store import (
    FileSystemObjectStore,
    InMemoryObjectStore,
    S3ObjectStore,
)
from Medical_KG_rev.utils.http_client import (
    BackoffStrategy,
    CircuitBreakerConfig,
    HttpClient,
    RateLimitConfig,
    RetryConfig,
)

logger = structlog.get_logger(__name__)


class GateConditionError(RuntimeError):
    """Raised when a gate stage condition fails."""


def _sequence_length(value: Any) -> int:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return 0


def _count_single(value: Any) -> int:
    return 1 if value is not None else 0


def _handle_download_output(state: dict[str, Any], _: str, output: Any) -> None:
    state["downloaded_files"] = output


def _handle_mineru_output(state: dict[str, Any], _: str, output: Any) -> None:
    if hasattr(output, "ir_document"):
        state["document"] = getattr(output, "ir_document")
        state["mineru_metadata"] = getattr(output, "metadata", {})
        state["mineru_duration"] = getattr(output, "duration_seconds", 0.0)
    else:
        state["document"] = output


def _handle_gate_output(state: dict[str, Any], _: str, output: Any) -> None:  # pragma: no cover - no-op
    return None


@dataclass(slots=True)
class PdfDownloadRecord:
    """Represents the outcome of a single PDF download operation."""

    tenant_id: str
    document_id: str
    url: str
    storage_key: str | None
    size_bytes: int | None
    content_type: str | None
    checksum: str | None
    duration_seconds: float
    resumed: bool
    status: str
    error: str | None = None


@dataclass(slots=True)
class PdfDownloadStage:
    """Download PDF resources referenced by parsed documents."""

    name: str
    service: PdfDownloadService
    storage: PdfStorageClient

    def execute(self, ctx: StageContext, document: IrDocument | None) -> list[PdfDownloadRecord]:
        if document is None:
            return []
        pdf_url = self._extract_pdf_url(document)
        if not pdf_url:
            logger.info(
                "dagster.stage.download.skip",
                stage=self.name,
                reason="missing-pdf-url",
                tenant_id=ctx.tenant_id,
                doc_id=document.id,
            )
            return []
        stored = None
        try:
            result = self.service.download(pdf_url, correlation_id=ctx.correlation_id)
            try:
                stored = self.storage.store_pdf(
                    ctx.tenant_id,
                    document.id,
                    result.data,
                    checksum=result.checksum,
                    content_type=result.content_type,
                )
            except StorageError as exc:
                self.storage.cleanup_document(ctx.tenant_id, document.id)
                error = f"Failed to persist PDF: {exc}"
                record_pdf_download_failure(
                    source=document.source or "unknown",
                    error_type="storage-error",
                )
                logger.error(
                    "dagster.stage.download.storage_failed",
                    stage=self.name,
                    tenant_id=ctx.tenant_id,
                    doc_id=document.id,
                    url=pdf_url,
                    error=str(exc),
                )
                return [
                    PdfDownloadRecord(
                        tenant_id=ctx.tenant_id,
                        document_id=document.id,
                        url=pdf_url,
                        storage_key=None,
                        size_bytes=None,
                        content_type=None,
                        checksum=None,
                        duration_seconds=result.duration_seconds,
                        resumed=result.resumed,
                        status="failed",
                        error=error,
                    )
                ]
        except PdfDownloadError as exc:
            error_type = getattr(exc, "error_type", "download-error")
            record_pdf_download_failure(
                source=document.source or "unknown", error_type=error_type
            )
            logger.error(
                "dagster.stage.download.failed",
                stage=self.name,
                tenant_id=ctx.tenant_id,
                doc_id=document.id,
                url=pdf_url,
                error=str(exc),
                error_type=error_type,
            )
            self.storage.cleanup_document(ctx.tenant_id, document.id)
            return [
                PdfDownloadRecord(
                    tenant_id=ctx.tenant_id,
                    document_id=document.id,
                    url=pdf_url,
                    storage_key=None,
                    size_bytes=None,
                    content_type=None,
                    checksum=None,
                    duration_seconds=0.0,
                    resumed=False,
                    status="failed",
                    error=str(exc),
                )
            ]
        except Exception as exc:  # pragma: no cover - defensive cleanup
            if stored:
                self.storage.delete_pdf(stored.key)
            self.storage.cleanup_document(ctx.tenant_id, document.id)
            raise
        if stored is None:
            raise PdfDownloadError("PDF storage unexpectedly returned no artefact")
        record_pdf_download_success(
            source=document.source or "unknown",
            size_bytes=stored.size_bytes,
            duration_seconds=result.duration_seconds,
        )
        logger.info(
            "dagster.stage.download.completed",
            stage=self.name,
            tenant_id=ctx.tenant_id,
            doc_id=document.id,
            url=pdf_url,
            bytes=stored.size_bytes,
        )
        return [
            PdfDownloadRecord(
                tenant_id=ctx.tenant_id,
                document_id=document.id,
                url=pdf_url,
                storage_key=stored.key,
                size_bytes=stored.size_bytes,
                content_type=stored.content_type,
                checksum=stored.checksum,
                duration_seconds=result.duration_seconds,
                resumed=result.resumed,
                status="success",
            )
        ]

    def _extract_pdf_url(self, document: IrDocument) -> str | None:
        if document.pdf_url:
            return str(document.pdf_url)
        metadata_pdf = None
        if isinstance(document.metadata, Mapping):
            metadata_pdf = document.metadata.get("pdf")
        if isinstance(metadata_pdf, Mapping):
            candidate = metadata_pdf.get("url")
            if isinstance(candidate, str):
                return candidate
        return None


@dataclass(slots=True)
class GateCondition:
    key: str
    expected: Any = True


@dataclass(slots=True)
class GateStage:
    """Gate stage validating state conditions before proceeding."""

    name: str
    conditions: tuple[GateCondition, ...]

    def execute(self, ctx: StageContext, upstream: Any) -> None:
        state = upstream if isinstance(upstream, dict) else {"value": upstream}
        for condition in self.conditions:
            value = state
            for part in condition.key.split("."):
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = getattr(value, part, None)
            if value != condition.expected:
                logger.warning(
                    "dagster.stage.gate.blocked",
                    stage=self.name,
                    tenant_id=ctx.tenant_id,
                    key=condition.key,
                    expected=condition.expected,
                    actual=value,
                )
                raise GateConditionError(
                    f"Gate '{self.name}' blocked: expected {condition.key} == {condition.expected!r}"
                )
        logger.debug(
            "dagster.stage.gate.passed",
            stage=self.name,
            tenant_id=ctx.tenant_id,
            conditions=len(self.conditions),
        )


@dataclass(slots=True)
class MineruStage:
    """Run MinerU processing on downloaded PDFs."""

    name: str
    service: MineruProcessingService
    storage: PdfStorageClient

    def execute(self, ctx: StageContext, downloads: list[PdfDownloadRecord]) -> Any:
        record = next((item for item in downloads if item.status == "success" and item.storage_key), None)
        if record is None or not record.storage_key:
            record_pdf_processing_failure(source="mineru", error_type="no-download")
            raise MineruProcessingError("No successful PDF downloads available for processing")
        payload = self.storage.fetch_pdf(record.storage_key)
        try:
            result = self.service.process_pdf(
                tenant_id=ctx.tenant_id,
                document_id=ctx.doc_id or record.document_id,
                pdf_bytes=payload,
                correlation_id=ctx.correlation_id,
                source_label="mineru",
            )
        except MineruProcessingError as exc:
            logger.error(
                "dagster.stage.mineru.failed",
                stage=self.name,
                tenant_id=ctx.tenant_id,
                doc_id=record.document_id,
                storage_key=record.storage_key,
                error=str(exc),
            )
            raise
        logger.info(
            "dagster.stage.mineru.completed",
            stage=self.name,
            tenant_id=ctx.tenant_id,
            doc_id=record.document_id,
            blocks=len(result.ir_document.sections),
        )
        return result


_STORAGE_CLIENTS: dict[str, PdfStorageClient] = {}


def _resolve_storage_client(config: Mapping[str, Any]) -> PdfStorageClient:
    storage_config = config.get("storage") if isinstance(config, Mapping) else None
    name = "default"
    if isinstance(storage_config, Mapping):
        name = str(storage_config.get("name", "default"))
    if name in _STORAGE_CLIENTS:
        return _STORAGE_CLIENTS[name]
    backend_type = "memory"
    base_prefix = "pdfs"
    if isinstance(storage_config, Mapping):
        backend_type = str(storage_config.get("type", "memory")).lower()
        base_prefix = str(storage_config.get("base_prefix", "pdfs"))
    if backend_type in {"s3", "minio"}:
        bucket = storage_config.get("bucket") if isinstance(storage_config, Mapping) else None
        if not bucket:
            raise ValueError("S3 storage backend requires a 'bucket' configuration")
        backend = S3ObjectStore(str(bucket))
    elif backend_type in {"filesystem", "fs", "local"}:
        base_path = storage_config.get("base_path") if isinstance(storage_config, Mapping) else None
        if not base_path:
            raise ValueError("Filesystem storage backend requires a 'base_path' configuration")
        backend = FileSystemObjectStore(str(base_path))
    elif backend_type == "memory":
        backend = InMemoryObjectStore()
    else:
        raise ValueError(f"Unsupported PDF storage backend '{backend_type}'")
    client = PdfStorageClient(backend, base_prefix=base_prefix)
    _STORAGE_CLIENTS[name] = client
    return client


def _create_http_client(config: Mapping[str, Any]) -> HttpClient:
    http_config = config.get("http") if isinstance(config, Mapping) else None
    attempts = 3
    initial_backoff = 0.5
    timeout = 30.0
    rate_limit_config: RateLimitConfig | None = None
    circuit_breaker_config: CircuitBreakerConfig | None = None
    if isinstance(http_config, Mapping):
        attempts = int(http_config.get("attempts", attempts))
        initial_backoff = float(http_config.get("backoff_initial", initial_backoff))
        timeout = float(http_config.get("timeout", timeout))
        rate_limit = http_config.get("rate_limit")
        if isinstance(rate_limit, Mapping):
            rate_limit_config = RateLimitConfig(
                rate_per_second=float(rate_limit.get("rate_per_second", 5.0)),
                burst=rate_limit.get("burst"),
            )
        breaker = http_config.get("circuit_breaker")
        if isinstance(breaker, Mapping):
            circuit_breaker_config = CircuitBreakerConfig(
                failure_threshold=int(breaker.get("failure_threshold", 5)),
                recovery_timeout=float(breaker.get("recovery_timeout", 60.0)),
            )
    retry = RetryConfig(
        attempts=max(1, attempts),
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        backoff_initial=max(0.1, initial_backoff),
        backoff_max=timeout,
        timeout=timeout,
    )
    return HttpClient(
        retry=retry,
        rate_limit=rate_limit_config,
        circuit_breaker=circuit_breaker_config,
    )


def register_download_stage() -> StageRegistration:
    """Register the PDF download stage."""

    def _builder(definition: StageDefinition) -> PdfDownloadStage:
        config = definition.config or {}
        storage = _resolve_storage_client(config)
        http_client = _create_http_client(config)
        service = PdfDownloadService(http_client)
        return PdfDownloadStage(name=definition.name, service=service, storage=storage)

    metadata = StageMetadata(
        stage_type="download",
        state_key="downloaded_files",
        output_handler=_handle_download_output,
        output_counter=_sequence_length,
        description="Downloads PDF resources for downstream MinerU processing",
        dependencies=("parse",),
    )
    return StageRegistration(metadata=metadata, builder=_builder)


def register_gate_stage() -> StageRegistration:
    """Register the built-in gate stage plugin."""

    def _builder(definition: StageDefinition) -> GateStage:
        config = definition.config or {}
        conditions_config = config.get("conditions") or []
        parsed: list[GateCondition] = []
        for entry in conditions_config:
            if isinstance(entry, Mapping):
                key = entry.get("key") or "value"
                parsed.append(GateCondition(key=str(key), expected=entry.get("expected", True)))
            elif isinstance(entry, str):
                parsed.append(GateCondition(key=entry, expected=True))
        if not parsed:
            parsed.append(GateCondition(key="value", expected=True))
        return GateStage(name=definition.name, conditions=tuple(parsed))

    metadata = StageMetadata(
        stage_type="gate",
        state_key=None,
        output_handler=_handle_gate_output,
        output_counter=lambda _: 0,
        description="Halts pipeline execution until configured conditions are met",
        dependencies=("download",),
    )
    return StageRegistration(metadata=metadata, builder=_builder)


def register_mineru_stage() -> StageRegistration:
    """Register the MinerU PDF processing stage."""

    def _builder(definition: StageDefinition) -> MineruStage:
        config = definition.config or {}
        storage = _resolve_storage_client(config)
        mineru_config = config.get("mineru") if isinstance(config, Mapping) else {}
        service = MineruProcessingService(config=mineru_config or {})
        return MineruStage(name=definition.name, service=service, storage=storage)

    metadata = StageMetadata(
        stage_type="mineru",
        state_key="document",
        output_handler=_handle_mineru_output,
        output_counter=_count_single,
        description="Processes downloaded PDFs through MinerU to produce IR documents",
        dependencies=("download",),
    )
    return StageRegistration(metadata=metadata, builder=_builder)


__all__ = [
    "GateCondition",
    "GateConditionError",
    "GateStage",
    "PdfDownloadRecord",
    "PdfDownloadStage",
    "MineruStage",
    "register_download_stage",
    "register_gate_stage",
    "register_mineru_stage",
]
