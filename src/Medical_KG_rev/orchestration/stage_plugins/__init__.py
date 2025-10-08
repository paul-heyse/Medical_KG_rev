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
from Medical_KG_rev.orchestration.kafka import KafkaClient
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.stages.contracts import StageContext
from Medical_KG_rev.services.mineru.service import (
    MineruGpuUnavailableError,
    MineruOutOfMemoryError,
    MineruProcessor,
)
from Medical_KG_rev.services.pdf import (
    DownloadCircuitBreaker,
    GpuResourceManager,
    MineruProcessingError,
    MineruProcessingResult,
    MineruProcessingService,
    PdfDeadLetterQueue,
    PdfDownloadError,
    PdfDownloadRequest,
    PdfDownloadResult,
    PdfDownloadService,
    PdfStorageClient,
    PdfStorageConfig,
    PdfUrlValidator,
)
from Medical_KG_rev.storage.object_store import (
    InMemoryObjectStore,
    LocalFileObjectStore,
    S3ObjectStore,
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
    state["download_artifact"] = output


def _handle_mineru_output(state: dict[str, Any], _: str, output: Any) -> None:
    if hasattr(output, "ir_document"):
        state["document"] = getattr(output, "ir_document")
        state["mineru_metadata"] = getattr(output, "metadata", {})
        state["mineru_duration"] = getattr(output, "duration_seconds", 0.0)
    else:
        state["document"] = output


def _handle_gate_output(state: dict[str, Any], _: str, output: Any) -> None:  # pragma: no cover - no-op
    return None


def _handle_mineru_output(state: dict[str, Any], _: str, output: Any) -> None:
    state["mineru_result"] = output
    ir_document = None
    if isinstance(output, MineruProcessingResult):
        ir_document = getattr(getattr(output.response, "document", None), "ir_document", None)
    if isinstance(ir_document, Document):
        state["document"] = ir_document


_STORAGE_CACHE: dict[str, PdfStorageClient] = {}


def _storage_cache_key(config: Mapping[str, Any]) -> str:
    storage = config.get("storage") or {}
    backend = str(storage.get("backend", "memory")).lower()
    alias = storage.get("alias") or "default"
    if backend == "local":
        path = storage.get("path") or "/tmp/medicalkg/pdf"
        return f"local:{alias}:{path}"
    if backend == "s3":
        bucket = storage.get("bucket")
        return f"s3:{alias}:{bucket}"
    return f"memory:{alias}"


def _get_storage_client(config: Mapping[str, Any]) -> PdfStorageClient:
    key = _storage_cache_key(config)
    if key in _STORAGE_CACHE:
        return _STORAGE_CACHE[key]
    storage_cfg = config.get("storage") or {}
    backend_name = str(storage_cfg.get("backend", "memory")).lower()
    if backend_name == "local":
        base_path = storage_cfg.get("path") or "/tmp/medicalkg/pdf"
        backend = LocalFileObjectStore(base_path)
    elif backend_name == "s3":
        bucket = storage_cfg.get("bucket")
        if not bucket:
            raise ValueError("S3 storage backend requires a 'bucket' value")
        backend = S3ObjectStore(bucket=str(bucket))
    else:
        backend = InMemoryObjectStore()
    client = PdfStorageClient(
        backend=backend,
        config=PdfStorageConfig(
            base_prefix=str(storage_cfg.get("base_prefix", "pdf")),
            enable_access_logging=bool(storage_cfg.get("access_logging", True)),
        ),
    )
    _STORAGE_CACHE[key] = client
    return client


@dataclass(slots=True)
class DownloadStage:
    """Download PDFs referenced by documents and persist them to storage."""

    name: str
    download_service: PdfDownloadService
    storage: PdfStorageClient
    ledger: JobLedger | None = None
    dead_letter_queue: PdfDeadLetterQueue | None = None

    def bind_runtime(self, *, ledger: JobLedger, kafka: KafkaClient | None = None) -> None:
        self.ledger = ledger
        if kafka:
            self.dead_letter_queue = PdfDeadLetterQueue(kafka)

    def execute(self, ctx: StageContext, upstream: Any) -> PdfDownloadResult | None:
        document = self._resolve_document(upstream)
        if document is None:
            logger.warning(
                "dagster.stage.download.missing_document",
                stage=self.name,
                tenant_id=ctx.tenant_id,
            )
            return None
        pdf_url = document.pdf_url or document.metadata.get("pdf_url")
        if not pdf_url:
            if self.ledger and ctx.job_id:
                self.ledger.update_pdf_state(
                    ctx.job_id,
                    history_status="skipped",
                    detail="Document does not specify a pdf_url",
                )
            logger.info(
                "dagster.stage.download.skipped",
                stage=self.name,
                tenant_id=ctx.tenant_id,
                document_id=document.id,
            )
            return None
        request = PdfDownloadRequest(
            tenant_id=ctx.tenant_id,
            document_id=document.id,
            url=str(pdf_url),
            correlation_id=ctx.correlation_id,
            expected_size=document.pdf_size,
            expected_content_type=document.pdf_content_type,
            metadata={"source": document.source},
        )
        try:
            result = self.download_service.download(request)
        except PdfDownloadError as exc:
            if self.ledger and ctx.job_id:
                self.ledger.record_pdf_failure(
                    ctx.job_id,
                    stage=self.name,
                    reason=str(exc),
                    retryable=exc.retryable,
                    code=exc.code,
                )
                if not exc.retryable:
                    self.ledger.rollback_pdf_state(ctx.job_id, reason=str(exc))
            self.storage.run(
                self.storage.cleanup_document(ctx.tenant_id, document.id)
            )
            if (
                self.dead_letter_queue
                and ctx.job_id
                and not exc.retryable
            ):
                self.dead_letter_queue.publish(
                    job_id=ctx.job_id,
                    tenant_id=ctx.tenant_id,
                    document_id=document.id,
                    stage=self.name,
                    reason=exc.code or "download_error",
                    payload={"url": request.url},
                )
            raise
        if self.ledger and ctx.job_id:
            self.ledger.clear_pdf_error(ctx.job_id)
            self.ledger.set_pdf_downloaded(
                ctx.job_id,
                True,
                url=result.request.url,
                storage_key=result.storage_key,
                size=result.size,
                content_type=result.content_type,
                checksum=result.checksum,
            )
        logger.info(
            "dagster.stage.download.completed",
            stage=self.name,
            tenant_id=ctx.tenant_id,
            document_id=document.id,
            bytes=result.size,
        )
        return result

    def _resolve_document(self, upstream: Any) -> Document | None:
        if isinstance(upstream, Document):
            return upstream
        if isinstance(upstream, Mapping):
            candidate = upstream.get("document")
            if isinstance(candidate, Document):
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
    """Run MinerU over downloaded PDFs and update ledger state."""

    name: str
    processing_service: MineruProcessingService
    ledger: JobLedger | None = None
    dead_letter_queue: PdfDeadLetterQueue | None = None

    def bind_runtime(self, *, ledger: JobLedger, kafka: KafkaClient | None = None) -> None:
        self.ledger = ledger
        if kafka:
            self.dead_letter_queue = PdfDeadLetterQueue(kafka)

    def execute(self, ctx: StageContext, upstream: Any) -> MineruProcessingResult | None:
        document = None
        download_result = None
        if isinstance(upstream, Mapping):
            candidate = upstream.get("document")
            if isinstance(candidate, Document):
                document = candidate
            artifact = upstream.get("download_artifact") or upstream.get("downloaded_files")
            if isinstance(artifact, list):
                artifact = artifact[0] if artifact else None
            if isinstance(artifact, PdfDownloadResult):
                download_result = artifact
        elif isinstance(upstream, PdfDownloadResult):
            download_result = upstream
        if download_result is None:
            if self.ledger and ctx.job_id:
                self.ledger.record_pdf_failure(
                    ctx.job_id,
                    stage=self.name,
                    reason="PDF download artifact missing",
                    retryable=False,
                    code="missing_artifact",
                )
                self.ledger.rollback_pdf_state(ctx.job_id, reason="missing download artifact")
            if self.dead_letter_queue and ctx.job_id:
                self.dead_letter_queue.publish(
                    job_id=ctx.job_id,
                    tenant_id=ctx.tenant_id,
                    document_id=document.id if document else "unknown",
                    stage=self.name,
                    reason="missing_artifact",
                    payload={},
                )
            logger.warning(
                "dagster.stage.mineru.missing_artifact",
                stage=self.name,
                tenant_id=ctx.tenant_id,
            )
            return None
        try:
            result = self.processing_service.process(
                tenant_id=ctx.tenant_id,
                document_id=download_result.request.document_id,
                storage_key=download_result.storage_key,
                checksum=download_result.checksum,
                correlation_id=ctx.correlation_id,
            )
        except (MineruOutOfMemoryError, MineruGpuUnavailableError) as exc:
            if self.ledger and ctx.job_id:
                self.ledger.record_pdf_failure(
                    ctx.job_id,
                    stage=self.name,
                    reason=str(exc),
                    retryable=True,
                    code=type(exc).__name__,
                )
                self.ledger.record_pdf_partial(
                    ctx.job_id,
                    stage=self.name,
                    detail=str(exc),
                    retryable=True,
                )
            raise
        except MineruProcessingError as exc:
            if self.ledger and ctx.job_id:
                self.ledger.record_pdf_failure(
                    ctx.job_id,
                    stage=self.name,
                    reason=str(exc),
                    retryable=exc.retryable,
                    code=exc.code,
                )
                if exc.retryable:
                    self.ledger.record_pdf_partial(
                        ctx.job_id,
                        stage=self.name,
                        detail=str(exc),
                        retryable=True,
                    )
                else:
                    self.ledger.rollback_pdf_state(ctx.job_id, reason=str(exc))
            if (
                self.dead_letter_queue
                and ctx.job_id
                and not exc.retryable
            ):
                self.dead_letter_queue.publish(
                    job_id=ctx.job_id,
                    tenant_id=ctx.tenant_id,
                    document_id=download_result.request.document_id,
                    stage=self.name,
                    reason=exc.code or "mineru_error",
                    payload={"storage_key": download_result.storage_key},
                )
            raise
        if self.ledger and ctx.job_id:
            self.ledger.clear_pdf_error(ctx.job_id)
            self.ledger.set_pdf_ir_ready(
                ctx.job_id,
                True,
                checksum=result.checksum,
            )
        logger.info(
            "dagster.stage.mineru.completed",
            stage=self.name,
            tenant_id=ctx.tenant_id,
            document_id=download_result.request.document_id,
            duration=round(result.duration_seconds, 3),
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
        storage = _get_storage_client(config)
        validator_cfg = config.get("validator") or {}
        validator = PdfUrlValidator(
            timeout=float(validator_cfg.get("timeout", 15.0)),
            max_attempts=int(validator_cfg.get("max_attempts", 3)),
        )
        breaker_cfg = config.get("circuit_breaker") or {}
        circuit_breaker = DownloadCircuitBreaker(
            failure_threshold=int(breaker_cfg.get("failure_threshold", 5)),
            recovery_timeout=float(breaker_cfg.get("recovery_timeout_seconds", 300.0)),
        )
        download_service = PdfDownloadService(
            storage=storage,
            validator=validator,
            timeout=float(config.get("timeout", 60.0)),
            max_attempts=int(config.get("max_attempts", 3)),
            chunk_size=int(config.get("chunk_size", 512 * 1024)),
            circuit_breaker=circuit_breaker,
            failure_alert_threshold=float(config.get("failure_alert_threshold", 0.5)),
            failure_window_seconds=float(config.get("failure_window_seconds", 300.0)),
            backoff_initial=float(config.get("backoff_initial", 1.0)),
            backoff_max=float(config.get("backoff_max", 10.0)),
        )
        return DownloadStage(
            name=definition.name,
            download_service=download_service,
            storage=storage,
        )

    metadata = StageMetadata(
        stage_type="download",
        state_key="download_artifact",
        output_handler=_handle_download_output,
        output_counter=lambda output: 1 if output else 0,
        description="Downloads PDF resources referenced by documents",
        dependencies=("parse",),
    )
    return StageRegistration(metadata=metadata, builder=_builder)


def register_mineru_stage() -> StageRegistration:
    """Register the MinerU processing stage plugin."""

    def _builder(definition: StageDefinition) -> MineruStage:
        config = definition.config or {}
        storage = _get_storage_client(config)
        gpu_cfg = config.get("gpu") or {}
        gpu_manager = GpuResourceManager(
            max_concurrent=gpu_cfg.get("max_concurrent"),
            gpu_memory_mb=gpu_cfg.get("memory_mb"),
        )
        processor = MineruProcessor()
        processing_cfg = config.get("processing") or {}
        processing_service = MineruProcessingService(
            processor=processor,
            storage=storage,
            gpu_manager=gpu_manager,
            timeout=float(processing_cfg.get("timeout", config.get("timeout", 300.0))),
            sla_seconds=float(processing_cfg.get("sla_seconds", 240.0)),
            enable_state_recovery=bool(processing_cfg.get("stateful_recovery", True)),
        )
        return MineruStage(name=definition.name, processing_service=processing_service)

    metadata = StageMetadata(
        stage_type="mineru",
        state_key="mineru_result",
        output_handler=_handle_mineru_output,
        output_counter=lambda output: 1 if output else 0,
        description="Processes downloaded PDFs with MinerU to produce structured output",
        dependencies=("download", "parse"),
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
