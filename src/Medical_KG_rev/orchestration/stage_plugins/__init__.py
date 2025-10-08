from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import structlog

from Medical_KG_rev.observability.metrics import (
    record_gate_event,
    record_gate_wait_duration,
    record_pdf_download_bytes,
    record_pdf_download_event,
    record_pdf_download_duration,
)
from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition
from Medical_KG_rev.orchestration.dagster.stage_registry import (
    StageMetadata,
    StageRegistration,
)
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.stages.contracts import StageContext
from Medical_KG_rev.utils.http_client import BackoffStrategy, HttpClient, RetryConfig

logger = structlog.get_logger(__name__)


class GateConditionError(RuntimeError):
    """Raised when a gate stage condition fails or times out."""


def _sequence_length(value: Any) -> int:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return 0


def _count_single(value: Any) -> int:
    return 1 if value is not None else 0


def _handle_download_output(state: dict[str, Any], _: str, output: Any) -> None:
    state["download"] = output


def _handle_gate_output(state: dict[str, Any], _: str, output: Any) -> None:  # pragma: no cover - gate returns metadata only
    state.setdefault("gates", {})
    if isinstance(output, Mapping):
        state["gates"][output.get("gate", "unknown")] = dict(output)


@dataclass(slots=True, frozen=True)
class _UrlExtractor:
    source: str
    path: str


@dataclass(slots=True, frozen=True)
class _StorageConfig:
    base_path: Path
    filename_template: str


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
    """Download PDF assets referenced by upstream adapter payloads."""

    name: str
    extractors: tuple[_UrlExtractor, ...]
    storage: _StorageConfig
    timeout_seconds: float
    max_attempts: int
    user_agent: str
    http_client: HttpClient = field(repr=False)
    _ledger: JobLedger | None = field(default=None, init=False, repr=False)

    def bind_runtime(self, *, job_ledger: JobLedger | None = None) -> None:
        self._ledger = job_ledger

    def execute(self, ctx: StageContext, upstream: Any) -> Mapping[str, Any]:
        payloads: Sequence[Mapping[str, Any]] = []
        if isinstance(upstream, Mapping):
            payloads = [upstream]
        elif isinstance(upstream, Sequence):
            payloads = [item for item in upstream if isinstance(item, Mapping)]

        candidate_urls = self._collect_urls(ctx, payloads)
        pipeline = ctx.pipeline_name or "unknown"
        start_time = time.perf_counter()
        attempt = 0
        failures: list[dict[str, Any]] = []
        for url in candidate_urls:
            attempt += 1
            try:
                logger.info(
                    "dagster.stage.download.fetch",
                    stage=self.name,
                    url=url,
                    tenant_id=ctx.tenant_id,
                    pipeline=ctx.pipeline_name,
                    correlation_id=ctx.correlation_id,
                )
                response = self.http_client.request(
                    "GET",
                    url,
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                content = response.content
                checksum = hashlib.sha256(content).hexdigest()
                target_path = self._write_file(ctx, url, content)
                elapsed = time.perf_counter() - start_time
                result = {
                    "status": "downloaded",
                    "url": url,
                    "path": str(target_path),
                    "size_bytes": len(content),
                    "sha256": checksum,
                    "content_type": response.headers.get("content-type"),
                    "downloaded_at": datetime.now(UTC).isoformat(),
                    "attempt": attempt,
                }
                self._record_success(ctx, result)
                record_pdf_download_duration(pipeline, "success", elapsed)
                record_pdf_download_bytes(pipeline, len(content))
                record_pdf_download_event(pipeline, "success")
                return {
                    "status": "success",
                    "files": [result],
                    "attempts": attempt,
                    "gate_ready": True,
                    "elapsed_seconds": elapsed,
                }
            except Exception as exc:  # pragma: no cover - exercised via failure test path
                logger.warning(
                    "dagster.stage.download.failed",
                    stage=self.name,
                    tenant_id=ctx.tenant_id,
                    pipeline=ctx.pipeline_name,
                    correlation_id=ctx.correlation_id,
                    url=url,
                    error=str(exc),
                )
                failures.append({"url": url, "error": str(exc), "attempt": attempt})
                record_pdf_download_event(pipeline, "failure")

        elapsed = time.perf_counter() - start_time
        self._record_failure(ctx, failures)
        record_pdf_download_duration(pipeline, "failure", elapsed)
        return {
            "status": "failed",
            "files": [],
            "attempts": attempt,
            "failures": failures,
            "elapsed_seconds": elapsed,
        }

    def _collect_urls(
        self,
        ctx: StageContext,
        payloads: Sequence[Mapping[str, Any]],
    ) -> list[str]:
        urls: list[str] = []
        for extractor in self.extractors:
            sources: Iterable[Any]
            if extractor.source == "payload":
                sources = payloads
            elif extractor.source == "context":
                sources = [ctx.metadata]
            else:
                logger.debug(
                    "dagster.stage.download.unknown_source",
                    stage=self.name,
                    source=extractor.source,
                )
                continue
            for source in sources:
                value = self._resolve_path(source, extractor.path)
                if isinstance(value, str) and value.startswith("http"):
                    urls.append(value)
                elif isinstance(value, Sequence):
                    for item in value:
                        if isinstance(item, str) and item.startswith("http"):
                            urls.append(item)
        # remove duplicates preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique.append(url)
        return unique

    def _resolve_path(self, data: Any, path: str) -> Any:
        current = [data]
        for raw_segment in path.split("."):
            segment = raw_segment
            explode = False
            if raw_segment.endswith("[]"):
                explode = True
                segment = raw_segment[:-2]
            next_values: list[Any] = []
            for value in current:
                if isinstance(value, Mapping):
                    next_value = value.get(segment)
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    try:
                        index = int(segment)
                    except ValueError:
                        continue
                    next_value = value[index] if 0 <= index < len(value) else None
                else:
                    next_value = getattr(value, segment, None)
                if explode and isinstance(next_value, Sequence) and not isinstance(next_value, (str, bytes)):
                    next_values.extend(next_value)
                elif explode and isinstance(next_value, Mapping):
                    next_values.extend(next_value.values())
                elif next_value is not None:
                    next_values.append(next_value)
            current = next_values
        if not current:
            return None
        if len(current) == 1:
            return current[0]
        return current

    def _write_file(self, ctx: StageContext, url: str, content: bytes) -> Path:
        target_dir = self.storage.base_path
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = self._render_filename(ctx, url)
        path = target_dir / filename
        path.write_bytes(content)
        return path

    def _render_filename(self, ctx: StageContext, url: str) -> str:
        identifier = ctx.doc_id or ctx.job_id or hashlib.sha1(url.encode(), usedforsecurity=False).hexdigest()
        template_context = {
            "doc_id": identifier,
            "job_id": ctx.job_id or identifier,
            "tenant_id": ctx.tenant_id,
            "timestamp": datetime.now(UTC).strftime("%Y%m%d%H%M%S"),
        }
        try:
            rendered = self.storage.filename_template.format(**template_context)
        except KeyError:
            rendered = f"{identifier}.pdf"
        safe = [char if char.isalnum() or char in {"-", "_", "."} else "-" for char in rendered]
        filename = "".join(safe)
        if not filename.lower().endswith(".pdf"):
            filename = f"{filename}.pdf"
        return filename

    def _record_success(self, ctx: StageContext, result: Mapping[str, Any]) -> None:
        if not self._ledger or not ctx.job_id:
            return
        metadata = {
            "pdf_url": result.get("url"),
            "pdf_path": result.get("path"),
            "pdf_sha256": result.get("sha256"),
            "pdf_downloaded_at": result.get("downloaded_at"),
        }
        self._ledger.set_pdf_downloaded(ctx.job_id, True)
        self._ledger.update_metadata(ctx.job_id, metadata)

    def _record_failure(self, ctx: StageContext, failures: Sequence[Mapping[str, Any]]) -> None:
        if not self._ledger or not ctx.job_id:
            return
        failure = failures[-1] if failures else {}
        metadata = {
            "pdf_downloaded": False,
            "pdf_download_error": failure.get("error"),
            "pdf_last_attempt_url": failure.get("url"),
        }
        self._ledger.set_pdf_downloaded(ctx.job_id, False)
        self._ledger.update_metadata(ctx.job_id, metadata)


@dataclass(slots=True)
class GateStage:
    """Wait for ledger conditions to resume downstream execution."""

    name: str
    gate_name: str
    field: str
    expected: Any
    resume_stage: str
    timeout_seconds: float
    poll_interval_seconds: float
    _ledger: JobLedger | None = field(default=None, init=False, repr=False)

    def bind_runtime(self, *, job_ledger: JobLedger | None = None) -> None:
        self._ledger = job_ledger

    def execute(self, ctx: StageContext, upstream: Any) -> Mapping[str, Any]:  # pragma: no cover - validated in unit tests
        if self._ledger is None:
            raise GateConditionError(f"Gate '{self.name}' requires JobLedger binding")
        if ctx.job_id is None:
            raise GateConditionError(f"Gate '{self.name}' requires job context for ledger evaluation")

        pipeline = ctx.pipeline_name or "unknown"
        start = time.perf_counter()
        attempts = 0
        deadline = start + self.timeout_seconds
        last_value: Any = None
        while True:
            entry = self._ledger.get(ctx.job_id)
            attempts += 1
            if entry is None:
                raise GateConditionError(f"Gate '{self.name}' could not locate ledger entry for job '{ctx.job_id}'")
            last_value = self._resolve_field(entry, self.field)
            if last_value == self.expected:
                elapsed = time.perf_counter() - start
                metadata = {
                    "gate": self.gate_name,
                    "elapsed_seconds": elapsed,
                    "attempts": attempts,
                    "value": last_value,
                }
                record_gate_wait_duration(pipeline, self.gate_name, elapsed)
                record_gate_event(pipeline, self.gate_name, "passed")
                logger.info(
                    "dagster.stage.gate.passed",
                    stage=self.name,
                    tenant_id=ctx.tenant_id,
                    pipeline=ctx.pipeline_name,
                    correlation_id=ctx.correlation_id,
                    gate=self.gate_name,
                    elapsed=elapsed,
                    attempts=attempts,
                )
                self._ledger.update_metadata(
                    ctx.job_id,
                    {
                        f"gate.{self.gate_name}.passed_at": datetime.now(UTC).isoformat(),
                        f"gate.{self.gate_name}.attempts": attempts,
                        f"gate.{self.gate_name}.elapsed_seconds": elapsed,
                    },
                )
                return metadata
            if time.perf_counter() >= deadline:
                elapsed = time.perf_counter() - start
                record_gate_wait_duration(pipeline, self.gate_name, elapsed)
                record_gate_event(pipeline, self.gate_name, "timeout")
                logger.warning(
                    "dagster.stage.gate.timeout",
                    stage=self.name,
                    tenant_id=ctx.tenant_id,
                    pipeline=ctx.pipeline_name,
                    correlation_id=ctx.correlation_id,
                    gate=self.gate_name,
                    elapsed=elapsed,
                    attempts=attempts,
                    last_value=last_value,
                )
                raise GateConditionError(
                    f"Gate '{self.name}' timed out after {elapsed:.2f}s waiting for {self.field} == {self.expected!r}"
                )
            record_gate_event(pipeline, self.gate_name, "waiting")
            time.sleep(self.poll_interval_seconds)

    def _resolve_field(self, entry: Any, path: str) -> Any:
        value: Any = entry
        for segment in path.split("."):
            if isinstance(value, Mapping):
                value = value.get(segment)
            else:
                value = getattr(value, segment, None)
        return value


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
        extractor_configs = config.get("url_extractors", [])
        extractors = tuple(
            _UrlExtractor(source=str(item.get("source")), path=str(item.get("path")))
            for item in extractor_configs
            if isinstance(item, Mapping)
        )
        storage_config = config.get("storage", {}) if isinstance(config.get("storage"), Mapping) else {}
        storage = _StorageConfig(
            base_path=Path(str(storage_config.get("base_path"))),
            filename_template=str(storage_config.get("filename_template", "{job_id}.pdf")),
        )
        http_config = config.get("http", {}) if isinstance(config.get("http"), Mapping) else {}
        timeout = float(http_config.get("timeout_seconds", 45))
        attempts = int(http_config.get("max_attempts", 3))
        user_agent = str(http_config.get("user_agent", "Medical-KG-Pipeline/1.0"))
        retry = RetryConfig(
            attempts=attempts,
            timeout=timeout,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            backoff_initial=1.0,
            backoff_max=min(60.0, timeout * 2),
        )
        client = HttpClient(retry=retry)
        return DownloadStage(
            name=definition.name,
            extractors=extractors,
            storage=storage,
            timeout_seconds=timeout,
            max_attempts=attempts,
            user_agent=user_agent,
            http_client=client,
        )

    metadata = StageMetadata(
        stage_type="download",
        state_key="download",
        output_handler=_handle_download_output,
        output_counter=_sequence_length,
        description="Downloads external PDF resources and stores them for MinerU processing",
        dependencies=("ingest",),
    )
    return StageRegistration(metadata=metadata, builder=_builder)


def register_gate_stage() -> StageRegistration:
    """Register the built-in gate stage plugin."""

    def _builder(definition: StageDefinition) -> GateStage:
        config = definition.config or {}
        gate_name = str(config.get("gate") or definition.name)
        field = str(config.get("field", "pdf_ir_ready"))
        expected = config.get("equals", True)
        resume_stage = str(config.get("resume_stage", "chunk"))
        timeout = float(config.get("timeout_seconds", 900))
        poll = float(config.get("poll_interval_seconds", 15))
        return GateStage(
            name=definition.name,
            gate_name=gate_name,
            field=field,
            expected=expected,
            resume_stage=resume_stage,
            timeout_seconds=timeout,
            poll_interval_seconds=poll,
        )

    metadata = StageMetadata(
        stage_type="gate",
        state_key=None,
        output_handler=_handle_gate_output,
        output_counter=lambda _: 0,
        description="Halts pipeline execution until ledger conditions are satisfied",
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
    "DownloadStage",
    "GateConditionError",
    "GateStage",
    "PdfDownloadRecord",
    "PdfDownloadStage",
    "MineruStage",
    "register_download_stage",
    "register_gate_stage",
    "register_mineru_stage",
]
