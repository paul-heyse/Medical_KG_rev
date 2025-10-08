from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import structlog

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import structlog

from Medical_KG_rev.orchestration.dagster.configuration import (
    GateDefinition,
    StageDefinition,
)
from Medical_KG_rev.orchestration.dagster.gates import (
    GateConditionError,
    GateConditionEvaluator,
    GateEvaluationResult,
    GateTimeoutError,
    build_gate_result,
)
from Medical_KG_rev.orchestration.dagster.stage_registry import (
    StageMetadata,
    StageRegistration,
)
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.stages.contracts import StageContext
from Medical_KG_rev.utils.http_client import BackoffStrategy, HttpClient, RetryConfig

if TYPE_CHECKING:  # pragma: no cover - hints only
    from Medical_KG_rev.orchestration.ledger import JobLedger

logger = structlog.get_logger(__name__)


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
            )
        logger.debug(
            "dagster.stage.download.completed",
            stage=self.name,
            tenant_id=ctx.tenant_id,
            files=len(results),
        )
        return results


@dataclass(slots=True)
class GateStage:
    """Gate stage validating ledger state before proceeding."""

    name: str
    definition: GateDefinition
    evaluator: GateConditionEvaluator
    timeout_seconds: int | None
    poll_interval: float
    max_attempts: int | None
    retry_backoff: float

    def evaluate(
        self,
        ctx: StageContext,
        ledger: "JobLedger",
        state: dict[str, Any],
    ) -> GateEvaluationResult:
        job_id = ctx.job_id or state.get("job_id")
        if not job_id:
            raise GateConditionError(f"Gate '{self.name}' cannot evaluate without a job identifier")

        gate_state = state.setdefault("gates", {}).setdefault(self.definition.name, {})
        attempts = 0
        start_time = time.perf_counter()
        deadline = (
            start_time + self.timeout_seconds if self.timeout_seconds is not None else None
        )

        while True:
            attempts += 1
            entry = ledger.get(job_id)
            if entry is None:
                raise GateConditionError(
                    f"Gate '{self.name}' could not locate job '{job_id}' in the ledger"
                )

            satisfied, details = self.evaluator.evaluate(entry, state, gate_state)
            gate_state.update(
                {
                    "last_details": details,
                    "attempts": attempts,
                    "status": "passed" if satisfied else "waiting",
                }
            )

            if satisfied:
                result = build_gate_result(
                    self.definition,
                    True,
                    attempts,
                    start_time,
                    details,
                )
                gate_state["result"] = result.details
                return result

            failure_reason = _describe_gate_failure(details)
            gate_state["status"] = "failed"
            gate_state["error"] = failure_reason
            failure_result = build_gate_result(
                self.definition,
                False,
                attempts,
                start_time,
                details,
                last_error=failure_reason,
            )
            gate_state["result"] = failure_result.details

            current_time = time.perf_counter()
            if deadline is not None and current_time >= deadline:
                raise GateTimeoutError(
                    f"Gate '{self.name}' timed out after {self.timeout_seconds} seconds",
                    result=failure_result,
                )
            if self.max_attempts is not None and attempts >= self.max_attempts:
                raise GateConditionError(
                    f"Gate '{self.name}' failed after {attempts} attempts: {failure_reason}",
                    result=failure_result,
                )

            sleep_for = self.poll_interval if attempts == 1 else max(self.poll_interval, self.retry_backoff)
            logger.debug(
                "dagster.stage.gate.retrying",
                stage=self.name,
                tenant_id=ctx.tenant_id,
                attempts=attempts,
                sleep_seconds=sleep_for,
                reason=failure_reason,
            )
            time.sleep(sleep_for)


def _describe_gate_failure(details: Mapping[str, Any]) -> str:
    clauses = details.get("clauses", []) if isinstance(details, Mapping) else []
    for clause in clauses:
        if not isinstance(clause, Mapping):
            continue
        for predicate in clause.get("all", []) or []:
            if isinstance(predicate, Mapping) and not predicate.get("passed", True):
                field = predicate.get("field")
                expected = predicate.get("expected")
                actual = predicate.get("actual")
                operator = predicate.get("operator")
                return (
                    f"{field}={actual!r} did not satisfy {operator} {expected!r}"
                )
        for predicate in clause.get("any", []) or []:
            if isinstance(predicate, Mapping) and not predicate.get("passed", True):
                field = predicate.get("field")
                expected = predicate.get("expected")
                actual = predicate.get("actual")
                operator = predicate.get("operator")
                return (
                    f"{field}={actual!r} did not satisfy {operator} {expected!r}"
                )
    return "gate conditions not satisfied"


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
        gate_payload = config.get("definition") or config
        gate_definition = GateDefinition.model_validate(gate_payload)
        evaluator = GateConditionEvaluator(gate_definition)
        retry = gate_definition.retry
        return GateStage(
            name=definition.name,
            definition=gate_definition,
            evaluator=evaluator,
            timeout_seconds=gate_definition.timeout_seconds,
            poll_interval=gate_definition.poll_interval_seconds,
            max_attempts=retry.max_attempts if retry else None,
            retry_backoff=retry.backoff_seconds if retry else gate_definition.poll_interval_seconds,
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
    "GateTimeoutError",
    "register_download_stage",
    "register_gate_stage",
    "register_mineru_stage",
]
