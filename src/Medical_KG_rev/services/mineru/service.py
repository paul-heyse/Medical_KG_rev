"""MinerU split-container service implementation."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata

import structlog

from Medical_KG_rev.config.settings import MineruSettings, get_settings
from Medical_KG_rev.storage.object_store import FigureStorageClient

from .cli_wrapper import (
    MineruCliBase,
    MineruCliError,
    MineruCliInput,
    MineruCliResult,
    SimulatedMineruCli,
    create_cli,
)
from .metrics import MINERU_CLI_FAILURES_TOTAL
from .output_parser import MineruOutputParser, MineruOutputParserError, ParsedDocument
from .pipeline import MineruPipeline, PipelineMetrics
from .postprocessor import MineruPostProcessor
from .types import (
    Document,
    MineruBatchResponse,
    MineruRequest,
    MineruResponse,
    ProcessingMetadata,
)
from .vllm_client import VLLMClient, VLLMClientError

logger = structlog.get_logger(__name__)


class MineruOutOfMemoryError(MineruCliError):
    """Raised when MinerU CLI indicates an out-of-memory failure."""


class MineruProcessor:
    """MinerU processor configured for split-container vLLM usage."""

    def __init__(
        self,
        *,
        settings: MineruSettings | None = None,
        cli: MineruCliBase | None = None,
        parser: MineruOutputParser | None = None,
        postprocessor: MineruPostProcessor | None = None,
        figure_storage: FigureStorageClient | None = None,
        worker_id: str | None = None,
        vllm_client: VLLMClient | None = None,
    ) -> None:
        self._settings = settings or get_settings().mineru
        self._worker_id = worker_id or threading.current_thread().name
        parser_instance = parser or MineruOutputParser()
        postprocessor_instance = postprocessor or MineruPostProcessor(
            figure_storage=figure_storage
        )
        self._pipeline = MineruPipeline(
            parser=parser_instance,
            postprocessor=postprocessor_instance,
            metrics=PipelineMetrics(worker_id=self._worker_id),
        )
        self._parser = parser_instance
        self._postprocessor = postprocessor_instance
        self._cli = cli or create_cli(self._settings)
        self._mineru_version = self._ensure_mineru_version()
        self._vllm_client = vllm_client or self._build_vllm_client()
        self._ensure_vllm_health()
        logger.info(
            "mineru.processor.initialised",
            worker_id=self._worker_id,
            backend=self._settings.workers.backend,
            vllm_url=str(self._settings.vllm_server.base_url),
        )

    def process(self, request: MineruRequest) -> MineruResponse:
        batch = self.process_batch([request])
        if not batch.documents:
            raise MineruCliError("MinerU CLI returned no outputs")
        return MineruResponse(
            document=batch.documents[0],
            processed_at=batch.processed_at,
            duration_seconds=batch.duration_seconds,
            metadata=batch.metadata[0],
        )

    def process_batch(self, requests: Sequence[MineruRequest]) -> MineruBatchResponse:
        request_list = list(requests)
        if not request_list:
            now = datetime.now(timezone.utc)
            return MineruBatchResponse(
                documents=[], processed_at=now, duration_seconds=0.0, metadata=[]
            )

        batch_limit = max(1, self._settings.workers.batch_size)
        batches = [request_list[i : i + batch_limit] for i in range(0, len(request_list), batch_limit)]
        logger.bind(
            size=len(request_list), batches=len(batches), batch_limit=batch_limit
        ).info("mineru.process.batch_started")

        start_monotonic = time.monotonic()
        aggregated_documents: list[Document] = []
        aggregated_metadata: list[ProcessingMetadata] = []
        processed_at: datetime | None = None

        for index, batch in enumerate(batches, start=1):
            partial = self._run_cli_batch(batch, batch_index=index, total_batches=len(batches))
            aggregated_documents.extend(partial.documents)
            aggregated_metadata.extend(partial.metadata)
            processed_at = partial.processed_at

        duration = time.monotonic() - start_monotonic
        logger.bind(
            size=len(request_list), batches=len(batches), duration=round(duration, 4)
        ).info("mineru.process.batch_completed")

        return MineruBatchResponse(
            documents=aggregated_documents,
            processed_at=processed_at or datetime.now(timezone.utc),
            duration_seconds=duration,
            metadata=aggregated_metadata,
        )

    def _run_cli_batch(
        self,
        requests: Sequence[MineruRequest],
        *,
        batch_index: int,
        total_batches: int,
    ) -> MineruBatchResponse:
        cli_inputs = [
            MineruCliInput(document_id=request.document_id, content=request.content)
            for request in requests
        ]

        def orchestrate(
            executor: Callable[[Sequence[MineruCliInput]], tuple[MineruCliResult, str, int]],
        ) -> MineruBatchResponse:
            return self._pipeline.execute(
                requests=requests,
                cli_inputs=cli_inputs,
                execute_cli=executor,
                metadata_builder=self._build_metadata,
                batch_index=batch_index,
                total_batches=total_batches,
                record_gpu_memory=None,
            )

        try:
            return orchestrate(self._execute_cli)
        except MineruCliError as exc:
            logger.bind(
                reason="cli-error", error=str(exc), batch=batch_index
            ).error("mineru.process.failed")
            if self._handle_cli_failure(exc):
                return orchestrate(self._execute_simulated_cli)
            raise
        except MineruOutputParserError:
            raise
        except Exception as exc:  # pragma: no cover - surfaced to caller
            logger.bind(error=str(exc)).exception("mineru.process.unexpected")
            raise

    def _execute_cli(
        self, cli_inputs: Sequence[MineruCliInput]
    ) -> tuple["MineruCliResult", str, int]:
        cli_result = self._cli.run_batch(cli_inputs)
        return cli_result, "vllm-http", 0

    def _execute_simulated_cli(
        self, cli_inputs: Sequence[MineruCliInput]
    ) -> tuple["MineruCliResult", str, int]:
        simulated = self._cli
        if not isinstance(simulated, SimulatedMineruCli):
            simulated = SimulatedMineruCli(self._settings)
            self._cli = simulated
        cli_result = simulated.run_batch(cli_inputs)
        return cli_result, "simulated", 0

    def _handle_cli_failure(self, exc: MineruCliError) -> bool:
        reason = "oom" if self._looks_like_oom(str(exc)) else "cli-error"
        MINERU_CLI_FAILURES_TOTAL.labels(reason=reason).inc()
        if reason == "oom":
            raise MineruOutOfMemoryError(str(exc)) from exc
        return True

    def _ensure_mineru_version(self) -> str | None:
        try:
            installed = importlib_metadata.version("mineru")
        except importlib_metadata.PackageNotFoundError:  # pragma: no cover - optional in tests
            logger.warning("mineru.version.missing")
            return None

        spec = self._settings.expected_version.strip()
        minimum = spec[2:] if spec.startswith(">=") else spec
        if minimum and self._compare_versions(installed, minimum) < 0:
            raise RuntimeError(
                f"MinerU version {installed} does not satisfy expectation '{self._settings.expected_version}'"
            )
        logger.bind(
            installed=installed, expected=self._settings.expected_version
        ).info("mineru.version.validated")
        return installed

    @staticmethod
    def _compare_versions(installed: str, minimum: str) -> int:
        def _normalize(value: str) -> tuple[int, ...]:
            parts: list[int] = []
            for token in value.split("."):
                digits = "".join(ch for ch in token if ch.isdigit())
                if digits:
                    parts.append(int(digits))
                else:
                    break
            return tuple(parts)

        from itertools import zip_longest

        left = _normalize(installed)
        right = _normalize(minimum)
        for lhs, rhs in zip_longest(left, right, fillvalue=0):
            if lhs > rhs:
                return 1
            if lhs < rhs:
                return -1
        return 0

    def _build_metadata(
        self,
        *,
        request: MineruRequest,
        parsed: ParsedDocument,
        gpu_label: str,
        started_at: datetime,
        completed_at: datetime,
        cli_result,
        planned_memory_mb: int,
    ) -> ProcessingMetadata:
        model_names = {
            "layout": parsed.metadata.get("layout_model", "unknown"),
            "table": parsed.metadata.get("table_model", "unknown"),
            "vision": parsed.metadata.get("vision_model", "unknown"),
        }
        return ProcessingMetadata(
            document_id=request.document_id,
            mineru_version=self._mineru_version,
            model_names=model_names,
            gpu_id=gpu_label,
            worker_id=self._worker_id,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=float(cli_result.duration_seconds),
            cli_stdout=cli_result.stdout.strip(),
            cli_stderr=cli_result.stderr.strip(),
            cli_descriptor=self._cli.describe(),
            planned_memory_mb=planned_memory_mb,
        )

    def _looks_like_oom(self, message: str) -> bool:
        lowered = message.lower()
        return "out of memory" in lowered or "oom" in lowered

    def _build_vllm_client(self) -> VLLMClient:
        breaker_settings = self._settings.http_client.circuit_breaker
        circuit_breaker = None
        if breaker_settings.enabled:
            from .circuit_breaker import CircuitBreaker

            circuit_breaker = CircuitBreaker(
                failure_threshold=breaker_settings.failure_threshold,
                recovery_timeout=breaker_settings.recovery_timeout_seconds,
                success_threshold=breaker_settings.success_threshold,
            )
        client = VLLMClient(
            base_url=str(self._settings.vllm_server.base_url),
            timeout=self._settings.http_client.timeout_seconds,
            max_connections=self._settings.http_client.connection_pool_size,
            max_keepalive_connections=self._settings.http_client.keepalive_connections,
            circuit_breaker=circuit_breaker,
            retry_attempts=self._settings.http_client.retry_attempts,
            retry_backoff_multiplier=self._settings.http_client.retry_backoff_multiplier,
        )
        return client

    def _ensure_vllm_health(self) -> None:
        async def _check() -> bool:
            try:
                return await self._vllm_client.health_check()
            except VLLMClientError as exc:
                logger.error("mineru.vllm.health_check_failed", error=str(exc))
                return False

        try:
            healthy = asyncio.run(_check())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                healthy = loop.run_until_complete(_check())
            finally:
                loop.close()
        if not healthy:
            raise RuntimeError(
                f"vLLM server unavailable at {self._settings.vllm_server.base_url}"
            )

    @property
    def vllm_client(self) -> VLLMClient:
        """Expose the configured vLLM client for observability and testing."""

        return self._vllm_client


class MineruGrpcService:
    """Async gRPC servicer bridging proto definitions to the processor."""

    def __init__(self, processor: MineruProcessor) -> None:
        self._processor = processor

    async def ProcessPdf(self, request, context):  # type: ignore[override]
        mineru_request = MineruRequest(
            tenant_id=request.tenant_id,
            document_id=request.document_id,
            content=request.content,
        )
        try:
            response = self._processor.process(mineru_request)
        except MineruOutOfMemoryError as exc:
            if context is not None:
                import grpc

                await context.abort(
                    code=grpc.StatusCode.RESOURCE_EXHAUSTED,
                    details=str(exc),
                )
            raise
        except (MineruCliError, VLLMClientError) as exc:
            if context is not None:
                import grpc

                await context.abort(
                    code=grpc.StatusCode.INTERNAL,
                    details=str(exc),
                )
            raise

        try:
            from Medical_KG_rev.proto.gen import mineru_pb2  # type: ignore import-error
        except ImportError:  # pragma: no cover - generated stubs absent during CI
            return None

        reply = mineru_pb2.ProcessPdfResponse()
        self._populate_document(reply.document, response.document)
        self._populate_metadata(reply.metadata, response.metadata)
        return reply

    @staticmethod
    def _populate_document(proto_document, document: Document) -> None:
        proto_document.document_id = document.document_id
        proto_document.tenant_id = document.tenant_id
        for block in document.blocks:
            item = proto_document.blocks.add()
            item.id = block.id
            item.page = block.page
            item.kind = block.kind
            item.text = block.text or ""
            item.confidence = block.confidence or 0.0
            item.reading_order = block.reading_order or 0

    @staticmethod
    def _populate_metadata(proto_metadata, metadata: ProcessingMetadata) -> None:
        proto_metadata.document_id = metadata.document_id
        proto_metadata.worker_id = metadata.worker_id or ""
        proto_metadata.started_at = metadata.started_at.isoformat()
        proto_metadata.completed_at = metadata.completed_at.isoformat()
        proto_metadata.duration_seconds = metadata.duration_seconds
        proto_metadata.cli_stdout = metadata.cli_stdout
        proto_metadata.cli_stderr = metadata.cli_stderr
        proto_metadata.cli_descriptor = metadata.cli_descriptor
        if metadata.mineru_version:
            proto_metadata.mineru_version = metadata.mineru_version


__all__ = [
    "MineruProcessor",
    "MineruResponse",
    "MineruRequest",
    "MineruBatchResponse",
    "MineruOutOfMemoryError",
    "MineruGrpcService",
]
