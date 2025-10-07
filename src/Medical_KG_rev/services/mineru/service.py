from __future__ import annotations

import itertools
import os
import threading
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata

import structlog

try:  # pragma: no cover - optional dependency in unit tests
    import grpc
except Exception:  # pragma: no cover
    grpc = None  # type: ignore

from Medical_KG_rev.config.settings import MineruSettings, get_settings
from Medical_KG_rev.services.gpu.manager import GpuManager, GpuNotAvailableError
from Medical_KG_rev.storage.object_store import FigureStorageClient

from .cli_wrapper import (
    MineruCliBase,
    MineruCliError,
    MineruCliInput,
    SimulatedMineruCli,
    create_cli,
)
from .gpu_budget import GpuBudgetPlanner
from .gpu_manager import MineruGpuManager
from .metrics import (
    MINERU_CLI_FAILURES_TOTAL,
    MINERU_FIGURE_EXTRACTION_COUNT,
    MINERU_GPU_MEMORY_USAGE_BYTES,
    MINERU_PDF_PAGES_PROCESSED_TOTAL,
    MINERU_PROCESSING_DURATION_SECONDS,
    MINERU_TABLE_EXTRACTION_COUNT,
)
from .output_parser import MineruOutputParser, MineruOutputParserError, ParsedDocument
from .postprocessor import MineruPostProcessor
from .types import (
    Document,
    MineruBatchResponse,
    MineruRequest,
    MineruResponse,
    ProcessingMetadata,
)

logger = structlog.get_logger(__name__)


_BYTES_PER_MB = 1024 * 1024


class MineruOutOfMemoryError(MineruCliError):
    """Raised when MinerU CLI indicates an out-of-memory failure."""


class MineruProcessor:
    """Integrates the MinerU CLI with GPU orchestration."""

    def __init__(
        self,
        gpu: GpuManager,
        *,
        settings: MineruSettings | None = None,
        cli: MineruCliBase | None = None,
        parser: MineruOutputParser | None = None,
        postprocessor: MineruPostProcessor | None = None,
        figure_storage: FigureStorageClient | None = None,
        min_memory_mb: int | None = None,
        worker_id: str | None = None,
        fail_fast: bool = True,
    ) -> None:
        self._settings = settings or get_settings().mineru
        self._gpu = gpu
        self._cli = cli or create_cli(self._settings)
        self._parser = parser or MineruOutputParser()
        self._postprocessor = postprocessor or MineruPostProcessor(
            figure_storage=figure_storage
        )
        self._required_memory_mb = min_memory_mb or self._settings.workers.vram_per_worker_mb
        self._worker_id = worker_id or threading.current_thread().name
        self._mineru_gpu = MineruGpuManager(gpu, self._settings)
        self._mineru_version = self._ensure_mineru_version()
        self._apply_cpu_environment()
        self._mineru_gpu.ensure_cuda_version()
        self._gpu_budget = GpuBudgetPlanner(
            self._required_memory_mb, self._settings.workers.reservation_margin
        )
        if fail_fast:
            try:
                self._gpu.wait_for_gpu(timeout=max(5.0, self._settings.workers.timeout_seconds / 10))
            except GpuNotAvailableError as exc:
                logger.bind(error=str(exc)).error("mineru.startup.gpu_unavailable")
                raise

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
            return MineruBatchResponse(documents=[], processed_at=now, duration_seconds=0.0, metadata=[])

        batch_limit = self._settings.workers.batch_limit
        batches = list(self._chunk_requests(request_list, batch_limit))
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

    def _chunk_requests(
        self, requests: Sequence[MineruRequest], limit: int
    ) -> Iterable[Sequence[MineruRequest]]:
        limit = max(1, limit)
        for start in range(0, len(requests), limit):
            yield requests[start : start + limit]

    def _run_cli_batch(
        self,
        requests: Sequence[MineruRequest],
        *,
        batch_index: int,
        total_batches: int,
    ) -> MineruBatchResponse:
        request_map = {request.document_id: request for request in requests}
        started_at = datetime.now(timezone.utc)
        start_monotonic = time.monotonic()
        cli_inputs = [
            MineruCliInput(document_id=request.document_id, content=request.content)
            for request in requests
        ]

        gpu_label: str = "unknown"
        planned_memory_mb = 0
        try:
            cli_result, gpu_label, planned_memory_mb = self._execute_cli(cli_inputs)
        except GpuNotAvailableError:
            logger.bind(reason="gpu-unavailable", batch=batch_index).error(
                "mineru.process.failed"
            )
            if self._settings.simulate_if_unavailable:
                cli_result, gpu_label, planned_memory_mb = self._execute_simulated_cli(cli_inputs)
            else:
                raise
        except MineruCliError as exc:
            logger.bind(
                reason="cli-error", error=str(exc), batch=batch_index
            ).error("mineru.process.failed")
            if self._handle_cli_failure(exc):
                cli_result, gpu_label, planned_memory_mb = self._execute_simulated_cli(cli_inputs)
            else:
                raise

        completed_at = datetime.now(timezone.utc)
        duration = time.monotonic() - start_monotonic

        if not cli_result.outputs:
            raise MineruCliError("MinerU CLI returned no outputs")

        MINERU_PROCESSING_DURATION_SECONDS.labels(
            worker_id=self._worker_id,
            gpu_id=gpu_label,
        ).observe(cli_result.duration_seconds)
        if gpu_label.startswith("cuda:"):
            self._record_gpu_memory(gpu_label)

        documents: list[Document] = []
        metadata_entries: list[ProcessingMetadata] = []
        for output in cli_result.outputs:
            request = request_map.get(output.document_id)
            if request is None:
                logger.bind(
                    document_id=output.document_id, batch=batch_index
                ).warning("mineru.process.output_without_request")
                continue
            try:
                parsed = self._parser.parse_path(output.path)
            except MineruOutputParserError as exc:
                logger.bind(error=str(exc)).error("mineru.output.parse_failed")
                raise

            metadata = self._build_metadata(
                request=request,
                parsed=parsed,
                gpu_label=gpu_label,
                started_at=started_at,
                completed_at=completed_at,
                cli_result=cli_result,
                planned_memory_mb=planned_memory_mb,
            )
            document = self._postprocessor.build_document(parsed, request, metadata.as_dict())
            documents.append(document)
            metadata_entries.append(metadata)
            self._record_extraction_metrics(parsed)

            logger.bind(
                document_id=document.document_id,
                blocks=len(document.blocks),
                tables=len(document.tables),
                figures=len(document.figures),
                equations=len(document.equations),
                batch=batch_index,
                total_batches=total_batches,
            ).info("mineru.process.completed")

        return MineruBatchResponse(
            documents=documents,
            processed_at=completed_at,
            duration_seconds=duration,
            metadata=metadata_entries,
        )

    def _record_extraction_metrics(self, parsed: ParsedDocument) -> None:
        unique_pages = {block.page for block in parsed.blocks}
        MINERU_PDF_PAGES_PROCESSED_TOTAL.labels(worker_id=self._worker_id).inc(len(unique_pages))
        MINERU_TABLE_EXTRACTION_COUNT.labels(worker_id=self._worker_id).observe(len(parsed.tables))
        MINERU_FIGURE_EXTRACTION_COUNT.labels(worker_id=self._worker_id).observe(len(parsed.figures))

    def _plan_memory_reservation(self) -> int:
        if self._required_memory_mb <= 0:
            return 0
        device = self._gpu.get_device()
        planned = self._gpu_budget.plan(device)
        if planned < self._required_memory_mb:
            logger.bind(
                device=device.index,
                configured=self._required_memory_mb,
                planned=planned,
                total=device.total_memory_mb,
            ).info("mineru.gpu.memory_budget_adjusted")
        return planned

    def _execute_cli(
        self, cli_inputs: list[MineruCliInput]
    ) -> tuple["MineruCliResult", str, int]:
        planned_memory_mb = self._plan_memory_reservation()
        with self._gpu.device_session(
            "mineru", required_memory_mb=planned_memory_mb, warmup=True
        ) as device:
            gpu_label = f"cuda:{device.index}"
            MINERU_GPU_MEMORY_USAGE_BYTES.labels(
                gpu_id=gpu_label, state="required"
            ).set(float(planned_memory_mb * _BYTES_PER_MB))
            cli_result = self._cli.run_batch(cli_inputs, gpu_id=device.index)
        return cli_result, gpu_label, planned_memory_mb

    def _execute_simulated_cli(
        self, cli_inputs: list[MineruCliInput]
    ) -> tuple["MineruCliResult", str, int]:
        simulated = self._cli
        if not isinstance(simulated, SimulatedMineruCli):
            simulated = SimulatedMineruCli(self._settings)
            self._cli = simulated
        cli_result = simulated.run_batch(cli_inputs, gpu_id=-1)
        MINERU_GPU_MEMORY_USAGE_BYTES.labels(gpu_id="cpu", state="required").set(0.0)
        return cli_result, "cpu", 0

    def _handle_cli_failure(self, exc: MineruCliError) -> bool:
        reason = "oom" if self._looks_like_oom(str(exc)) else "cli-error"
        MINERU_CLI_FAILURES_TOTAL.labels(reason=reason).inc()
        if reason == "oom":
            raise MineruOutOfMemoryError(str(exc)) from exc
        if self._settings.simulate_if_unavailable and not isinstance(self._cli, SimulatedMineruCli):
            logger.bind(reason=reason).warning("mineru.cli.fallback_to_simulation")
            return True
        return False

    def _apply_cpu_environment(self) -> None:
        for key, value in self._settings.cpu.export_environment().items():
            os.environ.setdefault(key, value)

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

        left = _normalize(installed)
        right = _normalize(minimum)
        for lhs, rhs in itertools.zip_longest(left, right, fillvalue=0):
            if lhs > rhs:
                return 1
            if lhs < rhs:
                return -1
        return 0

    def _record_gpu_memory(self, gpu_label: str) -> None:
        try:
            import torch
        except Exception:  # pragma: no cover - optional dependency
            return

        try:
            device_index = int(gpu_label.split(":")[-1])
        except (ValueError, IndexError):  # pragma: no cover - malformed label
            return

        try:
            allocated = torch.cuda.memory_allocated(device_index)
            reserved = (
                torch.cuda.memory_reserved(device_index)
                if hasattr(torch.cuda, "memory_reserved")
                else 0
            )
        except Exception:  # pragma: no cover - torch runtime errors best-effort
            return

        MINERU_GPU_MEMORY_USAGE_BYTES.labels(gpu_id=gpu_label, state="allocated").set(float(allocated))
        MINERU_GPU_MEMORY_USAGE_BYTES.labels(gpu_id=gpu_label, state="reserved").set(float(reserved))

    def _looks_like_oom(self, message: str) -> bool:
        lowered = message.lower()
        return "out of memory" in lowered or "cuda oom" in lowered or "cuda error: out of memory" in lowered

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
        except GpuNotAvailableError as exc:
            if grpc is not None and context is not None:
                await context.abort(  # pragma: no cover - exercised in integration
                    code=grpc.StatusCode.RESOURCE_EXHAUSTED,
                    details=str(exc),
                )
            raise
        except MineruOutOfMemoryError as exc:
            if grpc is not None and context is not None:
                await context.abort(
                    code=grpc.StatusCode.RESOURCE_EXHAUSTED,
                    details=str(exc),
                )
            raise
        except MineruCliError as exc:
            if grpc is not None and context is not None:
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

    async def BatchProcessPdf(self, request, context):  # type: ignore[override]
        if not hasattr(request, "requests"):
            return None  # pragma: no cover - defensive guard

        mineru_requests = [
            MineruRequest(tenant_id=item.tenant_id, document_id=item.document_id, content=item.content)
            for item in request.requests
        ]
        try:
            batch_response = self._processor.process_batch(mineru_requests)
        except MineruOutOfMemoryError as exc:
            if grpc is not None and context is not None:
                await context.abort(
                    code=grpc.StatusCode.RESOURCE_EXHAUSTED,
                    details=str(exc),
                )
            raise
        except MineruCliError as exc:
            if grpc is not None and context is not None:
                await context.abort(
                    code=grpc.StatusCode.INTERNAL,
                    details=str(exc),
                )
            raise

        try:
            from Medical_KG_rev.proto.gen import mineru_pb2  # type: ignore import-error
        except ImportError:  # pragma: no cover - generated stubs absent during CI
            return None

        reply = mineru_pb2.BatchProcessPdfResponse()
        for document, metadata in zip(batch_response.documents, batch_response.metadata):
            proto_document = reply.documents.add()
            self._populate_document(proto_document, document)
            proto_metadata = reply.metadata.add()
            self._populate_metadata(proto_metadata, metadata)
        return reply

    @staticmethod
    def _populate_document(proto_document, document: Document) -> None:
        proto_document.document_id = document.document_id
        proto_document.tenant_id = document.tenant_id
        for block in document.blocks:
            reply_block = proto_document.blocks.add()
            reply_block.id = block.id
            reply_block.page = block.page
            reply_block.kind = block.kind
            reply_block.text = block.text or ""
            reply_block.confidence = float(block.confidence or 0.0)
            reply_block.reading_order = int(block.reading_order or 0)
            if block.bbox:
                (
                    reply_block.bbox.x0,
                    reply_block.bbox.y0,
                    reply_block.bbox.x1,
                    reply_block.bbox.y1,
                ) = block.bbox
        for table in document.tables:
            proto_table = proto_document.tables.add()
            proto_table.id = table.id
            proto_table.page = table.page
            if table.bbox:
                (
                    proto_table.bbox.x0,
                    proto_table.bbox.y0,
                    proto_table.bbox.x1,
                    proto_table.bbox.y1,
                ) = table.bbox
            proto_table.caption = table.caption or ""
            proto_table.headers.extend(table.headers)
            for cell in table.cells:
                proto_cell = proto_table.cells.add()
                proto_cell.row = cell.row
                proto_cell.column = cell.column
                proto_cell.content = cell.content
                proto_cell.rowspan = cell.rowspan
                proto_cell.colspan = cell.colspan
                if cell.bbox:
                    (
                        proto_cell.bbox.x0,
                        proto_cell.bbox.y0,
                        proto_cell.bbox.x1,
                        proto_cell.bbox.y1,
                    ) = cell.bbox
                if cell.confidence is not None:
                    proto_cell.confidence = float(cell.confidence)
        for figure in document.figures:
            proto_figure = proto_document.figures.add()
            proto_figure.id = figure.id
            proto_figure.page = figure.page
            proto_figure.image_path = figure.image_path
            proto_figure.caption = figure.caption or ""
            proto_figure.figure_type = figure.figure_type or ""
            proto_figure.mime_type = figure.mime_type or ""
            if figure.width is not None:
                proto_figure.width = int(figure.width)
            if figure.height is not None:
                proto_figure.height = int(figure.height)
            if figure.bbox:
                (
                    proto_figure.bbox.x0,
                    proto_figure.bbox.y0,
                    proto_figure.bbox.x1,
                    proto_figure.bbox.y1,
                ) = figure.bbox
        for equation in document.equations:
            proto_equation = proto_document.equations.add()
            proto_equation.id = equation.id
            proto_equation.page = equation.page
            proto_equation.latex = equation.latex
            proto_equation.mathml = equation.mathml or ""
            proto_equation.display = equation.display
            if equation.bbox:
                (
                    proto_equation.bbox.x0,
                    proto_equation.bbox.y0,
                    proto_equation.bbox.x1,
                    proto_equation.bbox.y1,
                ) = equation.bbox
        if document.metadata:
            proto_document.metadata.entries.update(
                {str(key): str(value) for key, value in document.metadata.items()}
            )

    @staticmethod
    def _populate_metadata(proto_metadata, metadata: ProcessingMetadata) -> None:
        proto_metadata.document_id = metadata.document_id
        proto_metadata.mineru_version = metadata.mineru_version or ""
        proto_metadata.gpu_id = metadata.gpu_id or ""
        proto_metadata.worker_id = metadata.worker_id or ""
        proto_metadata.started_at = metadata.started_at.astimezone(timezone.utc).isoformat()
        proto_metadata.completed_at = metadata.completed_at.astimezone(timezone.utc).isoformat()
        proto_metadata.duration_seconds = metadata.duration_seconds
        proto_metadata.cli_stdout = metadata.cli_stdout
        proto_metadata.cli_stderr = metadata.cli_stderr
        proto_metadata.model_names.update(metadata.model_names)


__all__ = [
    "MineruProcessor",
    "MineruGrpcService",
    "MineruRequest",
    "MineruResponse",
    "MineruOutOfMemoryError",
]
