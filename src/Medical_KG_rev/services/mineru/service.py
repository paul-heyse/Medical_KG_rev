from __future__ import annotations

import time
from datetime import datetime, timezone

import structlog

try:  # pragma: no cover - optional dependency in unit tests
    import grpc
except Exception:  # pragma: no cover
    grpc = None  # type: ignore

from Medical_KG_rev.config.settings import MineruSettings, get_settings
from Medical_KG_rev.services.gpu.manager import GpuManager, GpuNotAvailableError

from .cli_wrapper import MineruCliBase, MineruCliError, MineruCliInput, create_cli
from .output_parser import MineruOutputParser, MineruOutputParserError
from .postprocessor import MineruPostProcessor
from .types import Document, MineruRequest, MineruResponse

logger = structlog.get_logger(__name__)


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
        min_memory_mb: int | None = None,
    ) -> None:
        self._settings = settings or get_settings().mineru
        self._gpu = gpu
        self._cli = cli or create_cli(self._settings)
        self._parser = parser or MineruOutputParser()
        self._postprocessor = postprocessor or MineruPostProcessor()
        self._required_memory_mb = min_memory_mb or self._settings.workers.vram_per_worker_mb

    def process(self, request: MineruRequest) -> MineruResponse:
        logger.info(
            "mineru.process.started",
            document_id=request.document_id,
            tenant_id=request.tenant_id,
        )
        device_index: int | None = None
        start = time.monotonic()
        try:
            with self._gpu.device_session(
                "mineru", required_memory_mb=self._required_memory_mb, warmup=True
            ) as device:
                device_index = device.index
                cli_result = self._cli.run_batch(
                    [
                        MineruCliInput(
                            document_id=request.document_id,
                            content=request.content,
                        )
                    ],
                    gpu_id=device.index,
                )
        except GpuNotAvailableError:
            logger.error("mineru.process.failed", reason="gpu-unavailable")
            raise
        except MineruCliError as exc:
            logger.error("mineru.process.failed", reason="cli-error", error=str(exc))
            raise

        if not cli_result.outputs:
            raise MineruCliError("MinerU CLI returned no outputs")

        output = cli_result.outputs[0]
        try:
            parsed = self._parser.parse_path(output.path)
        except MineruOutputParserError as exc:
            logger.error("mineru.output.parse_failed", error=str(exc))
            raise

        duration = time.monotonic() - start
        provenance = {
            "cli": self._cli.describe(),
            "gpu_device": f"cuda:{device_index}" if device_index is not None else "unknown",
            "duration_seconds": cli_result.duration_seconds,
            "stdout": cli_result.stdout.strip(),
            "stderr": cli_result.stderr.strip(),
        }
        document: Document = self._postprocessor.build_document(parsed, request, provenance)
        logger.info(
            "mineru.process.completed",
            document_id=document.document_id,
            blocks=len(document.blocks),
            tables=len(document.tables),
            figures=len(document.figures),
            equations=len(document.equations),
        )
        return MineruResponse(
            document=document,
            processed_at=datetime.now(timezone.utc),
            duration_seconds=duration,
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
        except MineruCliError as exc:
            if grpc is not None and context is not None:
                await context.abort(
                    code=grpc.StatusCode.INTERNAL,
                    details=str(exc),
                )
            raise

        from Medical_KG_rev.proto.gen import mineru_pb2  # type: ignore import-error

        reply = mineru_pb2.ProcessPdfResponse()
        document = reply.document
        document.document_id = response.document.document_id
        document.tenant_id = response.document.tenant_id
        for block in response.document.blocks:
            reply_block = document.blocks.add()
            reply_block.id = block.id
            reply_block.page = block.page
            reply_block.kind = block.kind
            reply_block.text = block.text or ""
            reply_block.confidence = float(block.confidence or 0.0)
            if block.bbox:
                (
                    reply_block.bbox.x0,
                    reply_block.bbox.y0,
                    reply_block.bbox.x1,
                    reply_block.bbox.y1,
                ) = block.bbox
        return reply


__all__ = ["MineruProcessor", "MineruGrpcService", "MineruRequest", "MineruResponse"]
