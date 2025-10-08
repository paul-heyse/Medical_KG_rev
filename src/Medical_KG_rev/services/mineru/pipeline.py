"""Reusable MinerU pipeline orchestration helpers."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Sequence

import structlog

from .cli_wrapper import MineruCliError, MineruCliInput, MineruCliResult
from .metrics import (
    MINERU_FIGURE_EXTRACTION_COUNT,
    MINERU_PDF_PAGES_PROCESSED_TOTAL,
    MINERU_PROCESSING_DURATION_SECONDS,
    MINERU_TABLE_EXTRACTION_COUNT,
)
from .output_parser import MineruOutputParser, MineruOutputParserError, ParsedDocument
from .postprocessor import MineruPostProcessor
from .types import Document, MineruBatchResponse, MineruRequest, ProcessingMetadata

logger = structlog.get_logger(__name__)


class PipelineMetrics:
    """Encapsulates Prometheus metric emission for MinerU batches."""

    def __init__(self, worker_id: str) -> None:
        self._worker_id = worker_id

    def record_cli_duration(self, gpu_label: str, duration: float) -> None:
        MINERU_PROCESSING_DURATION_SECONDS.labels(
            worker_id=self._worker_id, gpu_id=gpu_label
        ).observe(duration)

    def record_extraction(self, parsed: ParsedDocument) -> None:
        unique_pages = {block.page for block in parsed.blocks}
        MINERU_PDF_PAGES_PROCESSED_TOTAL.labels(worker_id=self._worker_id).inc(
            len(unique_pages)
        )
        MINERU_TABLE_EXTRACTION_COUNT.labels(worker_id=self._worker_id).observe(
            len(parsed.tables)
        )
        MINERU_FIGURE_EXTRACTION_COUNT.labels(worker_id=self._worker_id).observe(
            len(parsed.figures)
        )


class MineruPipeline:
    """Runs the MinerU CLI, parser, and post-processor for a batch of requests."""

    def __init__(
        self,
        *,
        parser: MineruOutputParser,
        postprocessor: MineruPostProcessor,
        metrics: PipelineMetrics,
    ) -> None:
        self._parser = parser
        self._postprocessor = postprocessor
        self._metrics = metrics

    def execute(
        self,
        *,
        requests: Sequence[MineruRequest],
        cli_inputs: Sequence[MineruCliInput],
        execute_cli: Callable[[Sequence[MineruCliInput]], tuple[MineruCliResult, str, int]],
        metadata_builder: Callable[..., ProcessingMetadata],
        batch_index: int,
        total_batches: int,
        record_gpu_memory: Callable[[str], None] | None = None,
    ) -> MineruBatchResponse:
        if not requests:
            now = datetime.now(timezone.utc)
            return MineruBatchResponse(
                documents=[], processed_at=now, duration_seconds=0.0, metadata=[]
            )

        request_map = {request.document_id: request for request in requests}
        logger.bind(
            batch=batch_index,
            total_batches=total_batches,
            size=len(requests),
        ).info("mineru.pipeline.batch_started")

        start_monotonic = time.monotonic()
        started_at = datetime.now(timezone.utc)
        cli_result, gpu_label, planned_memory_mb = execute_cli(cli_inputs)
        completed_at = datetime.now(timezone.utc)
        duration = time.monotonic() - start_monotonic

        if not cli_result.outputs:
            raise MineruCliError("MinerU CLI returned no outputs")

        self._metrics.record_cli_duration(gpu_label, cli_result.duration_seconds)
        if record_gpu_memory and gpu_label.startswith("cuda:"):
            record_gpu_memory(gpu_label)

        documents: list[Document] = []
        metadata_entries: list[ProcessingMetadata] = []

        for output in cli_result.outputs:
            request = request_map.get(output.document_id)
            if request is None:
                logger.bind(
                    document_id=output.document_id,
                    batch=batch_index,
                ).warning("mineru.pipeline.output_without_request")
                continue
            try:
                parsed = self._parser.parse_path(output.path)
            except MineruOutputParserError as exc:
                logger.bind(
                    document_id=output.document_id,
                    batch=batch_index,
                    error=str(exc),
                ).error("mineru.pipeline.parse_failed")
                raise

            metadata = metadata_builder(
                request=request,
                parsed=parsed,
                gpu_label=gpu_label,
                started_at=started_at,
                completed_at=completed_at,
                cli_result=cli_result,
                planned_memory_mb=planned_memory_mb,
            )
            document = self._postprocessor.build_document(
                parsed, request, metadata.as_dict()
            )
            documents.append(document)
            metadata_entries.append(metadata)
            self._metrics.record_extraction(parsed)
            logger.bind(
                document_id=document.document_id,
                blocks=len(document.blocks),
                tables=len(document.tables),
                figures=len(document.figures),
                equations=len(document.equations),
                batch=batch_index,
                total_batches=total_batches,
            ).info("mineru.pipeline.document_completed")

        logger.bind(
            batch=batch_index,
            total_batches=total_batches,
            gpu=gpu_label,
            planned_memory_mb=planned_memory_mb,
            documents=len(documents),
            duration=round(duration, 4),
        ).info("mineru.pipeline.batch_completed")

        return MineruBatchResponse(
            documents=documents,
            processed_at=completed_at,
            duration_seconds=duration,
            metadata=metadata_entries,
        )


__all__ = ["MineruPipeline", "PipelineMetrics"]
