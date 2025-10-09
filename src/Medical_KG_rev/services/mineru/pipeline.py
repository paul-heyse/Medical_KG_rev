"""Reusable MinerU pipeline orchestration helpers.

This module provides orchestration utilities for running MinerU CLI
batches, including metrics collection, error handling, and result
processing. It coordinates between the CLI wrapper, output parser,
and post-processor to produce structured document outputs.

Key Components:
    - MineruPipeline: Main orchestration class for batch processing
    - PipelineMetrics: Prometheus metrics collection for batches
    - Error handling and logging for pipeline operations

Responsibilities:
    - Orchestrate MinerU CLI execution for batches
    - Collect and emit Prometheus metrics
    - Handle errors and logging throughout the pipeline
    - Coordinate parsing and post-processing of results
    - Track processing duration and resource usage

Collaborators:
    - MinerU CLI wrapper for execution
    - Output parser for result processing
    - Post-processor for document construction
    - Metrics system for observability

Side Effects:
    - Emits Prometheus metrics for monitoring
    - Logs pipeline operations and statistics
    - May raise exceptions for processing failures

Thread Safety:
    - Thread-safe: Pipeline instances are stateless
    - Metrics collection is thread-safe

Performance Characteristics:
    - O(n) processing time for n documents in batch
    - Memory usage scales with batch size
    - Efficient metrics collection with minimal overhead
    - Supports concurrent batch processing

Example:
    >>> pipeline = MineruPipeline(parser=parser, postprocessor=postprocessor, metrics=metrics)
    >>> response = pipeline.execute(
    ...     requests=requests,
    ...     cli_inputs=cli_inputs,
    ...     execute_cli=execute_cli,
    ...     metadata_builder=metadata_builder,
    ...     batch_index=0,
    ...     total_batches=1
    ... )
    >>> assert len(response.documents) > 0
"""

from __future__ import annotations

# ==============================================================================
# IMPORTS
# ==============================================================================
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

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

logger = structlog.get_logger(__name__)


# ==============================================================================
# METRICS COLLECTION
# ==============================================================================

class PipelineMetrics:
    """Encapsulates Prometheus metric emission for MinerU batches.

    This class provides a convenient interface for collecting and emitting
    Prometheus metrics during MinerU batch processing. It handles metric
    labeling and aggregation for monitoring pipeline performance.

    Attributes:
        _worker_id: Unique identifier for the worker instance

    Thread Safety:
        - Thread-safe: Prometheus client handles concurrent access
        - All metric operations are atomic

    Example:
        >>> metrics = PipelineMetrics("worker-1")
        >>> metrics.record_cli_duration("cuda:0", 120.5)
        >>> metrics.record_extraction(parsed_document)
    """

    def __init__(self, worker_id: str) -> None:
        """Initialize pipeline metrics collector.

        Args:
            worker_id: Unique identifier for the worker instance

        Example:
            >>> metrics = PipelineMetrics("worker-1")
            >>> assert metrics._worker_id == "worker-1"
        """
        self._worker_id = worker_id

    def record_cli_duration(self, gpu_label: str, duration: float) -> None:
        """Record CLI execution duration.

        Args:
            gpu_label: GPU identifier (e.g., "cuda:0")
            duration: Duration in seconds

        Example:
            >>> metrics = PipelineMetrics("worker-1")
            >>> metrics.record_cli_duration("cuda:0", 120.5)
        """
        MINERU_PROCESSING_DURATION_SECONDS.labels(
            worker_id=self._worker_id, gpu_id=gpu_label
        ).observe(duration)

    def record_extraction(self, parsed: ParsedDocument) -> None:
        """Record extraction statistics for a parsed document.

        Args:
            parsed: Parsed document with extracted content

        Example:
            >>> metrics = PipelineMetrics("worker-1")
            >>> metrics.record_extraction(parsed_document)
        """
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


# ==============================================================================
# PIPELINE ORCHESTRATION
# ==============================================================================

class MineruPipeline:
    """Runs the MinerU CLI, parser, and post-processor for a batch of requests.

    This class orchestrates the complete MinerU processing pipeline for
    batches of documents, including CLI execution, output parsing,
    post-processing, and metrics collection. It handles error recovery
    and provides comprehensive logging throughout the process.

    Attributes:
        _parser: Output parser for MinerU results
        _postprocessor: Post-processor for document construction
        _metrics: Metrics collector for observability

    Thread Safety:
        - Thread-safe: Pipeline instances are stateless
        - All operations are atomic within a single batch

    Example:
        >>> pipeline = MineruPipeline(
        ...     parser=parser,
        ...     postprocessor=postprocessor,
        ...     metrics=metrics
        ... )
        >>> response = pipeline.execute(
        ...     requests=requests,
        ...     cli_inputs=cli_inputs,
        ...     execute_cli=execute_cli,
        ...     metadata_builder=metadata_builder,
        ...     batch_index=0,
        ...     total_batches=1
        ... )
    """

    def __init__(
        self,
        *,
        parser: MineruOutputParser,
        postprocessor: MineruPostProcessor,
        metrics: PipelineMetrics,
    ) -> None:
        """Initialize MinerU pipeline.

        Args:
            parser: Output parser for MinerU results
            postprocessor: Post-processor for document construction
            metrics: Metrics collector for observability

        Example:
            >>> pipeline = MineruPipeline(
            ...     parser=parser,
            ...     postprocessor=postprocessor,
            ...     metrics=metrics
            ... )
        """
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
        """Execute MinerU pipeline for a batch of requests.

        Args:
            requests: Sequence of MinerU requests to process
            cli_inputs: Sequence of CLI inputs for execution
            execute_cli: Function to execute MinerU CLI
            metadata_builder: Function to build processing metadata
            batch_index: Index of current batch
            total_batches: Total number of batches
            record_gpu_memory: Optional function to record GPU memory usage

        Returns:
            Batch response with processed documents and metadata

        Raises:
            MineruCliError: If CLI execution fails
            MineruOutputParserError: If output parsing fails

        Example:
            >>> response = pipeline.execute(
            ...     requests=requests,
            ...     cli_inputs=cli_inputs,
            ...     execute_cli=execute_cli,
            ...     metadata_builder=metadata_builder,
            ...     batch_index=0,
            ...     total_batches=1
            ... )
            >>> assert len(response.documents) > 0
        """
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


# ==============================================================================
# EXPORTS
# ==============================================================================


__all__ = ["MineruPipeline", "PipelineMetrics"]
