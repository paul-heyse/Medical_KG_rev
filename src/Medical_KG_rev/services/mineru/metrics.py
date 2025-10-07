"""Prometheus metrics emitted by the MinerU GPU integration."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

MINERU_PROCESSING_DURATION_SECONDS = Histogram(
    "mineru_processing_duration_seconds",
    "Time spent running the MinerU CLI for a batch of PDFs.",
    labelnames=("worker_id", "gpu_id"),
)

MINERU_GPU_MEMORY_USAGE_BYTES = Gauge(
    "mineru_gpu_memory_usage_bytes",
    "Observed GPU memory usage for MinerU workers (in bytes).",
    labelnames=("gpu_id", "state"),
)

MINERU_CLI_FAILURES_TOTAL = Counter(
    "mineru_cli_failures_total",
    "Number of MinerU CLI failures grouped by reason.",
    labelnames=("reason",),
)

MINERU_PDF_PAGES_PROCESSED_TOTAL = Counter(
    "mineru_pdf_pages_processed_total",
    "Total PDF pages processed by MinerU workers.",
    labelnames=("worker_id",),
)

MINERU_TABLE_EXTRACTION_COUNT = Histogram(
    "mineru_table_extraction_count",
    "Number of tables extracted per document processed by MinerU.",
    labelnames=("worker_id",),
)

MINERU_FIGURE_EXTRACTION_COUNT = Histogram(
    "mineru_figure_extraction_count",
    "Number of figures extracted per document processed by MinerU.",
    labelnames=("worker_id",),
)

MINERU_WORKER_QUEUE_DEPTH = Gauge(
    "mineru_worker_queue_depth",
    "Depth of the MinerU worker queues, labelled by worker identifier.",
    labelnames=("worker_id",),
)

__all__ = [
    "MINERU_PROCESSING_DURATION_SECONDS",
    "MINERU_GPU_MEMORY_USAGE_BYTES",
    "MINERU_CLI_FAILURES_TOTAL",
    "MINERU_PDF_PAGES_PROCESSED_TOTAL",
    "MINERU_TABLE_EXTRACTION_COUNT",
    "MINERU_FIGURE_EXTRACTION_COUNT",
    "MINERU_WORKER_QUEUE_DEPTH",
]
