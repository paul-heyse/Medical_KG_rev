"""Prometheus metrics emitted by the MinerU GPU integration.

This module defines Prometheus metrics for monitoring MinerU GPU
integration performance, including processing duration, GPU memory
usage, failure rates, and extraction counts. These metrics provide
observability into the MinerU service operations and help identify
performance bottlenecks and failures.

Key Components:
    - Processing duration metrics for batch operations
    - GPU memory usage monitoring
    - Failure tracking by reason
    - Extraction count histograms
    - Worker queue depth monitoring

Responsibilities:
    - Define Prometheus metric collectors for MinerU operations
    - Provide labeled metrics for multi-worker environments
    - Track performance and resource utilization
    - Monitor failure rates and error patterns
    - Enable alerting on critical thresholds

Collaborators:
    - MinerU service for metric collection
    - Prometheus server for metric scraping
    - Grafana for metric visualization
    - Alerting systems for threshold monitoring

Side Effects:
    - Creates global metric collectors
    - Registers metrics with Prometheus registry
    - Consumes minimal memory for metric storage

Thread Safety:
    - Thread-safe: Prometheus client handles concurrent access
    - All metric operations are atomic

Performance Characteristics:
    - Minimal overhead for metric collection
    - Efficient storage with label-based organization
    - Supports high-frequency metric updates
    - Memory usage scales with unique label combinations

Example:
    >>> from Medical_KG_rev.services.mineru.metrics import MINERU_PROCESSING_DURATION_SECONDS
    >>> with MINERU_PROCESSING_DURATION_SECONDS.labels(worker_id="worker-1", gpu_id="gpu-0").time():
    ...     # MinerU processing code
    ...     pass

"""

from __future__ import annotations

# ==============================================================================
# IMPORTS
# ==============================================================================
from prometheus_client import Counter, Gauge, Histogram

# ==============================================================================
# METRIC DEFINITIONS
# ==============================================================================

# Processing Duration Metrics
MINERU_PROCESSING_DURATION_SECONDS = Histogram(
    "mineru_processing_duration_seconds",
    "Time spent running the MinerU CLI for a batch of PDFs.",
    labelnames=("worker_id", "gpu_id"),
)

# Resource Usage Metrics
MINERU_GPU_MEMORY_USAGE_BYTES = Gauge(
    "mineru_gpu_memory_usage_bytes",
    "Observed GPU memory usage for MinerU workers (in bytes).",
    labelnames=("gpu_id", "state"),
)

# Failure Tracking Metrics
MINERU_CLI_FAILURES_TOTAL = Counter(
    "mineru_cli_failures_total",
    "Number of MinerU CLI failures grouped by reason.",
    labelnames=("reason",),
)

# Throughput Metrics
MINERU_PDF_PAGES_PROCESSED_TOTAL = Counter(
    "mineru_pdf_pages_processed_total",
    "Total PDF pages processed by MinerU workers.",
    labelnames=("worker_id",),
)

# Extraction Quality Metrics
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

# Queue Depth Metrics
MINERU_WORKER_QUEUE_DEPTH = Gauge(
    "mineru_worker_queue_depth",
    "Depth of the MinerU worker queues, labelled by worker identifier.",
    labelnames=("worker_id",),
)

# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "MINERU_CLI_FAILURES_TOTAL",
    "MINERU_FIGURE_EXTRACTION_COUNT",
    "MINERU_GPU_MEMORY_USAGE_BYTES",
    "MINERU_PDF_PAGES_PROCESSED_TOTAL",
    "MINERU_PROCESSING_DURATION_SECONDS",
    "MINERU_TABLE_EXTRACTION_COUNT",
    "MINERU_WORKER_QUEUE_DEPTH",
]
