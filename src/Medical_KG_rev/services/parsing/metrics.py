"""Prometheus metrics for Docling Gemma3 VLM processing."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


DOCLING_VLM_PROCESSING_SECONDS = Histogram(
    "docling_vlm_processing_seconds",
    "Time spent processing a PDF document with the Docling VLM backend.",
    labelnames=("tenant_id", "outcome"),
)

DOCLING_VLM_GPU_MEMORY_MB = Gauge(
    "docling_vlm_gpu_memory_mb",
    "GPU memory usage observed during Docling VLM execution (in megabytes).",
    labelnames=("device", "state"),
)

DOCLING_VLM_MODEL_LOAD_SECONDS = Histogram(
    "docling_vlm_model_load_seconds",
    "Time required to materialise the Gemma3 VLM pipeline in memory.",
    labelnames=("revision",),
)

DOCLING_VLM_REQUESTS_TOTAL = Counter(
    "docling_vlm_requests_total",
    "Number of Docling VLM requests grouped by terminal outcome.",
    labelnames=("outcome",),
)

DOCLING_VLM_RETRY_ATTEMPTS_TOTAL = Counter(
    "docling_vlm_retry_attempts_total",
    "Number of retry attempts triggered for Docling VLM processing.",
    labelnames=("reason",),
)


__all__ = [
    "DOCLING_VLM_GPU_MEMORY_MB",
    "DOCLING_VLM_MODEL_LOAD_SECONDS",
    "DOCLING_VLM_PROCESSING_SECONDS",
    "DOCLING_VLM_REQUESTS_TOTAL",
    "DOCLING_VLM_RETRY_ATTEMPTS_TOTAL",
]

