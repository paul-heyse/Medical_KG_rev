"""Prometheus metrics for Docling VLM processing."""

from __future__ import annotations

import importlib

_prom = importlib.util.find_spec("prometheus_client")

if _prom is not None:  # pragma: no cover - optional runtime dependency
    from prometheus_client import Counter, Histogram

    DOCLING_PROCESSING_SECONDS = Histogram(
        "docling_vlm_processing_seconds",
        "Time spent processing PDFs with Docling VLM",
        buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
        labelnames=("status",),
    )
    DOCLING_GPU_MEMORY_MB = Histogram(
        "docling_vlm_gpu_memory_mb",
        "GPU memory requested for Docling VLM invocations",
        buckets=(4096, 8192, 12288, 16384, 20480, 24576, 32768),
    )
    DOCLING_RETRIES_TOTAL = Counter(
        "docling_vlm_retries_total",
        "Number of retry attempts performed by Docling VLM",
    )
else:  # pragma: no cover - provide no-op fallbacks

    class _NoopMetric:  # type: ignore[too-many-ancestors]
        def labels(self, *_, **__):
            return self

        def observe(self, *_: object, **__: object) -> None:
            return None

        def inc(self, *_: object, **__: object) -> None:
            return None

    DOCLING_PROCESSING_SECONDS = _NoopMetric()
    DOCLING_GPU_MEMORY_MB = _NoopMetric()
    DOCLING_RETRIES_TOTAL = _NoopMetric()


__all__ = [
    "DOCLING_GPU_MEMORY_MB",
    "DOCLING_PROCESSING_SECONDS",
    "DOCLING_RETRIES_TOTAL",
]
