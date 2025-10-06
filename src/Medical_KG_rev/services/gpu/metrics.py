"""Prometheus metrics for GPU microservices."""

from __future__ import annotations

from prometheus_client import Gauge

GPU_UTILIZATION = Gauge(
    "gpu_service_utilization_percent",
    "Reported GPU utilization percent for a microservice.",
    labelnames=("service", "device"),
)

GPU_MEMORY_USED = Gauge(
    "gpu_service_memory_megabytes",
    "GPU memory consumption in megabytes for a microservice.",
    labelnames=("service", "device", "state"),
)

__all__ = ["GPU_MEMORY_USED", "GPU_UTILIZATION"]
