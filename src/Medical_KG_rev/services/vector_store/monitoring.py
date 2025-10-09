"""Prometheus metrics for vector store operations."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

VECTOR_OPERATION_LATENCY = Histogram(
    "vector_operation_duration_seconds",
    "Latency distribution for vector store operations",
    labelnames=("operation", "namespace"),
)

VECTOR_OPERATION_COUNTER = Counter(
    "vector_operation_total",
    "Total vector store operations executed",
    labelnames=("operation", "namespace"),
)

VECTOR_MEMORY_USAGE = Gauge(
    "vector_backend_memory_bytes",
    "Observed memory usage for evaluated payloads",
    labelnames=("namespace",),
)

VECTOR_COMPRESSION_RATIO = Gauge(
    "vector_compression_ratio",
    "Compression ratio achieved for a given policy",
    labelnames=("namespace", "policy"),
)


def record_vector_operation(operation: str, namespace: str, duration_seconds: float, count: int) -> None:
    VECTOR_OPERATION_LATENCY.labels(operation=operation, namespace=namespace).observe(max(duration_seconds, 0.0))
    VECTOR_OPERATION_COUNTER.labels(operation=operation, namespace=namespace).inc(count)


def record_memory_usage(namespace: str, bytes_used: int) -> None:
    VECTOR_MEMORY_USAGE.labels(namespace=namespace).set(float(bytes_used))


def record_compression_ratio(namespace: str, policy: str, ratio: float) -> None:
    VECTOR_COMPRESSION_RATIO.labels(namespace=namespace, policy=policy).set(max(ratio, 0.0))


__all__ = [
    "record_compression_ratio",
    "record_memory_usage",
    "record_vector_operation",
]

