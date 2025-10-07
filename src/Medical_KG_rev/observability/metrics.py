"""Prometheus metric registration and helpers."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from time import perf_counter
from typing import Any

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch is optional in CPU-only environments
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from fastapi import FastAPI, Request, Response
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    FastAPI = None  # type: ignore[assignment]
    Request = Any  # type: ignore[assignment]
    Response = Any  # type: ignore[assignment]

from prometheus_client import (  # type: ignore
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from Medical_KG_rev.config.settings import AppSettings
from Medical_KG_rev.utils.logging import (
    bind_correlation_id,
    get_correlation_id,
    reset_correlation_id,
)

REQUEST_COUNTER = Counter(
    "api_requests",
    "Total HTTP requests served by the gateway",
    labelnames=("method", "path", "status"),
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "HTTP request latency distribution",
    labelnames=("method", "path"),
)
JOB_DURATION = Histogram(
    "job_duration_seconds",
    "Duration of ingest/retrieve operations",
    labelnames=("operation",),
)
CHUNKING_LATENCY = Histogram(
    "chunking_latency_seconds",
    "Latency distribution for chunking profiles",
    labelnames=("profile",),
)
CHUNK_SIZE = Histogram(
    "chunk_size_characters",
    "Distribution of chunk sizes by granularity",
    labelnames=("profile", "granularity"),
)
CHUNKING_CIRCUIT_STATE = Gauge(
    "chunking_circuit_breaker_state",
    "Circuit breaker state for chunking pipeline (0=closed, 1=open, 2=half-open)",
)
GPU_UTILISATION = Gauge(
    "gpu_utilization_percent",
    "GPU memory utilisation percentage",
    labelnames=("gpu",),
)
BUSINESS_EVENTS = Counter(
    "business_events",
    "Business event counters (documents ingested, retrievals)",
    labelnames=("event",),
)
RERANK_OPERATIONS = Counter(
    "reranking_operations_total",
    "Total reranking invocations",
    labelnames=("reranker", "tenant", "batch_size"),
)
RERANK_DURATION = Histogram(
    "reranking_duration_seconds",
    "Distribution of reranking latencies",
    labelnames=("reranker", "tenant"),
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0),
)
RERANK_ERRORS = Counter(
    "reranking_errors_total",
    "Number of reranking failures grouped by type",
    labelnames=("reranker", "error_type"),
)
RERANK_PAIRS = Counter(
    "reranking_pairs_processed_total",
    "Number of query/document pairs scored",
    labelnames=("reranker",),
)
RERANK_CIRCUIT = Gauge(
    "reranking_circuit_breaker_state",
    "Circuit breaker state per reranker (1=open)",
    labelnames=("reranker", "tenant"),
)
RERANK_GPU = Gauge(
    "reranking_gpu_utilization_percent",
    "GPU utilisation while reranking",
    labelnames=("reranker",),
)
PIPELINE_STAGE_DURATION = Histogram(
    "retrieval_pipeline_stage_duration_seconds",
    "Latency per stage of the retrieval pipeline",
    labelnames=("stage",),
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0),
)
RERANK_CACHE_HIT = Gauge(
    "reranking_cache_hit_rate",
    "Cache hit rate for reranker results",
    labelnames=("reranker",),
)
RERANK_LATENCY_ALERTS = Counter(
    "reranking_latency_alerts_total",
    "Number of reranking operations breaching latency SLOs",
    labelnames=("reranker",),
)
RERANK_GPU_MEMORY_ALERTS = Counter(
    "reranking_gpu_memory_alerts_total",
    "Alerts fired when GPU memory is exhausted during reranking",
    labelnames=("reranker",),
)


def _normalise_path(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", request.url.path)


def _update_gpu_metrics() -> None:
    if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():  # type: ignore[attr-defined]
        GPU_UTILISATION.labels(gpu="0").set(0.0)
        return

    for index in range(torch.cuda.device_count()):  # type: ignore[attr-defined]
        total = torch.cuda.get_device_properties(index).total_memory  # type: ignore[attr-defined]
        used = torch.cuda.memory_allocated(index)  # type: ignore[attr-defined]
        utilisation = float(used) / float(total) * 100 if total else 0.0
        GPU_UTILISATION.labels(gpu=str(index)).set(utilisation)


def register_metrics(app: FastAPI, settings: AppSettings) -> None:  # type: ignore[valid-type]
    if FastAPI is None:
        return
    if not settings.observability.metrics.enabled:
        return

    path = settings.observability.metrics.path
    correlation_header = settings.observability.logging.correlation_id_header

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next: Callable):
        if request.url.path == path:
            return await call_next(request)

        current = get_correlation_id()
        provided = request.headers.get(correlation_header) if correlation_header else None
        fallback = getattr(request.state, "correlation_id", None)
        correlation_id = current or provided or fallback or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        token = None if current else bind_correlation_id(correlation_id)
        timer = perf_counter()

        try:
            response: Response = await call_next(request)
        except Exception:
            duration = perf_counter() - timer
            REQUEST_COUNTER.labels(request.method, _normalise_path(request), "500").inc()
            REQUEST_LATENCY.labels(request.method, _normalise_path(request)).observe(duration)
            if token is not None:
                reset_correlation_id(token)
            raise

        duration = perf_counter() - timer
        path_template = _normalise_path(request)
        REQUEST_COUNTER.labels(request.method, path_template, str(response.status_code)).inc()
        REQUEST_LATENCY.labels(request.method, path_template).observe(duration)
        _update_gpu_metrics()

        if correlation_header:
            response.headers.setdefault(correlation_header, correlation_id)

        if token is not None:
            reset_correlation_id(token)
        return response

    @app.get(path, include_in_schema=False)
    async def metrics_endpoint() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def observe_job_duration(operation: str, duration_seconds: float) -> None:
    JOB_DURATION.labels(operation=operation).observe(max(duration_seconds, 0.0))


def record_business_event(event: str, amount: int = 1) -> None:
    BUSINESS_EVENTS.labels(event=event).inc(amount)


def record_reranking_operation(
    reranker: str,
    tenant: str,
    batch_size: int,
    duration_seconds: float,
    pairs: int,
    circuit_state: str,
    gpu_utilisation: float | None = None,
) -> None:
    RERANK_OPERATIONS.labels(reranker, tenant, str(batch_size)).inc()
    RERANK_DURATION.labels(reranker, tenant).observe(max(duration_seconds, 0.0))
    RERANK_PAIRS.labels(reranker).inc(max(pairs, 0))
    RERANK_CIRCUIT.labels(reranker, tenant).set(1.0 if circuit_state == "open" else 0.0)
    if gpu_utilisation is not None:
        RERANK_GPU.labels(reranker).set(max(gpu_utilisation, 0.0))


def record_reranking_error(reranker: str, error_type: str) -> None:
    RERANK_ERRORS.labels(reranker, error_type).inc()


def record_pipeline_stage(stage: str, duration_seconds: float) -> None:
    PIPELINE_STAGE_DURATION.labels(stage).observe(max(duration_seconds, 0.0))


def record_cache_hit_rate(reranker: str, hit_rate: float) -> None:
    RERANK_CACHE_HIT.labels(reranker).set(max(0.0, min(hit_rate, 1.0)))


def record_latency_alert(reranker: str, duration_seconds: float, slo_seconds: float) -> None:
    if duration_seconds > slo_seconds:
        RERANK_LATENCY_ALERTS.labels(reranker).inc()


def record_gpu_memory_alert(reranker: str) -> None:
    RERANK_GPU_MEMORY_ALERTS.labels(reranker).inc()
