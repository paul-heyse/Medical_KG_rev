"""Prometheus metric registration and helpers."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from time import perf_counter

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch is optional in CPU-only environments
    torch = None  # type: ignore

from fastapi import FastAPI, Request, Response
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
ORCHESTRATION_OPERATIONS = Counter(
    "orchestration_operations_total",
    "Total orchestration operations grouped by operation/tenant/status",
    labelnames=("operation", "tenant", "status"),
)
ORCHESTRATION_END_TO_END_DURATION = Histogram(
    "orchestration_end_to_end_duration_seconds",
    "Distribution of end-to-end orchestration latency",
    labelnames=("operation", "pipeline"),
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
)
ORCHESTRATION_STAGE_DURATION = Histogram(
    "orchestration_stage_duration_seconds",
    "Latency distribution per orchestration stage",
    labelnames=("operation", "stage"),
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5),
)
ORCHESTRATION_ERRORS = Counter(
    "orchestration_errors_total",
    "Total orchestration errors grouped by operation/stage/type",
    labelnames=("operation", "stage", "error_type"),
)
ORCHESTRATION_QUEUE_DEPTH = Gauge(
    "orchestration_job_queue_depth",
    "Number of queued jobs per orchestration stage",
    labelnames=("stage",),
)
ORCHESTRATION_CIRCUIT_STATE = Gauge(
    "orchestration_circuit_breaker_state",
    "Circuit breaker state for orchestration dependencies",
    labelnames=("service",),
)
ORCHESTRATION_DLQ_EVENTS = Counter(
    "orchestration_dead_letter_total",
    "Total messages routed to orchestration dead letter queue",
    labelnames=("stage", "error_type"),
)
ORCHESTRATION_DLQ_DEPTH = Gauge(
    "orchestration_dead_letter_queue_depth",
    "Depth of the orchestration dead letter queue",
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


def register_metrics(app: FastAPI, settings: AppSettings) -> None:
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


def record_orchestration_operation(operation: str, tenant: str, status: str) -> None:
    """Increment orchestration operation counter."""

    ORCHESTRATION_OPERATIONS.labels(operation, tenant, status).inc()


def observe_orchestration_duration(operation: str, pipeline: str, duration: float) -> None:
    """Record end-to-end orchestration latency."""

    ORCHESTRATION_END_TO_END_DURATION.labels(operation, pipeline).observe(max(duration, 0.0))


def observe_orchestration_stage(operation: str, stage: str, duration: float) -> None:
    """Record per-stage orchestration latency."""

    ORCHESTRATION_STAGE_DURATION.labels(operation, stage).observe(max(duration, 0.0))


def record_orchestration_error(operation: str, stage: str, error_type: str) -> None:
    """Increment orchestration error counters."""

    ORCHESTRATION_ERRORS.labels(operation, stage, error_type).inc()


def set_orchestration_queue_depth(stage: str, depth: int) -> None:
    """Update queue depth gauge for a stage."""

    ORCHESTRATION_QUEUE_DEPTH.labels(stage).set(max(depth, 0))


def set_orchestration_circuit_state(service: str, state: str) -> None:
    """Set circuit breaker state gauge for a downstream service."""

    value = {"closed": 0, "half_open": 0.5, "open": 1}.get(state, 0)
    ORCHESTRATION_CIRCUIT_STATE.labels(service).set(value)


def record_dead_letter_event(stage: str, error_type: str) -> None:
    """Track when a message is routed to the dead letter queue."""

    ORCHESTRATION_DLQ_EVENTS.labels(stage, error_type).inc()


def set_dead_letter_queue_depth(depth: int) -> None:
    """Update the depth of the orchestration dead letter queue."""

    ORCHESTRATION_DLQ_DEPTH.set(max(depth, 0))
