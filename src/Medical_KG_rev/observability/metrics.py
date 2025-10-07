"""Prometheus metric registration and helpers."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Mapping
from time import perf_counter
from typing import TYPE_CHECKING, Any

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch is optional in CPU-only environments
    torch = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from fastapi import FastAPI, Request, Response
else:  # pragma: no cover - optional dependency handling
    try:
        from fastapi import FastAPI, Request, Response
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        FastAPI = Request = Response = Any  # type: ignore[assignment]
        _FASTAPI_IMPORT_ERROR: ModuleNotFoundError | None = exc
    else:
        _FASTAPI_IMPORT_ERROR = None

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from Medical_KG_rev.config.settings import AppSettings
else:  # pragma: no cover - optional dependency handling
    try:
        from Medical_KG_rev.config.settings import AppSettings
    except (ModuleNotFoundError, ImportError) as exc:  # pragma: no cover - optional dependency
        AppSettings = Any  # type: ignore[assignment]
        _APP_SETTINGS_IMPORT_ERROR: Exception | None = exc
    else:
        _APP_SETTINGS_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency
    from prometheus_client import (  # type: ignore
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

    class _NoopMetric:
        def labels(self, *args, **kwargs):  # noqa: ANN002, ANN003 - signature mirrors prometheus
            return self

        def inc(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            return None

        def observe(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            return None

        def set(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            return None

    def _noop_metric(*args, **kwargs):  # noqa: ANN002, ANN003
        return _NoopMetric()

    Counter = Gauge = Histogram = _noop_metric  # type: ignore[assignment]

    def generate_latest() -> bytes:  # type: ignore[override]
        return b""

try:  # pragma: no cover - optional dependency chain
    from Medical_KG_rev.utils.logging import (
        bind_correlation_id,
        get_correlation_id,
        reset_correlation_id,
    )
except Exception:  # pragma: no cover - optional dependency chain
    def bind_correlation_id(*args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def get_correlation_id() -> str | None:
        return None

    def reset_correlation_id(_token) -> None:  # noqa: ANN001
        return None

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
INGESTION_DOCUMENTS = Counter(
    "orchestration_ingestion_documents_total",
    "Total documents submitted to ingestion pipelines",
    labelnames=("pipeline",),
)
INGESTION_STAGE_LATENCY = Histogram(
    "orchestration_ingestion_stage_latency_seconds",
    "Latency per ingestion stage",
    labelnames=("stage",),
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5),
)
INGESTION_STAGE_ERRORS = Counter(
    "orchestration_ingestion_stage_errors_total",
    "Errors observed per ingestion stage",
    labelnames=("stage", "error_type"),
)
QUERY_OPERATIONS = Counter(
    "orchestration_query_operations_total",
    "Count of query pipeline operations by status",
    labelnames=("pipeline", "tenant", "status"),
)
QUERY_LATENCY = Histogram(
    "orchestration_query_latency_seconds",
    "End-to-end query latency",
    labelnames=("pipeline",),
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1, 2),
)
JOB_STATUS = Gauge(
    "orchestration_jobs_total",
    "Job ledger counts per status",
    labelnames=("status",),
)
ORCHESTRATION_TIMEOUTS = Counter(
    "orchestration_stage_timeouts_total",
    "Number of stages exceeding timeout",
    labelnames=("operation", "stage"),
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


def _normalise_path(request: "Request") -> str:
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


def register_metrics(app: "FastAPI", settings: AppSettings) -> None:
    if "_FASTAPI_IMPORT_ERROR" in globals() and _FASTAPI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "FastAPI is required to register metrics. Install the 'fastapi' extra."
        ) from _FASTAPI_IMPORT_ERROR
    if "_APP_SETTINGS_IMPORT_ERROR" in globals() and _APP_SETTINGS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Medical_KG_rev configuration settings are unavailable. Install project dependencies."
        ) from _APP_SETTINGS_IMPORT_ERROR

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


def _observe_with_exemplar(metric, labels: tuple[str, ...], value: float) -> None:
    labelled = metric.labels(*labels)
    correlation_id = get_correlation_id()
    kwargs: dict[str, object] = {}
    if correlation_id:
        try:  # pragma: no cover - exemplar support optional
            kwargs["exemplar"] = {"correlation_id": correlation_id}
        except TypeError:
            kwargs = {}
    labelled.observe(max(value, 0.0), **kwargs)


def _increment_with_exemplar(metric, labels: tuple[str, ...], amount: float = 1.0) -> None:
    labelled = metric.labels(*labels)
    correlation_id = get_correlation_id()
    kwargs: dict[str, object] = {}
    if correlation_id:
        try:  # pragma: no cover - exemplar support optional
            kwargs["exemplar"] = {"correlation_id": correlation_id}
        except TypeError:
            kwargs = {}
    labelled.inc(amount, **kwargs)
        return response

    @app.get(path, include_in_schema=False)
    async def metrics_endpoint() -> "Response":
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def observe_job_duration(operation: str, duration_seconds: float) -> None:
    JOB_DURATION.labels(operation=operation).observe(max(duration_seconds, 0.0))


def record_business_event(event: str, amount: int = 1) -> None:
    BUSINESS_EVENTS.labels(event=event).inc(amount)


def observe_chunking_latency(profile: str, duration_seconds: float) -> None:
    CHUNKING_LATENCY.labels(profile=profile).observe(max(duration_seconds, 0.0))


def record_chunk_size(profile: str, granularity: str, characters: int) -> None:
    CHUNK_SIZE.labels(profile=profile, granularity=granularity).observe(max(characters, 0))


def set_chunking_circuit_state(state: int) -> None:
    CHUNKING_CIRCUIT_STATE.set(float(state))
