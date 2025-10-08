"""Prometheus metric registration and helpers."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from time import perf_counter
from typing import Any, Mapping

import structlog

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

from Medical_KG_rev.config.settings import AppSettings
from Medical_KG_rev.observability.alerts import get_alert_manager
from Medical_KG_rev.utils.logging import (
    bind_correlation_id,
    get_correlation_id,
    reset_correlation_id,
)
from prometheus_client import (  # type: ignore
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

MINERU_VLLM_REQUEST_DURATION = Histogram(
    "mineru_vllm_request_duration_seconds",
    "Duration of MinerU → vLLM API requests",
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

MINERU_VLLM_CLIENT_FAILURES = Counter(
    "mineru_vllm_client_failures_total",
    "Total MinerU vLLM client failures grouped by error type",
    labelnames=("error_type",),
)

MINERU_VLLM_CLIENT_RETRIES = Counter(
    "mineru_vllm_client_retries_total",
    "Number of retry attempts performed by the MinerU vLLM client",
    labelnames=("retry_number",),
)

MINERU_VLLM_CIRCUIT_BREAKER_STATE = Gauge(
    "mineru_vllm_circuit_breaker_state",
    "Circuit breaker state for MinerU vLLM client (0=closed, 1=half-open, 2=open)",
)

logger = structlog.get_logger(__name__)

REQUEST_COUNTER = Counter(
    "api_requests",
    "Total HTTP requests served by the gateway",
    labelnames=("method", "path", "status"),
)
CROSS_TENANT_ACCESS_ATTEMPTS = Counter(
    "medicalkg_cross_tenant_access_attempts_total",
    "Attempted cross-tenant accesses (blocked)",
    labelnames=("source_tenant", "target_tenant"),
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
CHUNKING_DOCUMENTS = Counter(
    "chunking_documents_total",
    "Total documents processed by the chunking pipeline",
    labelnames=("profile",),
)
CHUNKING_DURATION = Histogram(
    "chunking_duration_seconds",
    "Chunking duration distribution per profile",
    labelnames=("profile",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)
CHUNKS_PER_DOCUMENT = Histogram(
    "chunking_chunks_per_document",
    "Distribution of chunk counts per document",
    labelnames=("profile",),
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
)
CHUNKING_FAILURES = Counter(
    "medicalkg_chunking_errors_total",
    "Chunking failures grouped by profile and error type",
    labelnames=("profile", "error_type"),
)
MINERU_GATE_TRIGGERED = Counter(
    "mineru_gate_triggered_total",
    "Number of times the MinerU two-phase gate halted processing",
)
POSTPDF_START_TRIGGERED = Counter(
    "postpdf_start_triggered_total",
    "Number of times post-PDF resume was triggered",
)
GATE_EVALUATIONS = Counter(
    "orchestration_gate_evaluations_total",
    "Gate evaluation outcomes grouped by result",
    labelnames=("gate", "pipeline", "result"),
)
GATE_TIMEOUTS = Counter(
    "orchestration_gate_timeouts_total",
    "Number of gate evaluations that timed out",
    labelnames=("gate", "pipeline"),
)
PIPELINE_PHASE_TRANSITIONS = Counter(
    "orchestration_phase_transitions_total",
    "Pipeline phase transition counter",
    labelnames=("pipeline", "from_phase", "to_phase"),
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
JOB_STATUS_COUNTS = Gauge(
    "job_status_counts",
    "Current count of jobs by status",
    labelnames=("status",),
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

RESILIENCE_RETRY_ATTEMPTS = Counter(
    "orchestration_resilience_retry_total",
    "Number of retry attempts triggered by resilience policies",
    labelnames=("policy", "stage"),
)

RESILIENCE_CIRCUIT_STATE = Gauge(
    "orchestration_circuit_breaker_state",
    "Circuit breaker state per resilience policy (0=closed, 1=open, 2=half-open)",
    labelnames=("policy", "stage"),
)

RESILIENCE_RATE_LIMIT_WAIT = Histogram(
    "orchestration_rate_limit_wait_seconds",
    "Time spent waiting for rate limiter tokens per policy",
    labelnames=("policy", "stage"),
    buckets=(0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0),
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


def instrument_application(app: FastAPI, settings: AppSettings) -> None:  # type: ignore[valid-type]
    """Instrument FastAPI application with metrics."""
    # Add middleware for request metrics
    pass


def register_metrics(app: FastAPI, settings: AppSettings) -> None:  # type: ignore[valid-type]
    if FastAPI is None:
        logger.info(
            "metrics.registration.skipped",
            reason="fastapi_unavailable",
        )
        return
    if not settings.observability.metrics.enabled:
        logger.info(
            "metrics.registration.skipped",
            reason="disabled",
        )
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
    async def metrics_endpoint() -> "Response":
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def record_resilience_retry(policy: str, stage: str) -> None:
    """Increment retry counter for the supplied policy and stage."""

    RESILIENCE_RETRY_ATTEMPTS.labels(policy, stage).inc()


def record_resilience_circuit_state(policy: str, stage: str, state: str) -> None:
    """Update gauge with the numeric circuit breaker state."""

    mapping = {"closed": 0.0, "open": 1.0, "half-open": 2.0}
    RESILIENCE_CIRCUIT_STATE.labels(policy, stage).set(mapping.get(state.lower(), -1.0))


def record_resilience_rate_limit_wait(policy: str, stage: str, wait_seconds: float) -> None:
    """Observe rate limit wait duration."""

    RESILIENCE_RATE_LIMIT_WAIT.labels(policy, stage).observe(wait_seconds)


def record_gate_evaluation(gate: str, pipeline: str, result: str) -> None:
    """Record the outcome of a gate evaluation attempt."""

    _increment_with_exemplar(GATE_EVALUATIONS, (gate, pipeline or "unknown", result))


def record_gate_timeout(gate: str, pipeline: str) -> None:
    """Record a gate timeout occurrence."""

    _increment_with_exemplar(GATE_TIMEOUTS, (gate, pipeline or "unknown"))


def record_pipeline_phase_transition(
    pipeline: str, from_phase: str | None, to_phase: str
) -> None:
    """Record a transition between pipeline execution phases."""

    labels = (pipeline or "unknown", from_phase or "unknown", to_phase)
    _increment_with_exemplar(PIPELINE_PHASE_TRANSITIONS, labels)


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


def observe_job_duration(operation: str, duration_seconds: float) -> None:
    JOB_DURATION.labels(operation=operation).observe(max(duration_seconds, 0.0))



def record_chunking_document(profile: str, duration_seconds: float, chunks: int) -> None:
    """Record metrics for a completed chunking operation."""

    labels = (profile or "unknown",)
    _increment_with_exemplar(CHUNKING_DOCUMENTS, labels)
    _observe_with_exemplar(CHUNKING_DURATION, labels, max(duration_seconds, 0.0))
    _observe_with_exemplar(CHUNKS_PER_DOCUMENT, labels, float(max(chunks, 0)))


def record_chunking_failure(profile: str | None, error_type: str) -> None:
    """Increment chunking failure counter for the supplied error type."""

    labels = (profile or "unknown", error_type)
    _increment_with_exemplar(CHUNKING_FAILURES, labels)


def increment_mineru_gate_triggered() -> None:
    """Increment the MinerU gate triggered counter."""

    MINERU_GATE_TRIGGERED.inc()


def increment_postpdf_start_triggered() -> None:
    """Increment the post-PDF start triggered counter."""

    POSTPDF_START_TRIGGERED.inc()




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


def set_chunking_circuit_state(state: int) -> None:
    """Set the chunking circuit breaker state."""
    CHUNKING_CIRCUIT_STATE.set(max(0.0, min(state, 2.0)))


def update_job_status_metrics(counts: dict[str, int]) -> None:
    """Update job status metrics with current counts."""
    for status, count in counts.items():
        JOB_STATUS_COUNTS.labels(status=status).set(count)


def observe_orchestration_duration(operation: str, pipeline: str, duration: float) -> None:
    """Observe orchestration operation duration."""
    ORCHESTRATION_DURATION.labels(operation=operation, pipeline=pipeline).observe(duration)


def observe_orchestration_stage(operation: str, stage: str, duration: float) -> None:
    """Observe orchestration stage duration."""
    ORCHESTRATION_STAGE_DURATION.labels(operation=operation, stage=stage).observe(duration)


def record_orchestration_error(operation: str, error_type: str) -> None:
    """Record orchestration error."""
    ORCHESTRATION_ERRORS.labels(operation=operation, error_type=error_type).inc()


def record_orchestration_operation(operation: str, tenant_id: str, status: str) -> None:
    """Record orchestration operation."""
    ORCHESTRATION_OPERATIONS.labels(operation=operation, status=status).inc()


def record_timeout_breach(operation: str, timeout_seconds: float) -> None:
    """Record timeout breach."""
    TIMEOUT_BREACHES.labels(operation=operation).inc()


def set_orchestration_circuit_state(state: int) -> None:
    """Set the orchestration circuit breaker state."""
    ORCHESTRATION_CIRCUIT_STATE.set(max(0.0, min(state, 2.0)))


def observe_chunking_latency(profile: str, duration: float) -> None:
    """Observe chunking operation latency."""
    CHUNKING_LATENCY.labels(profile=profile).observe(duration)


def record_chunk_size(profile: str, granularity: str, size: int) -> None:
    """Record chunk size distribution."""
    CHUNK_SIZE.labels(profile=profile, granularity=granularity).observe(size)


def observe_query_latency(strategy: str, duration: float) -> None:
    """Observe query latency."""
    QUERY_LATENCY.labels(strategy=strategy).observe(duration)


def observe_query_stage_latency(stage: str, duration: float) -> None:
    """Observe query stage latency."""
    QUERY_STAGE_LATENCY.labels(stage=stage).observe(duration)


def record_query_operation(strategy: str, tenant_id: str, status: str) -> None:
    """Record query operation."""
    QUERY_OPERATIONS.labels(strategy=strategy, status=status).inc()


def record_dead_letter_event(queue: str, reason: str) -> None:
    """Record dead letter event."""
    DEAD_LETTER_EVENTS.labels(queue=queue, reason=reason).inc()


def set_dead_letter_queue_depth(queue: str, depth: int) -> None:
    """Set dead letter queue depth."""
    DEAD_LETTER_QUEUE_DEPTH.labels(queue=queue).set(max(0, depth))


def set_orchestration_queue_depth(queue: str, depth: int) -> None:
    """Set orchestration queue depth."""
    ORCHESTRATION_QUEUE_DEPTH.labels(queue=queue).set(max(0, depth))


def observe_ingestion_stage_latency(stage: str, duration: float) -> None:
    """Observe ingestion stage latency."""
    INGESTION_STAGE_LATENCY.labels(stage=stage).observe(duration)


def record_business_event(event_type: str, tenant_id: str) -> None:
    """Record business event."""
    BUSINESS_EVENTS.labels(event=f"{event_type}:{tenant_id}").inc()


def record_ingestion_document(tenant_id: str, doc_type: str) -> None:
    """Record ingestion document."""
    INGESTION_DOCUMENTS.labels(tenant_id=tenant_id, doc_type=doc_type).inc()


def record_ingestion_error(tenant_id: str, error_type: str) -> None:
    """Record ingestion error."""
    INGESTION_ERRORS.labels(tenant_id=tenant_id, error_type=error_type).inc()


# Additional metric definitions
ORCHESTRATION_DURATION = Histogram(
    "orchestration_duration_seconds",
    "Duration of orchestration operations",
    labelnames=("operation", "pipeline"),
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
)

ORCHESTRATION_STAGE_DURATION = Histogram(
    "orchestration_stage_duration_seconds",
    "Duration of orchestration stages",
    labelnames=("operation", "stage"),
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
)

ORCHESTRATION_ERRORS = Counter(
    "orchestration_errors_total",
    "Total number of orchestration errors",
    labelnames=("operation", "error_type"),
)

ORCHESTRATION_OPERATIONS = Counter(
    "orchestration_operations_total",
    "Total number of orchestration operations",
    labelnames=("operation", "status"),
)

ADAPTER_PLUGIN_INVOCATIONS = Counter(
    "adapter_plugin_invocations_total",
    "Total number of adapter plugin executions",
    labelnames=("adapter", "domain"),
)

ADAPTER_PLUGIN_FAILURES = Counter(
    "adapter_plugin_failures_total",
    "Total number of adapter plugin failures",
    labelnames=("adapter", "domain"),
)

ADAPTER_PIPELINE_STAGE_DURATION = Histogram(
    "adapter_pipeline_stage_duration_seconds",
    "Latency distribution for adapter plugin pipeline stages",
    labelnames=("adapter", "stage"),
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0),
)

TIMEOUT_BREACHES = Counter(
    "timeout_breaches_total",
    "Total number of timeout breaches",
    labelnames=("operation",),
)

ORCHESTRATION_CIRCUIT_STATE = Gauge(
    "orchestration_circuit_state",
    "Orchestration circuit breaker state (0=closed, 1=half-open, 2=open)",
)

QUERY_LATENCY = Histogram(
    "query_latency_seconds",
    "Query latency distribution",
    labelnames=("strategy",),
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
)

QUERY_STAGE_LATENCY = Histogram(
    "query_stage_latency_seconds",
    "Query stage latency distribution",
    labelnames=("stage",),
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
)

QUERY_OPERATIONS = Counter(
    "query_operations_total",
    "Total number of query operations",
    labelnames=("strategy", "status"),
)

DEAD_LETTER_EVENTS = Counter(
    "dead_letter_events_total",
    "Total number of dead letter events",
    labelnames=("queue", "reason"),
)

DEAD_LETTER_QUEUE_DEPTH = Gauge(
    "dead_letter_queue_depth",
    "Current depth of dead letter queues",
    labelnames=("queue",),
)

ORCHESTRATION_QUEUE_DEPTH = Gauge(
    "orchestration_queue_depth",
    "Current depth of orchestration queues",
    labelnames=("queue",),
)

INGESTION_STAGE_LATENCY = Histogram(
    "ingestion_stage_latency_seconds",
    "Ingestion stage latency distribution",
    labelnames=("stage",),
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
)


INGESTION_DOCUMENTS = Counter(
    "ingestion_documents_total",
    "Total number of ingested documents",
    labelnames=("tenant_id", "doc_type"),
)

INGESTION_ERRORS = Counter(
    "ingestion_errors_total",
    "Total number of ingestion errors",
    labelnames=("tenant_id", "error_type"),
)
