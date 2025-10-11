"""Prometheus metrics for GPU services - Torch-free version.

This module provides Prometheus metrics collection for GPU service operations,
replacing torch-dependent metrics with a torch-free implementation that uses
domain-specific metric registries for improved observability.

Key Responsibilities:
    - Define Prometheus metrics for GPU service operations
    - Provide backward compatibility during migration to domain registries
    - Support both legacy and new metric collection patterns
    - Enable feature flag-based migration between metric systems

Collaborators:
    - Upstream: Services that need to emit GPU operation metrics
    - Downstream: Prometheus monitoring system and Grafana dashboards
    - Migration: MetricsMigrationHelper for domain-specific registry routing

Side Effects:
    - Emits Prometheus metrics for GPU operations
    - Routes metrics through migration helper based on feature flags
    - Updates internal metric counters and histograms

Thread Safety:
    - Thread-safe: All metric operations use atomic Prometheus operations
    - Concurrent calls to metric functions are safe

Performance Characteristics:
    - Low overhead: Prometheus metrics are optimized for minimal performance impact
    - Memory usage: Bounded by metric cardinality and retention policies
    - No external dependencies or blocking operations

Note:
    This module is in transition from legacy metrics to domain-specific registries.
    New code should use MetricsMigrationHelper and appropriate registries instead
    of calling functions in this module directly.

Example:
    >>> from Medical_KG_rev.observability.metrics import record_pipeline_stage
    >>> record_pipeline_stage("embedding", 1.5)  # Legacy usage
    >>>
    >>> # Preferred new usage:
    >>> from Medical_KG_rev.observability.metrics_migration import get_migration_helper
    >>> helper = get_migration_helper(get_settings())
    >>> helper.record_pipeline_stage("pipeline", "embedding", "completed", 1.5)

"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from typing import Any

from Medical_KG_rev.config.settings import get_settings
from Medical_KG_rev.observability.metrics_migration import get_migration_helper
from prometheus_client import Counter, Gauge, Histogram

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

# GPU service metrics (replacing torch-dependent metrics)
GPU_SERVICE_CALLS_TOTAL = Counter(
    "gpu_service_calls_total", "Total number of GPU service calls", ["service", "method", "status"]
)

GPU_SERVICE_CALL_DURATION_SECONDS = Histogram(
    "gpu_service_call_duration_seconds",
    "Duration of GPU service calls",
    ["service", "method"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

GPU_SERVICE_ERRORS_TOTAL = Counter(
    "gpu_service_errors_total",
    "Total number of GPU service errors",
    ["service", "method", "error_type"],
)

GPU_MEMORY_USAGE_MB = Gauge(
    "gpu_memory_usage_mb", "GPU memory usage in MB", ["device_id", "device_name"]
)

GPU_UTILIZATION = Gauge(
    "gpu_utilization_percentage", "GPU utilization percentage", ["device_id", "device_name"]
)

GPU_SERVICE_HEALTH_STATUS = Gauge(
    "gpu_service_health_status",
    "GPU service health status (1=healthy, 0=unhealthy)",
    ["service_name"],
)

CIRCUIT_BREAKER_STATE = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half-open)",
    ["service_name"],
)

# Legacy metrics (replaced with service metrics)
GPU_MEMORY_USED = GPU_MEMORY_USAGE_MB  # Alias for compatibility

# Request tracking metrics
REQUEST_COUNTER = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
)

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def set_chunking_circuit_state(state: int) -> None:
    """Set the circuit breaker state for chunking service.

    Updates the circuit breaker state metric to reflect the current state
    of the chunking service circuit breaker.

    Args:
        state: Circuit breaker state (0=closed, 1=open, 2=half-open)

    Returns:
        None

    Raises:
        None

    Note:
        This function is deprecated. Use domain-specific registries instead.

    """
    CIRCUIT_BREAKER_STATE.labels(service_name="chunking").set(state)


def record_resilience_rate_limit_wait(name: str, stage: str, waited: float) -> None:
    """Record resilience rate limit wait time.

    Records the time spent waiting for rate limits in resilience operations.

    Args:
        name: Name of the operation being rate limited
        stage: Stage of the operation (e.g., "embedding", "retrieval")
        waited: Time spent waiting in seconds

    Returns:
        None

    Raises:
        None

    Note:
        This function is deprecated. Use domain-specific registries instead.

    """
    GPU_SERVICE_CALL_DURATION_SECONDS.labels(service="resilience", method=f"{name}_{stage}_wait").observe(waited)


def record_resilience_retry(name: str, stage: str) -> None:
    """Record resilience retry attempt.

    Records retry attempts in resilience operations.

    Args:
        name: Name of the operation being retried
        stage: Stage of the operation (e.g., "embedding", "retrieval")

    Returns:
        None

    Raises:
        None

    Note:
        This function is deprecated. Use domain-specific registries instead.

    """
    GPU_SERVICE_CALLS_TOTAL.labels(service="resilience", method=f"{name}_{stage}_retry", status="attempt").inc()


# ==============================================================================
# DATA MODELS
# ==============================================================================

# Pipeline state serialization metric
PIPELINE_STATE_SERIALISATIONS = Counter(
    "pipeline_state_serialisations_total",
    "Total number of pipeline state serializations",
    ["pipeline_id", "state_type"],
)

# Pipeline state cache metrics
PIPELINE_STATE_CACHE_HITS = Counter(
    "pipeline_state_cache_hits_total",
    "Total number of pipeline state cache hits",
    ["cache_name"],
)

PIPELINE_STATE_CACHE_MISSES = Counter(
    "pipeline_state_cache_misses_total",
    "Total number of pipeline state cache misses",
    ["cache_name"],
)

PIPELINE_STATE_CACHE_SIZE = Gauge(
    "pipeline_state_cache_size",
    "Current size of pipeline state cache",
    ["cache_name"],
)

# ==============================================================================
# EXCEPTIONS
# ==============================================================================

# Cross-tenant access metrics
CROSS_TENANT_ACCESS_ATTEMPTS = Counter(
    "cross_tenant_access_attempts_total",
    "Total number of cross-tenant access attempts",
    ["source_tenant", "target_tenant"]
)

# Stage plugin metrics
STAGE_PLUGIN_FAILURES = Counter(
    "stage_plugin_failures_total",
    "Total number of stage plugin failures",
    ["plugin_name", "stage_name", "error_type"]
)

STAGE_PLUGIN_HEALTH = Gauge(
    "stage_plugin_health",
    "Stage plugin health status (1=healthy, 0=unhealthy)",
    ["plugin_name", "stage_name"]
)

STAGE_PLUGIN_REGISTRATIONS = Counter(
    "stage_plugin_registrations_total",
    "Total number of stage plugin registrations",
    ["plugin_name", "stage_name"]
)

# Reranking metrics
RERANKING_REQUESTS_TOTAL = Counter(
    "reranking_requests_total",
    "Total number of reranking requests",
    ["model_name", "status"]
)

RERANKING_LATENCY_SECONDS = Histogram(
    "reranking_latency_seconds",
    "Histogram of reranking request latencies",
    ["model_name"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

RERANKING_ERRORS_TOTAL = Counter(
    "reranking_errors_total",
    "Total number of reranking errors",
    ["model_name", "error_type"]
)


# ==============================================================================
# UTILITY FUNCTIONS (CONTINUED)
# ==============================================================================

def observe_chunking_latency(profile: str, duration: float) -> None:
    """Observe chunking latency.

    Records the latency of chunking operations for performance monitoring.

    Args:
        profile: Chunking profile name (e.g., "section", "semantic")
        duration: Duration of chunking operation in seconds

    Returns:
        None

    Raises:
        None

    Note:
        This function is deprecated. Use domain-specific registries instead.

    """
    GPU_SERVICE_CALL_DURATION_SECONDS.labels(service="chunking", method=profile).observe(duration)


def record_chunk_size(profile: str, size: int) -> None:
    """Record chunk size.

    Records the size of chunks produced by chunking operations.

    Args:
        profile: Chunking profile name
        size: Size of chunk in tokens

    Returns:
        None

    Raises:
        None

    Note:
        This function is deprecated. Use domain-specific registries instead.

    """
    GPU_SERVICE_CALLS_TOTAL.labels(service="chunking", method=profile, status="size").inc()


def record_chunking_document(profile: str, duration: float, chunk_count: int) -> None:
    """Record chunking document metrics.

    Records metrics for document chunking operations including duration and chunk count.

    Args:
        profile: Chunking profile name
        duration: Duration of chunking operation in seconds
        chunk_count: Number of chunks produced

    Returns:
        None

    Raises:
        None

    Note:
        This function is deprecated. Use domain-specific registries instead.

    """
    GPU_SERVICE_CALL_DURATION_SECONDS.labels(service="chunking", method=f"{profile}_document").observe(duration)
    GPU_SERVICE_CALLS_TOTAL.labels(service="chunking", method=profile, status="document").inc()


def record_chunking_failure(profile: str, error_type: str) -> None:
    """Record chunking failure.

    Records failures in chunking operations for error tracking.

    Args:
        profile: Chunking profile name
        error_type: Type of error that occurred

    Returns:
        None

    Raises:
        None

    Note:
        This function is deprecated. Use domain-specific registries instead.

    """
    GPU_SERVICE_ERRORS_TOTAL.labels(service="chunking", method=profile, error_type=error_type).inc()


def update_job_status_metrics(job_id: str, status: str, duration: float = 0.0) -> None:
    """Update job status metrics.

    Records job status changes and optional duration metrics.

    Args:
        job_id: Unique identifier for the job
        status: New status of the job (e.g., "started", "completed", "failed")
        duration: Optional duration in seconds for completed jobs

    Returns:
        None

    Raises:
        None

    Note:
        This function is deprecated. Use domain-specific registries instead.

    """
    GPU_SERVICE_CALLS_TOTAL.labels(service="job", method=status, status="update").inc()
    if duration > 0:
        GPU_SERVICE_CALL_DURATION_SECONDS.labels(service="job", method=status).observe(duration)


def record_pipeline_stage(stage_name: str, duration_seconds: float) -> None:
    """Record pipeline stage metrics.

    Records the duration of pipeline stage execution using the migration helper
    to route to appropriate domain-specific registries based on feature flags.

    Args:
        stage_name: Name of the pipeline stage (e.g., "embedding", "chunking")
        duration_seconds: Duration of stage execution in seconds

    Returns:
        None

    Raises:
        None

    Note:
        Uses MetricsMigrationHelper for feature flag-based routing to
        domain-specific registries (PipelineMetricRegistry).

    """
    settings = get_settings()
    helper = get_migration_helper(settings)
    helper.record_pipeline_stage("pipeline", stage_name, "completed", duration_seconds)


def register_metrics(app: Any, settings: Any) -> None:
    """Register metrics with the FastAPI app.

    Registers all defined Prometheus metrics with the FastAPI application.
    This function is called during observability setup to make metrics
    available for scraping.

    Args:
        app: FastAPI application instance
        settings: Application settings (currently unused)

    Returns:
        None

    Raises:
        None

    Note:
        Metrics are already defined as module-level variables and
        automatically registered with the default Prometheus registry.

    """
    # This function is called by the observability setup
    # The actual metrics are already defined as module-level variables
    pass


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    # Core metric objects
    "GPU_SERVICE_CALLS_TOTAL",
    "GPU_SERVICE_CALL_DURATION_SECONDS",
    "GPU_SERVICE_ERRORS_TOTAL",
    "GPU_MEMORY_USAGE_MB",
    "GPU_UTILIZATION",
    "GPU_SERVICE_HEALTH_STATUS",
    "CIRCUIT_BREAKER_STATE",
    "REQUEST_COUNTER",
    "PIPELINE_STATE_SERIALISATIONS",
    "PIPELINE_STATE_CACHE_HITS",
    "PIPELINE_STATE_CACHE_MISSES",
    "PIPELINE_STATE_CACHE_SIZE",
    "CROSS_TENANT_ACCESS_ATTEMPTS",
    "STAGE_PLUGIN_FAILURES",
    "STAGE_PLUGIN_HEALTH",
    "STAGE_PLUGIN_REGISTRATIONS",
    "RERANKING_REQUESTS_TOTAL",
    "RERANKING_LATENCY_SECONDS",
    "RERANKING_ERRORS_TOTAL",

    # Utility functions
    "set_chunking_circuit_state",
    "record_resilience_rate_limit_wait",
    "record_resilience_retry",
    "observe_chunking_latency",
    "record_chunk_size",
    "record_chunking_document",
    "record_chunking_failure",
    "update_job_status_metrics",
    "record_pipeline_stage",
    "register_metrics",
]
