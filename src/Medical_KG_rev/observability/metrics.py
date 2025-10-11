"""Prometheus metrics for GPU services - Torch-free version."""

from prometheus_client import Counter, Gauge, Histogram

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

# Circuit breaker state management functions
def set_chunking_circuit_state(state: int) -> None:
    """Set the circuit breaker state for chunking service."""
    CIRCUIT_BREAKER_STATE.labels(service_name="chunking").set(state)

# Job duration and business event tracking
def observe_job_duration(operation: str, duration_seconds: float) -> None:
    """Observe job duration for a specific operation."""
    # Use a generic histogram for job durations
    GPU_SERVICE_CALL_DURATION_SECONDS.labels(service="gateway", method=operation).observe(duration_seconds)

def record_business_event(event: str, tenant_id: str) -> None:
    """Record a business event."""
    # Use the GPU service calls counter for business events
    GPU_SERVICE_CALLS_TOTAL.labels(service="gateway", method=event, status="success").inc()

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

# Resilience circuit state function
def record_resilience_circuit_state(state: int) -> None:
    """Record resilience circuit breaker state."""
    CIRCUIT_BREAKER_STATE.labels(service_name="resilience").set(state)

# Missing metrics functions
def record_resilience_rate_limit_wait(name: str, stage: str, waited: float) -> None:
    """Record resilience rate limit wait time."""
    GPU_SERVICE_CALL_DURATION_SECONDS.labels(service="resilience", method=f"{name}_{stage}_wait").observe(waited)

def record_resilience_retry(name: str, stage: str) -> None:
    """Record resilience retry attempt."""
    GPU_SERVICE_CALLS_TOTAL.labels(service="resilience", method=f"{name}_{stage}_retry", status="attempt").inc()

# Chunking metrics
def observe_chunking_latency(profile: str, duration: float) -> None:
    """Observe chunking latency."""
    GPU_SERVICE_CALL_DURATION_SECONDS.labels(service="chunking", method=profile).observe(duration)

def record_chunk_size(profile: str, size: int) -> None:
    """Record chunk size."""
    GPU_SERVICE_CALLS_TOTAL.labels(service="chunking", method=profile, status="size").inc()

def record_chunking_document(profile: str, duration: float, chunk_count: int) -> None:
    """Record chunking document metrics."""
    GPU_SERVICE_CALL_DURATION_SECONDS.labels(service="chunking", method=f"{profile}_document").observe(duration)
    GPU_SERVICE_CALLS_TOTAL.labels(service="chunking", method=profile, status="document").inc()

def record_chunking_failure(profile: str, error_type: str) -> None:
    """Record chunking failure."""
    GPU_SERVICE_ERRORS_TOTAL.labels(service="chunking", method=profile, error_type=error_type).inc()

# Cross-tenant access metrics
CROSS_TENANT_ACCESS_ATTEMPTS = Counter(
    "cross_tenant_access_attempts_total",
    "Total number of cross-tenant access attempts",
    ["source_tenant", "target_tenant"]
)

# Job status metrics
def update_job_status_metrics(job_id: str, status: str, duration: float = 0.0) -> None:
    """Update job status metrics."""
    GPU_SERVICE_CALLS_TOTAL.labels(service="job", method=status, status="update").inc()
    if duration > 0:
        GPU_SERVICE_CALL_DURATION_SECONDS.labels(service="job", method=status).observe(duration)

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

# Missing metrics functions
def record_cache_hit_rate(cache_name: str, hit_rate: float) -> None:
    """Record cache hit rate."""
    GPU_SERVICE_CALLS_TOTAL.labels(service="cache", method=cache_name, status="hit_rate").inc()

def record_cache_miss_rate(cache_name: str, miss_rate: float) -> None:
    """Record cache miss rate."""
    GPU_SERVICE_CALLS_TOTAL.labels(service="cache", method=cache_name, status="miss_rate").inc()

def record_gpu_memory_alert(device_id: str, device_name: str, memory_usage_mb: float) -> None:
    """Record GPU memory alert."""
    GPU_MEMORY_USAGE_MB.labels(device_id=device_id, device_name=device_name).set(memory_usage_mb)

def record_latency_alert(service: str, method: str, latency_seconds: float) -> None:
    """Record latency alert."""
    GPU_SERVICE_CALL_DURATION_SECONDS.labels(service=service, method=method).observe(latency_seconds)

def record_reranking_operation(model_name: str, doc_count: int, duration_seconds: float) -> None:
    """Record reranking operation metrics."""
    GPU_SERVICE_CALLS_TOTAL.labels(service="reranking", method=model_name, status="success").inc()
    GPU_SERVICE_CALL_DURATION_SECONDS.labels(service="reranking", method=model_name).observe(duration_seconds)

def record_reranking_error(error_type: str) -> None:
    """Record reranking error."""
    GPU_SERVICE_ERRORS_TOTAL.labels(service="reranking", method="error", error_type=error_type).inc()

def record_pipeline_stage(stage_name: str, duration_seconds: float) -> None:
    """Record pipeline stage metrics."""
    GPU_SERVICE_CALLS_TOTAL.labels(service="pipeline", method=stage_name, status="completed").inc()
    GPU_SERVICE_CALL_DURATION_SECONDS.labels(service="pipeline", method=stage_name).observe(duration_seconds)
