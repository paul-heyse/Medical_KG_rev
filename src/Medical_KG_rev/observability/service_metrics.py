"""Service architecture metrics for monitoring and observability."""

from __future__ import annotations

import time
from typing import Any

from opentelemetry import metrics

from prometheus_client import Counter as PrometheusCounter
from prometheus_client import Gauge as PrometheusGauge
from prometheus_client import Histogram as PrometheusHistogram


class ServiceMetrics:
    """Metrics collector for service architecture monitoring."""

    def __init__(self) -> None:
        """Initialize service metrics."""
        self.meter = metrics.get_meter(__name__)

        # Service call metrics
        self.service_call_counter = self.meter.create_counter(
            name="service_calls_total", description="Total number of service calls", unit="1"
        )

        self.service_call_duration = self.meter.create_histogram(
            name="service_call_duration_seconds", description="Duration of service calls", unit="s"
        )

        self.service_call_errors = self.meter.create_counter(
            name="service_call_errors_total",
            description="Total number of service call errors",
            unit="1",
        )

        # Circuit breaker metrics
        self.circuit_breaker_state = self.meter.create_up_down_counter(
            name="circuit_breaker_state",
            description="Circuit breaker state (0=closed, 1=open, 2=half-open)",
            unit="1",
        )

        self.circuit_breaker_failures = self.meter.create_counter(
            name="circuit_breaker_failures_total",
            description="Total circuit breaker failures",
            unit="1",
        )

        # GPU service metrics
        self.gpu_utilization = self.meter.create_gauge(
            name="gpu_utilization_percent", description="GPU utilization percentage", unit="percent"
        )

        self.gpu_memory_usage = self.meter.create_gauge(
            name="gpu_memory_usage_bytes", description="GPU memory usage in bytes", unit="bytes"
        )

        # Service availability metrics
        self.service_availability = self.meter.create_gauge(
            name="service_availability", description="Service availability (0=down, 1=up)", unit="1"
        )

        # Prometheus metrics for compatibility
        self._prometheus_metrics = self._init_prometheus_metrics()

    def _init_prometheus_metrics(self) -> dict[str, Any]:
        """Initialize Prometheus metrics."""
        return {
            "service_calls_total": PrometheusCounter(
                "service_calls_total",
                "Total number of service calls",
                ["service", "method", "status"],
            ),
            "service_call_duration_seconds": PrometheusHistogram(
                "service_call_duration_seconds",
                "Duration of service calls",
                ["service", "method"],
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            ),
            "service_call_errors_total": PrometheusCounter(
                "service_call_errors_total",
                "Total number of service call errors",
                ["service", "method", "error_type"],
            ),
            "circuit_breaker_state": PrometheusGauge(
                "circuit_breaker_state", "Circuit breaker state", ["service", "state"]
            ),
            "gpu_utilization_percent": PrometheusGauge(
                "gpu_utilization_percent", "GPU utilization percentage", ["service", "device_id"]
            ),
            "gpu_memory_usage_bytes": PrometheusGauge(
                "gpu_memory_usage_bytes", "GPU memory usage in bytes", ["service", "device_id"]
            ),
            "service_availability": PrometheusGauge(
                "service_availability", "Service availability", ["service"]
            ),
        }

    def record_service_call(
        self,
        service: str,
        method: str,
        duration: float,
        status: str = "success",
        error_type: str | None = None,
    ) -> None:
        """Record service call metrics."""
        # OpenTelemetry metrics
        self.service_call_counter.add(
            1, attributes={"service": service, "method": method, "status": status}
        )

        self.service_call_duration.record(
            duration, attributes={"service": service, "method": method}
        )

        if status == "error" and error_type:
            self.service_call_errors.add(
                1, attributes={"service": service, "method": method, "error_type": error_type}
            )

        # Prometheus metrics
        self._prometheus_metrics["service_calls_total"].labels(
            service=service, method=method, status=status
        ).inc()

        self._prometheus_metrics["service_call_duration_seconds"].labels(
            service=service, method=method
        ).observe(duration)

        if status == "error" and error_type:
            self._prometheus_metrics["service_call_errors_total"].labels(
                service=service, method=method, error_type=error_type
            ).inc()

    def record_circuit_breaker_state(self, service: str, state: str, failures: int = 0) -> None:
        """Record circuit breaker metrics."""
        # OpenTelemetry metrics
        state_value = {"closed": 0, "open": 1, "half-open": 2}.get(state, 0)

        self.circuit_breaker_state.add(state_value, attributes={"service": service, "state": state})

        if failures > 0:
            self.circuit_breaker_failures.add(failures, attributes={"service": service})

        # Prometheus metrics
        self._prometheus_metrics["circuit_breaker_state"].labels(service=service, state=state).set(
            state_value
        )

    def record_gpu_metrics(
        self, service: str, device_id: str, utilization: float, memory_usage: int
    ) -> None:
        """Record GPU metrics."""
        # OpenTelemetry metrics
        self.gpu_utilization.set(
            utilization, attributes={"service": service, "device_id": device_id}
        )

        self.gpu_memory_usage.set(
            memory_usage, attributes={"service": service, "device_id": device_id}
        )

        # Prometheus metrics
        self._prometheus_metrics["gpu_utilization_percent"].labels(
            service=service, device_id=device_id
        ).set(utilization)

        self._prometheus_metrics["gpu_memory_usage_bytes"].labels(
            service=service, device_id=device_id
        ).set(memory_usage)

    def record_service_availability(self, service: str, available: bool) -> None:
        """Record service availability metrics."""
        availability_value = 1 if available else 0

        # OpenTelemetry metrics
        self.service_availability.set(availability_value, attributes={"service": service})

        # Prometheus metrics
        self._prometheus_metrics["service_availability"].labels(service=service).set(
            availability_value
        )

    def get_service_health_summary(self) -> dict[str, Any]:
        """Get service health summary."""
        # This would typically query the metrics backend
        # For now, return a placeholder structure
        return {
            "gpu_services": {
                "status": "healthy",
                "response_time_p95": 45.2,
                "error_rate": 0.001,
                "gpu_utilization": 75.3,
            },
            "embedding_services": {
                "status": "healthy",
                "response_time_p95": 123.7,
                "error_rate": 0.0005,
                "gpu_utilization": 68.9,
            },
            "reranking_services": {
                "status": "healthy",
                "response_time_p95": 89.4,
                "error_rate": 0.0008,
                "gpu_utilization": 71.2,
            },
        }


# Global metrics instance
service_metrics = ServiceMetrics()


class ServiceMetricsCollector:
    """Context manager for collecting service call metrics."""

    def __init__(self, service: str, method: str, metrics: ServiceMetrics | None = None) -> None:
        """Initialize metrics collector."""
        self.service = service
        self.method = method
        self.metrics = metrics or service_metrics
        self.start_time: float | None = None

    def __enter__(self) -> ServiceMetricsCollector:
        """Enter context manager."""
        self.start_time = time.time()
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any | None
    ) -> None:
        """Exit context manager."""
        if self.start_time is not None:
            duration = time.time() - self.start_time

            if exc_type is None:
                status = "success"
                error_type = None
            else:
                status = "error"
                error_type = exc_type.__name__

            self.metrics.record_service_call(
                service=self.service,
                method=self.method,
                duration=duration,
                status=status,
                error_type=error_type,
            )


def collect_service_metrics(service: str, method: str) -> ServiceMetricsCollector:
    """Create a service metrics collector."""
    return ServiceMetricsCollector(service, method)
