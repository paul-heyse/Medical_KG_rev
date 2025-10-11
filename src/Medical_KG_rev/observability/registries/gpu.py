"""GPU-specific metric registry for hardware metrics only."""

from Medical_KG_rev.observability.registries.base import BaseMetricRegistry
from prometheus_client import CollectorRegistry, Gauge


class GPUMetricRegistry(BaseMetricRegistry):
    """Metric registry for GPU hardware metrics only.

    Scope:
        - GPU memory usage, utilization, temperature
        - Device status and health
        - Hardware-specific metrics

    Out of Scope:
        - Service communication metrics (use gRPCMetricRegistry)
        - Pipeline metrics (use PipelineMetricRegistry)
        - Cache metrics (use CacheMetricRegistry)
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Initialize GPU metric registry.

        Args:
            registry: Prometheus collector registry to use (default registry if None)
        """
        super().__init__(domain="gpu", registry=registry)
        self.initialize_collectors()

    def initialize_collectors(self) -> None:
        """Initialize GPU-specific Prometheus collectors."""
        self._collectors["memory_usage"] = Gauge(
            "gpu_memory_usage_mb",
            "GPU memory usage in MB",
            ["device_id", "device_name"],
            registry=self._registry
        )

        self._collectors["utilization"] = Gauge(
            "gpu_utilization_percentage",
            "GPU utilization percentage",
            ["device_id", "device_name"],
            registry=self._registry
        )

        self._collectors["temperature"] = Gauge(
            "gpu_temperature_celsius",
            "GPU temperature in Celsius",
            ["device_id", "device_name"],
            registry=self._registry
        )

        self._collectors["health_status"] = Gauge(
            "gpu_device_health_status",
            "GPU device health (1=healthy, 0=unhealthy)",
            ["device_id", "device_name"],
            registry=self._registry
        )

        self._collectors["power_usage"] = Gauge(
            "gpu_power_usage_watts",
            "GPU power usage in watts",
            ["device_id", "device_name"],
            registry=self._registry
        )

        self._collectors["device_count"] = Gauge(
            "gpu_device_count",
            "Total number of GPU devices",
            [],
            registry=self._registry
        )

        self._collectors["memory_total"] = Gauge(
            "gpu_memory_total_mb",
            "Total GPU memory in MB",
            ["device_id", "device_name"],
            registry=self._registry
        )

        self._collectors["memory_available"] = Gauge(
            "gpu_memory_available_mb",
            "Available GPU memory in MB",
            ["device_id", "device_name"],
            registry=self._registry
        )

    def set_memory_usage(self, device_id: str, device_name: str, usage_mb: float) -> None:
        """Set GPU memory usage.

        Args:
            device_id: GPU device identifier
            device_name: GPU device name
            usage_mb: Memory usage in MB
        """
        self.get_collector("memory_usage").labels(
            device_id=device_id, device_name=device_name
        ).set(usage_mb)

    def set_utilization(self, device_id: str, device_name: str, percent: float) -> None:
        """Set GPU utilization percentage.

        Args:
            device_id: GPU device identifier
            device_name: GPU device name
            percent: Utilization percentage (0-100)
        """
        self.get_collector("utilization").labels(
            device_id=device_id, device_name=device_name
        ).set(percent)

    def set_temperature(self, device_id: str, device_name: str, celsius: float) -> None:
        """Set GPU temperature.

        Args:
            device_id: GPU device identifier
            device_name: GPU device name
            celsius: Temperature in Celsius
        """
        self.get_collector("temperature").labels(
            device_id=device_id, device_name=device_name
        ).set(celsius)

    def set_health_status(self, device_id: str, device_name: str, healthy: bool) -> None:
        """Set GPU device health status.

        Args:
            device_id: GPU device identifier
            device_name: GPU device name
            healthy: True if healthy, False if unhealthy
        """
        self.get_collector("health_status").labels(
            device_id=device_id, device_name=device_name
        ).set(1 if healthy else 0)

    def set_power_usage(self, device_id: str, device_name: str, watts: float) -> None:
        """Set GPU power usage.

        Args:
            device_id: GPU device identifier
            device_name: GPU device name
            watts: Power usage in watts
        """
        self.get_collector("power_usage").labels(
            device_id=device_id, device_name=device_name
        ).set(watts)

    def set_device_count(self, count: int) -> None:
        """Set total number of GPU devices.

        Args:
            count: Number of GPU devices
        """
        self.get_collector("device_count").set(count)

    def set_memory_total(self, device_id: str, device_name: str, total_mb: float) -> None:
        """Set total GPU memory.

        Args:
            device_id: GPU device identifier
            device_name: GPU device name
            total_mb: Total memory in MB
        """
        self.get_collector("memory_total").labels(
            device_id=device_id, device_name=device_name
        ).set(total_mb)

    def set_memory_available(self, device_id: str, device_name: str, available_mb: float) -> None:
        """Set available GPU memory.

        Args:
            device_id: GPU device identifier
            device_name: GPU device name
            available_mb: Available memory in MB
        """
        self.get_collector("memory_available").labels(
            device_id=device_id, device_name=device_name
        ).set(available_mb)

    def record_gpu_service_call(self, service: str, method: str, status: str) -> None:
        """Record GPU service call.

        Args:
            service: Service name
            method: Method name
            status: Call status (success, error, timeout)
        """
        # GPU registry no longer tracks service-call metrics. Surface improper usage.
        raise RuntimeError(
            "GPU registry no longer records service calls. Use gRPCMetricRegistry instead."
        )

    def observe_gpu_service_duration(self, service: str, method: str, duration_seconds: float) -> None:
        """Observe GPU service call duration.

        Args:
            service: Service name
            method: Method name
            duration_seconds: Call duration in seconds
        """
        # GPU registry no longer tracks service-call metrics. Surface improper usage.
        raise RuntimeError(
            "GPU registry no longer records service-call durations. Use gRPCMetricRegistry instead."
        )
