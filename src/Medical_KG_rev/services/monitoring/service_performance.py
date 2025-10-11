"""Service performance monitoring for torch isolation architecture."""

from __future__ import annotations

from typing import Any
import statistics
import time



class ServicePerformanceMonitor:
    """Monitor performance of GPU services."""

    def __init__(
        self,
        window_size: int = 100,
        alert_threshold_p95: float = 500.0,  # 500ms P95 threshold
        alert_threshold_error_rate: float = 0.05,  # 5% error rate threshold
    ) -> None:
        """Initialize performance monitor."""
        self.window_size = window_size
        self.alert_threshold_p95 = alert_threshold_p95
        self.alert_threshold_error_rate = alert_threshold_error_rate

        # Performance data storage
        self.response_times: dict[str, list[float]] = {}
        self.error_counts: dict[str, int] = {}
        self.total_requests: dict[str, int] = {}
        self.last_alert_time: dict[str, float] = {}

        # Performance metrics
        self.performance_metrics: dict[str, dict[str, Any]] = {}

    def record_request(
        self, service: str, method: str, response_time: float, success: bool = True
    ) -> None:
        """Record a service request."""
        key = f"{service}:{method}"

        # Initialize if needed
        if key not in self.response_times:
            self.response_times[key] = []
            self.error_counts[key] = 0
            self.total_requests[key] = 0

        # Record response time
        self.response_times[key].append(response_time)
        if len(self.response_times[key]) > self.window_size:
            self.response_times[key].pop(0)

        # Record request count
        self.total_requests[key] += 1

        # Record error count
        if not success:
            self.error_counts[key] += 1

        # Update performance metrics
        self._update_performance_metrics(key)

    def _update_performance_metrics(self, key: str) -> None:
        """Update performance metrics for a service."""
        if key not in self.response_times or not self.response_times[key]:
            return

        response_times = self.response_times[key]
        total_requests = self.total_requests[key]
        error_count = self.error_counts[key]

        # Calculate metrics
        metrics = {
            "response_time_p50": statistics.median(response_times),
            "response_time_p95": self._calculate_percentile(response_times, 95),
            "response_time_p99": self._calculate_percentile(response_times, 99),
            "response_time_mean": statistics.mean(response_times),
            "response_time_std": statistics.stdev(response_times) if len(response_times) > 1 else 0,
            "error_rate": error_count / total_requests if total_requests > 0 else 0,
            "total_requests": total_requests,
            "error_count": error_count,
            "last_updated": time.time(),
        }

        self.performance_metrics[key] = metrics

    def _calculate_percentile(self, values: list[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    def get_service_performance(self, service: str, method: str) -> dict[str, Any] | None:
        """Get performance metrics for a specific service method."""
        key = f"{service}:{method}"
        return self.performance_metrics.get(key)

    def get_all_performance_metrics(self) -> dict[str, dict[str, Any]]:
        """Get all performance metrics."""
        return self.performance_metrics.copy()

    def get_performance_alerts(self) -> list[dict[str, Any]]:
        """Get performance alerts for services exceeding thresholds."""
        alerts = []
        current_time = time.time()

        for key, metrics in self.performance_metrics.items():
            service, method = key.split(":", 1)

            # Check P95 latency threshold
            p95_latency = metrics.get("response_time_p95", 0)
            if p95_latency > self.alert_threshold_p95:
                # Throttle alerts to avoid spam
                last_alert = self.last_alert_time.get(f"{key}:p95", 0)
                if current_time - last_alert > 300:  # 5 minutes
                    alerts.append(
                        {
                            "service": service,
                            "method": method,
                            "type": "high_latency",
                            "message": f"P95 latency {p95_latency:.2f}ms exceeds threshold {self.alert_threshold_p95}ms",
                            "value": p95_latency,
                            "threshold": self.alert_threshold_p95,
                            "timestamp": current_time,
                        }
                    )
                    self.last_alert_time[f"{key}:p95"] = current_time

            # Check error rate threshold
            error_rate = metrics.get("error_rate", 0)
            if error_rate > self.alert_threshold_error_rate:
                # Throttle alerts to avoid spam
                last_alert = self.last_alert_time.get(f"{key}:error_rate", 0)
                if current_time - last_alert > 300:  # 5 minutes
                    alerts.append(
                        {
                            "service": service,
                            "method": method,
                            "type": "high_error_rate",
                            "message": f"Error rate {error_rate:.2%} exceeds threshold {self.alert_threshold_error_rate:.2%}",
                            "value": error_rate,
                            "threshold": self.alert_threshold_error_rate,
                            "timestamp": current_time,
                        }
                    )
                    self.last_alert_time[f"{key}:error_rate"] = current_time

        return alerts

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary across all services."""
        if not self.performance_metrics:
            return {"status": "no_data"}

        # Aggregate metrics across all services
        all_p95_latencies = [
            metrics.get("response_time_p95", 0) for metrics in self.performance_metrics.values()
        ]

        all_error_rates = [
            metrics.get("error_rate", 0) for metrics in self.performance_metrics.values()
        ]

        total_requests = sum(
            metrics.get("total_requests", 0) for metrics in self.performance_metrics.values()
        )

        total_errors = sum(
            metrics.get("error_count", 0) for metrics in self.performance_metrics.values()
        )

        return {
            "overall_p95_latency": max(all_p95_latencies) if all_p95_latencies else 0,
            "overall_error_rate": total_errors / total_requests if total_requests > 0 else 0,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "services_monitored": len(self.performance_metrics),
            "status": (
                "healthy"
                if max(all_p95_latencies) < self.alert_threshold_p95
                and max(all_error_rates) < self.alert_threshold_error_rate
                else "degraded"
            ),
        }

    def get_top_slow_services(self, limit: int = 5) -> list[tuple[str, float]]:
        """Get top slowest services by P95 latency."""
        service_latencies = [
            (key, metrics.get("response_time_p95", 0))
            for key, metrics in self.performance_metrics.items()
        ]

        return sorted(service_latencies, key=lambda x: x[1], reverse=True)[:limit]

    def get_top_error_services(self, limit: int = 5) -> list[tuple[str, float]]:
        """Get top services with highest error rates."""
        service_error_rates = [
            (key, metrics.get("error_rate", 0)) for key, metrics in self.performance_metrics.items()
        ]

        return sorted(service_error_rates, key=lambda x: x[1], reverse=True)[:limit]


class GPUPerformanceMonitor:
    """Specialized performance monitor for GPU services."""

    def __init__(self) -> None:
        """Initialize GPU performance monitor."""
        self.gpu_utilization_history: list[float] = []
        self.gpu_memory_history: list[int] = []
        self.gpu_temperature_history: list[float] = []
        self.window_size = 100

        # Performance thresholds
        self.utilization_threshold = 90.0  # 90% utilization
        self.memory_threshold = 0.9  # 90% memory usage
        self.temperature_threshold = 80.0  # 80°C temperature

    def record_gpu_metrics(
        self, utilization: float, memory_usage: int, memory_total: int, temperature: float
    ) -> None:
        """Record GPU performance metrics."""
        # Record utilization
        self.gpu_utilization_history.append(utilization)
        if len(self.gpu_utilization_history) > self.window_size:
            self.gpu_utilization_history.pop(0)

        # Record memory usage percentage
        memory_percentage = (memory_usage / memory_total) * 100 if memory_total > 0 else 0
        self.gpu_memory_history.append(memory_percentage)
        if len(self.gpu_memory_history) > self.window_size:
            self.gpu_memory_history.pop(0)

        # Record temperature
        self.gpu_temperature_history.append(temperature)
        if len(self.gpu_temperature_history) > self.window_size:
            self.gpu_temperature_history.pop(0)

    def get_gpu_performance_summary(self) -> dict[str, Any]:
        """Get GPU performance summary."""
        if not self.gpu_utilization_history:
            return {"status": "no_data"}

        return {
            "utilization": {
                "current": self.gpu_utilization_history[-1] if self.gpu_utilization_history else 0,
                "average": statistics.mean(self.gpu_utilization_history),
                "max": max(self.gpu_utilization_history),
                "min": min(self.gpu_utilization_history),
            },
            "memory": {
                "current": self.gpu_memory_history[-1] if self.gpu_memory_history else 0,
                "average": statistics.mean(self.gpu_memory_history),
                "max": max(self.gpu_memory_history),
                "min": min(self.gpu_memory_history),
            },
            "temperature": {
                "current": self.gpu_temperature_history[-1] if self.gpu_temperature_history else 0,
                "average": statistics.mean(self.gpu_temperature_history),
                "max": max(self.gpu_temperature_history),
                "min": min(self.gpu_temperature_history),
            },
            "status": self._get_gpu_status(),
        }

    def _get_gpu_status(self) -> str:
        """Get GPU status based on performance metrics."""
        if not self.gpu_utilization_history:
            return "unknown"

        current_utilization = self.gpu_utilization_history[-1]
        current_memory = self.gpu_memory_history[-1] if self.gpu_memory_history else 0
        current_temperature = (
            self.gpu_temperature_history[-1] if self.gpu_temperature_history else 0
        )

        if (
            current_utilization > self.utilization_threshold
            or current_memory > self.memory_threshold
            or current_temperature > self.temperature_threshold
        ):
            return "overloaded"
        elif current_utilization < 10:  # Low utilization
            return "idle"
        else:
            return "healthy"

    def get_gpu_alerts(self) -> list[dict[str, Any]]:
        """Get GPU performance alerts."""
        alerts = []
        current_time = time.time()

        if not self.gpu_utilization_history:
            return alerts

        current_utilization = self.gpu_utilization_history[-1]
        current_memory = self.gpu_memory_history[-1] if self.gpu_memory_history else 0
        current_temperature = (
            self.gpu_temperature_history[-1] if self.gpu_temperature_history else 0
        )

        # High utilization alert
        if current_utilization > self.utilization_threshold:
            alerts.append(
                {
                    "type": "high_utilization",
                    "message": f"GPU utilization {current_utilization:.1f}% exceeds threshold {self.utilization_threshold}%",
                    "value": current_utilization,
                    "threshold": self.utilization_threshold,
                    "timestamp": current_time,
                }
            )

        # High memory usage alert
        if current_memory > self.memory_threshold:
            alerts.append(
                {
                    "type": "high_memory",
                    "message": f"GPU memory usage {current_memory:.1f}% exceeds threshold {self.memory_threshold:.1%}",
                    "value": current_memory,
                    "threshold": self.memory_threshold,
                    "timestamp": current_time,
                }
            )

        # High temperature alert
        if current_temperature > self.temperature_threshold:
            alerts.append(
                {
                    "type": "high_temperature",
                    "message": f"GPU temperature {current_temperature:.1f}°C exceeds threshold {self.temperature_threshold}°C",
                    "value": current_temperature,
                    "threshold": self.temperature_threshold,
                    "timestamp": current_time,
                }
            )

        return alerts


class ServicePerformanceAggregator:
    """Aggregate performance information from multiple monitors."""

    def __init__(self) -> None:
        """Initialize performance aggregator."""
        self.monitors: dict[str, Any] = {}

    def add_monitor(self, name: str, monitor: Any) -> None:
        """Add a performance monitor."""
        self.monitors[name] = monitor

    def get_overall_performance(self) -> dict[str, Any]:
        """Get overall system performance."""
        performance_summary = {
            "overall_status": "healthy",
            "services": {},
            "gpu": {},
            "alerts": [],
            "last_updated": time.time(),
        }

        # Aggregate service performance
        for name, monitor in self.monitors.items():
            if hasattr(monitor, "get_all_performance_metrics"):
                performance_summary["services"].update(monitor.get_all_performance_metrics())
            elif hasattr(monitor, "get_gpu_performance_summary"):
                performance_summary["gpu"] = monitor.get_gpu_performance_summary()

        # Collect alerts
        for name, monitor in self.monitors.items():
            if hasattr(monitor, "get_performance_alerts"):
                performance_summary["alerts"].extend(monitor.get_performance_alerts())
            elif hasattr(monitor, "get_gpu_alerts"):
                performance_summary["alerts"].extend(monitor.get_gpu_alerts())

        # Determine overall status
        if performance_summary["alerts"]:
            performance_summary["overall_status"] = "degraded"

        return performance_summary

    def get_performance_recommendations(self) -> list[dict[str, Any]]:
        """Get performance optimization recommendations."""
        recommendations = []

        for name, monitor in self.monitors.items():
            if hasattr(monitor, "get_top_slow_services"):
                slow_services = monitor.get_top_slow_services(limit=3)
                for service, latency in slow_services:
                    if latency > 200:  # 200ms threshold for recommendations
                        recommendations.append(
                            {
                                "type": "optimization",
                                "service": service,
                                "recommendation": f"Consider optimizing {service} - P95 latency is {latency:.2f}ms",
                                "priority": "medium" if latency < 400 else "high",
                            }
                        )

            if hasattr(monitor, "get_gpu_performance_summary"):
                gpu_summary = monitor.get_gpu_performance_summary()
                if gpu_summary.get("status") == "overloaded":
                    recommendations.append(
                        {
                            "type": "scaling",
                            "service": "gpu",
                            "recommendation": "Consider scaling GPU services - GPU is overloaded",
                            "priority": "high",
                        }
                    )

        return recommendations
