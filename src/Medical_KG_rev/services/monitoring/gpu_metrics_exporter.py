"""GPU Metrics Exporter for Kubernetes HPA.

This module provides custom metrics for GPU services to enable
Kubernetes Horizontal Pod Autoscaler scaling based on GPU utilization.
"""

import asyncio
import logging
from typing import Any

import psutil
import pynvml

from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)


class GPUMetricsExporter:
    """Exports GPU metrics for Kubernetes HPA scaling."""

    def __init__(self, port: int = 8000):
        """Initialize GPU metrics exporter.

        Args:
            port: Port to expose metrics on

        """
        self.port = port
        self._initialize_nvidia_ml()
        self._setup_metrics()

    def _initialize_nvidia_ml(self) -> None:
        """Initialize NVIDIA ML library."""
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"Initialized NVIDIA ML with {self.device_count} devices")
        except Exception as e:
            logger.error(f"Failed to initialize NVIDIA ML: {e}")
            self.device_count = 0

    def _setup_metrics(self) -> None:
        """Set up Prometheus metrics."""
        # GPU Utilization Metrics
        self.gpu_utilization = Gauge(
            "gpu_utilization_percent", "GPU utilization percentage", ["device_id", "service_name"]
        )

        self.gpu_memory_usage = Gauge(
            "gpu_memory_usage_mb", "GPU memory usage in MB", ["device_id", "service_name"]
        )

        self.gpu_memory_total = Gauge(
            "gpu_memory_total_mb", "Total GPU memory in MB", ["device_id", "service_name"]
        )

        self.gpu_temperature = Gauge(
            "gpu_temperature_celsius", "GPU temperature in Celsius", ["device_id", "service_name"]
        )

        # Service-specific metrics
        self.service_request_rate = Counter(
            "service_requests_total", "Total service requests", ["service_name", "method", "status"]
        )

        self.service_response_time = Histogram(
            "service_response_time_seconds", "Service response time", ["service_name", "method"]
        )

        # System metrics
        self.cpu_usage = Gauge("cpu_usage_percent", "CPU usage percentage", ["service_name"])

        self.memory_usage = Gauge("memory_usage_mb", "Memory usage in MB", ["service_name"])

    def get_gpu_metrics(self, service_name: str) -> dict[str, Any]:
        """Get GPU metrics for a specific service.

        Args:
            service_name: Name of the service

        Returns:
            Dictionary of GPU metrics

        """
        metrics: dict[str, Any] = {}

        if self.device_count == 0:
            return metrics

        try:
            for device_id in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

                # GPU Utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                self.gpu_utilization.labels(
                    device_id=str(device_id), service_name=service_name
                ).set(gpu_util)

                # GPU Memory
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used_mb = memory_info.used // (1024 * 1024)
                memory_total_mb = memory_info.total // (1024 * 1024)

                self.gpu_memory_usage.labels(
                    device_id=str(device_id), service_name=service_name
                ).set(memory_used_mb)

                self.gpu_memory_total.labels(
                    device_id=str(device_id), service_name=service_name
                ).set(memory_total_mb)

                # GPU Temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    self.gpu_temperature.labels(
                        device_id=str(device_id), service_name=service_name
                    ).set(temperature)
                except Exception:
                    # Temperature not available on all devices
                    pass

                metrics[f"gpu_{device_id}_utilization"] = gpu_util
                metrics[f"gpu_{device_id}_memory_used_mb"] = memory_used_mb
                metrics[f"gpu_{device_id}_memory_total_mb"] = memory_total_mb

        except Exception as e:
            logger.error(f"Error getting GPU metrics: {e}")

        return metrics

    def get_system_metrics(self, service_name: str) -> dict[str, Any]:
        """Get system metrics for a specific service.

        Args:
            service_name: Name of the service

        Returns:
            Dictionary of system metrics

        """
        metrics: dict[str, Any] = {}

        try:
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.labels(service_name=service_name).set(cpu_percent)
            metrics["cpu_usage_percent"] = cpu_percent

            # Memory Usage
            memory = psutil.virtual_memory()
            memory_used_mb = memory.used // (1024 * 1024)
            self.memory_usage.labels(service_name=service_name).set(memory_used_mb)
            metrics["memory_usage_mb"] = memory_used_mb

        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")

        return metrics

    def record_service_request(
        self, service_name: str, method: str, status: str, response_time: float
    ) -> None:
        """Record service request metrics.

        Args:
            service_name: Name of the service
            method: gRPC method name
            status: Response status
            response_time: Response time in seconds

        """
        self.service_request_rate.labels(
            service_name=service_name, method=method, status=status
        ).inc()

        self.service_response_time.labels(service_name=service_name, method=method).observe(
            response_time
        )

    async def start_metrics_server(self) -> None:
        """Start the metrics server."""
        start_http_server(self.port)
        logger.info(f"Started metrics server on port {self.port}")

    async def collect_metrics_loop(self, service_name: str, interval: int = 30) -> None:
        """Continuously collect and update metrics.

        Args:
            service_name: Name of the service
            interval: Collection interval in seconds

        """
        while True:
            try:
                # Collect GPU metrics
                gpu_metrics = self.get_gpu_metrics(service_name)

                # Collect system metrics
                system_metrics = self.get_system_metrics(service_name)

                # Log metrics for debugging
                logger.debug(f"GPU metrics: {gpu_metrics}")
                logger.debug(f"System metrics: {system_metrics}")

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(interval)


class ServiceMetricsCollector:
    """Collects metrics from multiple services."""

    def __init__(self, services: list[str], port: int = 8000):
        """Initialize metrics collector.

        Args:
            services: List of service names
            port: Port to expose metrics on

        """
        self.services = services
        self.exporter = GPUMetricsExporter(port)
        self._running = False

    async def start(self) -> None:
        """Start the metrics collector."""
        if self._running:
            return

        self._running = True

        # Start metrics server
        await self.exporter.start_metrics_server()

        # Start metrics collection for each service
        tasks = []
        for service in self.services:
            task = asyncio.create_task(self.exporter.collect_metrics_loop(service))
            tasks.append(task)

        # Wait for all tasks
        await asyncio.gather(*tasks)

    def stop(self) -> None:
        """Stop the metrics collector."""
        self._running = False


# Service-specific metric collectors
class EmbeddingServiceMetricsCollector(ServiceMetricsCollector):
    """Metrics collector for embedding service."""

    def __init__(self, port: int = 8001):
        super().__init__(["embedding-service"], port)


class RerankingServiceMetricsCollector(ServiceMetricsCollector):
    """Metrics collector for reranking service."""

    def __init__(self, port: int = 8002):
        super().__init__(["reranking-service"], port)


class DoclingVLMServiceMetricsCollector(ServiceMetricsCollector):
    """Metrics collector for Docling VLM service."""

    def __init__(self, port: int = 8003):
        super().__init__(["docling-vlm-service"], port)


class GPUManagementServiceMetricsCollector(ServiceMetricsCollector):
    """Metrics collector for GPU management service."""

    def __init__(self, port: int = 8004):
        super().__init__(["gpu-management-service"], port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU Metrics Exporter")
    parser.add_argument("--service", required=True, help="Service name")
    parser.add_argument("--port", type=int, default=8000, help="Metrics port")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create and start metrics collector
    collector = ServiceMetricsCollector([args.service], args.port)

    try:
        asyncio.run(collector.start())
    except KeyboardInterrupt:
        collector.stop()
        print("Metrics collector stopped")
