"""Custom Metrics Adapter for Kubernetes HPA.

This module provides a custom metrics adapter that exposes GPU metrics
to Kubernetes Horizontal Pod Autoscaler.
"""

import asyncio
import logging
import time
from typing import Any

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)


class CustomMetricsAdapter:
    """Custom metrics adapter for Kubernetes HPA."""

    def __init__(self, prometheus_url: str = "http://prometheus:9090"):
        """Initialize custom metrics adapter.

        Args:
            prometheus_url: URL of Prometheus server

        """
        self.prometheus_url = prometheus_url
        self.app = web.Application()
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Set up HTTP routes."""
        self.app.router.add_get("/health", self.health_check)
        self.app.router.add_get("/metrics", self.get_metrics)
        self.app.router.add_get(
            "/apis/custom.metrics.k8s.io/v1beta1/namespaces/{namespace}/pods/*/{metric}",
            self.get_pod_metrics,
        )
        self.app.router.add_get(
            "/apis/custom.metrics.k8s.io/v1beta1/namespaces/{namespace}/services/*/{metric}",
            self.get_service_metrics,
        )

    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "healthy"})

    async def get_metrics(self, request: web.Request) -> web.Response:
        """Get available metrics."""
        metrics = [
            "gpu_utilization_percent",
            "gpu_memory_usage_mb",
            "gpu_memory_total_mb",
            "gpu_temperature_celsius",
            "service_requests_total",
            "service_response_time_seconds",
            "cpu_usage_percent",
            "memory_usage_mb",
        ]
        return web.json_response({"metrics": metrics})

    async def get_pod_metrics(self, request: web.Request) -> web.Response:
        """Get pod-level metrics."""
        namespace = request.match_info["namespace"]
        metric = request.match_info["metric"]

        try:
            # Query Prometheus for pod metrics
            query = f'{metric}{{namespace="{namespace}"}}'
            metrics_data = await self._query_prometheus(query)

            # Format response for Kubernetes
            response = {
                "kind": "PodMetricsList",
                "apiVersion": "custom.metrics.k8s.io/v1beta1",
                "metadata": {},
                "items": [],
            }

            for metric_data in metrics_data:
                pod_name = metric_data.get("pod", "unknown")
                value = float(metric_data.get("value", [0, "0"])[1])

                response["items"].append(
                    {
                        "metadata": {"name": pod_name, "namespace": namespace},
                        "timestamp": time.time(),
                        "window": "30s",
                        "value": str(value),
                    }
                )

            return web.json_response(response)

        except Exception as e:
            logger.error(f"Error getting pod metrics: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def get_service_metrics(self, request: web.Request) -> web.Response:
        """Get service-level metrics."""
        namespace = request.match_info["namespace"]
        metric = request.match_info["metric"]

        try:
            # Query Prometheus for service metrics
            query = f'{metric}{{namespace="{namespace}"}}'
            metrics_data = await self._query_prometheus(query)

            # Format response for Kubernetes
            response = {
                "kind": "ServiceMetricsList",
                "apiVersion": "custom.metrics.k8s.io/v1beta1",
                "metadata": {},
                "items": [],
            }

            for metric_data in metrics_data:
                service_name = metric_data.get("service", "unknown")
                value = float(metric_data.get("value", [0, "0"])[1])

                response["items"].append(
                    {
                        "metadata": {"name": service_name, "namespace": namespace},
                        "timestamp": time.time(),
                        "window": "30s",
                        "value": str(value),
                    }
                )

            return web.json_response(response)

        except Exception as e:
            logger.error(f"Error getting service metrics: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _query_prometheus(self, query: str) -> list[dict[str, Any]]:
        """Query Prometheus for metrics.

        Args:
            query: Prometheus query

        Returns:
            List of metric data

        """
        url = f"{self.prometheus_url}/api/v1/query"
        params = {"query": query}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Prometheus query failed: {response.status}")

                data = await response.json()

                if data.get("status") != "success":
                    raise Exception(
                        f"Prometheus query failed: {data.get('error', 'Unknown error')}"
                    )

                return data.get("data", {}).get("result", [])

    async def start(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start the metrics adapter server.

        Args:
            host: Host to bind to
            port: Port to bind to

        """
        runner = web.AppRunner(self.app)
        await runner.setup()

        site = web.TCPSite(runner, host, port)
        await site.start()

        logger.info(f"Custom metrics adapter started on {host}:{port}")

        # Keep running
        try:
            await asyncio.Future()
        except KeyboardInterrupt:
            logger.info("Shutting down custom metrics adapter")
        finally:
            await runner.cleanup()


class GPUUtilizationMetricsAdapter(CustomMetricsAdapter):
    """Specialized adapter for GPU utilization metrics."""

    def __init__(self, prometheus_url: str = "http://prometheus:9090"):
        super().__init__(prometheus_url)

    async def get_gpu_utilization_metrics(self, namespace: str) -> dict[str, float]:
        """Get GPU utilization metrics for a namespace.

        Args:
            namespace: Kubernetes namespace

        Returns:
            Dictionary of GPU utilization metrics

        """
        query = f'gpu_utilization_percent{{namespace="{namespace}"}}'
        metrics_data = await self._query_prometheus(query)

        result: dict[str, float] = {}
        for metric_data in metrics_data:
            service_name = metric_data.get("service_name", "unknown")
            value = float(metric_data.get("value", [0, "0"])[1])
            result[service_name] = value

        return result

    async def get_gpu_memory_metrics(self, namespace: str) -> dict[str, float]:
        """Get GPU memory metrics for a namespace.

        Args:
            namespace: Kubernetes namespace

        Returns:
            Dictionary of GPU memory metrics

        """
        query = f'gpu_memory_usage_mb{{namespace="{namespace}"}}'
        metrics_data = await self._query_prometheus(query)

        result: dict[str, float] = {}
        for metric_data in metrics_data:
            service_name = metric_data.get("service_name", "unknown")
            value = float(metric_data.get("value", [0, "0"])[1])
            result[service_name] = value

        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Custom Metrics Adapter")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--prometheus-url", default="http://prometheus:9090", help="Prometheus URL")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create and start adapter
    adapter = CustomMetricsAdapter(args.prometheus_url)

    try:
        asyncio.run(adapter.start(args.host, args.port))
    except KeyboardInterrupt:
        print("Custom metrics adapter stopped")
