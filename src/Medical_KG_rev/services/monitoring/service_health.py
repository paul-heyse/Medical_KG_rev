"""Service health monitoring for torch isolation architecture."""

from __future__ import annotations

from typing import Any
import asyncio
import time

from grpc_health.v1 import health_pb2, health_pb2_grpc
import grpc

from Medical_KG_rev.observability.service_metrics import service_metrics



class ServiceHealthMonitor:
    """Monitor health of GPU services."""

    def __init__(
        self, service_endpoints: dict[str, str], check_interval: int = 30, timeout: int = 10
    ) -> None:
        """Initialize service health monitor."""
        self.service_endpoints = service_endpoints
        self.check_interval = check_interval
        self.timeout = timeout
        self.health_status: dict[str, dict[str, Any]] = {}
        self._running = False
        self._monitor_task: asyncio.Task | None = None

    async def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_all_services()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)

    async def _check_all_services(self) -> None:
        """Check health of all services."""
        tasks = []
        for service_name, endpoint in self.service_endpoints.items():
            task = asyncio.create_task(self._check_service_health(service_name, endpoint))
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_service_health(self, service_name: str, endpoint: str) -> None:
        """Check health of a single service."""
        try:
            start_time = time.time()

            # Create gRPC channel and health stub
            channel = grpc.aio.insecure_channel(endpoint)
            stub = health_pb2_grpc.HealthStub(channel)

            # Check service health
            request = health_pb2.HealthCheckRequest(service="")
            response = await asyncio.wait_for(stub.Check(request), timeout=self.timeout)

            await channel.close()

            response_time = time.time() - start_time
            is_healthy = response.status == health_pb2.HealthCheckResponse.SERVING

            # Update health status
            self.health_status[service_name] = {
                "status": "healthy" if is_healthy else "unhealthy",
                "response_time": response_time,
                "last_check": time.time(),
                "error": None,
            }

            # Record metrics
            service_metrics.record_service_availability(service_name, is_healthy)
            service_metrics.record_service_call(
                service=service_name,
                method="health_check",
                duration=response_time,
                status="success" if is_healthy else "error",
            )

        except TimeoutError:
            self.health_status[service_name] = {
                "status": "timeout",
                "response_time": self.timeout,
                "last_check": time.time(),
                "error": "Health check timeout",
            }

            service_metrics.record_service_availability(service_name, False)
            service_metrics.record_service_call(
                service=service_name,
                method="health_check",
                duration=self.timeout,
                status="error",
                error_type="timeout",
            )

        except Exception as e:
            self.health_status[service_name] = {
                "status": "error",
                "response_time": None,
                "last_check": time.time(),
                "error": str(e),
            }

            service_metrics.record_service_availability(service_name, False)
            service_metrics.record_service_call(
                service=service_name,
                method="health_check",
                duration=0,
                status="error",
                error_type=type(e).__name__,
            )

    def get_service_health(self, service_name: str) -> dict[str, Any] | None:
        """Get health status of a specific service."""
        return self.health_status.get(service_name)

    def get_all_services_health(self) -> dict[str, dict[str, Any]]:
        """Get health status of all services."""
        return self.health_status.copy()

    def is_service_healthy(self, service_name: str) -> bool:
        """Check if a service is healthy."""
        health = self.health_status.get(service_name)
        return health is not None and health["status"] == "healthy"

    def get_unhealthy_services(self) -> list[str]:
        """Get list of unhealthy services."""
        return [
            service_name
            for service_name, health in self.health_status.items()
            if health["status"] != "healthy"
        ]

    def get_service_response_times(self) -> dict[str, float]:
        """Get response times for all services."""
        return {
            service_name: health["response_time"]
            for service_name, health in self.health_status.items()
            if health["response_time"] is not None
        }

    def get_service_error_rates(self) -> dict[str, float]:
        """Get error rates for all services."""
        error_rates = {}
        for service_name, health in self.health_status.items():
            if health["status"] != "healthy":
                error_rates[service_name] = 1.0
            else:
                error_rates[service_name] = 0.0
        return error_rates


class GPUHealthMonitor:
    """Specialized health monitor for GPU services."""

    def __init__(self, gpu_service_endpoint: str, check_interval: int = 30) -> None:
        """Initialize GPU health monitor."""
        self.gpu_service_endpoint = gpu_service_endpoint
        self.check_interval = check_interval
        self.gpu_status: dict[str, Any] = {}
        self._running = False
        self._monitor_task: asyncio.Task | None = None

    async def start_monitoring(self) -> None:
        """Start GPU health monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self) -> None:
        """Stop GPU health monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self) -> None:
        """Main GPU monitoring loop."""
        while self._running:
            try:
                await self._check_gpu_health()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"GPU health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)

    async def _check_gpu_health(self) -> None:
        """Check GPU health and utilization."""
        try:
            # This would typically call the GPU service to get status
            # For now, we'll simulate GPU health checks
            gpu_utilization = 75.3  # Simulated
            gpu_memory_usage = 8 * 1024 * 1024 * 1024  # 8GB simulated

            self.gpu_status = {
                "available": True,
                "utilization": gpu_utilization,
                "memory_usage": gpu_memory_usage,
                "memory_total": 16 * 1024 * 1024 * 1024,  # 16GB simulated
                "temperature": 65.2,  # Simulated
                "last_check": time.time(),
            }

            # Record GPU metrics
            service_metrics.record_gpu_metrics(
                service="gpu_services",
                device_id="0",
                utilization=gpu_utilization,
                memory_usage=gpu_memory_usage,
            )

        except Exception as e:
            self.gpu_status = {"available": False, "error": str(e), "last_check": time.time()}

    def get_gpu_status(self) -> dict[str, Any]:
        """Get current GPU status."""
        return self.gpu_status.copy()

    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return self.gpu_status.get("available", False)

    def get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        return self.gpu_status.get("utilization", 0.0)

    def get_gpu_memory_usage(self) -> int:
        """Get GPU memory usage in bytes."""
        return self.gpu_status.get("memory_usage", 0)


class ServiceHealthAggregator:
    """Aggregate health information from multiple monitors."""

    def __init__(self) -> None:
        """Initialize health aggregator."""
        self.monitors: dict[str, Any] = {}

    def add_monitor(self, name: str, monitor: Any) -> None:
        """Add a health monitor."""
        self.monitors[name] = monitor

    def get_overall_health(self) -> dict[str, Any]:
        """Get overall system health."""
        health_summary = {
            "overall_status": "healthy",
            "services": {},
            "gpu_status": "available",
            "last_updated": time.time(),
        }

        # Aggregate service health
        for name, monitor in self.monitors.items():
            if hasattr(monitor, "get_all_services_health"):
                health_summary["services"].update(monitor.get_all_services_health())
            elif hasattr(monitor, "get_gpu_status"):
                health_summary["gpu_status"] = monitor.get_gpu_status()

        # Determine overall status
        unhealthy_services = [
            name
            for name, health in health_summary["services"].items()
            if health.get("status") != "healthy"
        ]

        if unhealthy_services:
            health_summary["overall_status"] = "degraded"
            health_summary["unhealthy_services"] = unhealthy_services

        return health_summary

    def get_health_alerts(self) -> list[dict[str, Any]]:
        """Get health alerts for unhealthy services."""
        alerts = []

        for name, monitor in self.monitors.items():
            if hasattr(monitor, "get_unhealthy_services"):
                unhealthy = monitor.get_unhealthy_services()
                for service in unhealthy:
                    alerts.append(
                        {
                            "service": service,
                            "type": "service_unhealthy",
                            "message": f"Service {service} is unhealthy",
                            "timestamp": time.time(),
                        }
                    )

            if hasattr(monitor, "is_gpu_available") and not monitor.is_gpu_available():
                alerts.append(
                    {
                        "service": "gpu",
                        "type": "gpu_unavailable",
                        "message": "GPU is not available",
                        "timestamp": time.time(),
                    }
                )

        return alerts
