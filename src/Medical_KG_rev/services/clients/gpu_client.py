"""gRPC client for GPU service.

Handles communication with GPU gRPC service for GPU management.
"""

import logging
import time
from typing import Any

import grpc
from grpc import aio

from ..clients.circuit_breaker import CircuitBreaker, CircuitBreakerState
from ..clients.errors import ServiceError, ServiceTimeoutError, ServiceUnavailableError
from ..registry import ServiceRegistry
from ..security.mtls import create_mtls_channel, mTLSManager

# Import generated gRPC stubs (will be generated from proto)
try:
    from ...proto.gpu_service_pb2 import (
        AllocateGPURequest,
        AllocateGPUResponse,
        GPUAllocation,
        GPUDevice,
        GPUStatus,
        HealthRequest,
        HealthResponse,
        ListDevicesRequest,
        ListDevicesResponse,
        ReleaseGPURequest,
        ReleaseGPUResponse,
        StatsRequest,
        StatsResponse,
        StatusRequest,
        StatusResponse,
    )
    from ...proto.gpu_service_pb2_grpc import GPUServiceStub
except ImportError:
    # Fallback for development - will be replaced by generated stubs
    StatusRequest = None
    StatusResponse = None
    ListDevicesRequest = None
    ListDevicesResponse = None
    AllocateGPURequest = None
    AllocateGPUResponse = None
    ReleaseGPURequest = None
    ReleaseGPUResponse = None
    GPUDevice = None
    GPUAllocation = None
    GPUStatus = None
    HealthRequest = None
    HealthResponse = None
    StatsRequest = None
    StatsResponse = None
    GPUServiceStub = None

logger = logging.getLogger(__name__)


class GPUClient:
    """gRPC client for GPU service.

    Handles:
    - gRPC communication with GPU service
    - Circuit breaker patterns for service resilience
    - Error handling and retry logic
    - Performance monitoring and metrics
    """

    def __init__(
        self,
        service_endpoint: str,
        service_registry: ServiceRegistry | None = None,
        circuit_breaker_config: dict[str, Any] | None = None,
        mtls_manager: mTLSManager | None = None,
        service_name: str = "gpu-management",
    ):
        """Initialize the GPU client.

        Args:
            service_endpoint: gRPC endpoint for the GPU service
            service_registry: Optional service registry for discovery
            circuit_breaker_config: Circuit breaker configuration
            mtls_manager: Optional mTLS manager for secure communication
            service_name: Name of the service for mTLS

        """
        self.service_endpoint = service_endpoint
        self.service_registry = service_registry
        self.mtls_manager = mtls_manager
        self.service_name = service_name
        self.channel: aio.Channel | None = None
        self.stub: GPUServiceStub | None = None

        # Circuit breaker configuration
        circuit_config = circuit_breaker_config or {}
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_config.get("failure_threshold", 5),
            recovery_timeout=circuit_config.get("recovery_timeout", 60),
            expected_exception=ServiceError,
        )

        # Performance tracking
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "circuit_breaker_state": CircuitBreakerState.CLOSED,
        }

    async def initialize(self) -> None:
        """Initialize the gRPC client."""
        try:
            # Create gRPC channel (secure if mTLS enabled)
            if self.mtls_manager:
                self.channel = create_mtls_channel(
                    self.service_endpoint, self.mtls_manager, self.service_name
                )
                logger.info(f"GPU client initialized with mTLS for {self.service_endpoint}")
            else:
                self.channel = aio.insecure_channel(self.service_endpoint)
                logger.info(f"GPU client initialized without mTLS for {self.service_endpoint}")

            # Create service stub
            if GPUServiceStub:
                self.stub = GPUServiceStub(self.channel)
            else:
                logger.warning("gRPC stubs not available - using mock implementation")

            # Test connection
            await self._test_connection()
            logger.info(f"GPU client initialized for {self.service_endpoint}")

        except Exception as e:
            logger.error(f"Failed to initialize GPU client: {e}")
            raise ServiceError(f"Client initialization failed: {e}")

    async def close(self) -> None:
        """Close the gRPC client."""
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None

    async def _test_connection(self) -> None:
        """Test connection to GPU service."""
        try:
            if not self.stub:
                raise ServiceError("Service stub not available")

            # Test with health check
            request = HealthRequest(service_name="gpu_service")
            response = await self.stub.GetHealth(request)

            if response.status != "healthy":
                raise ServiceError(f"Service not healthy: {response.message}")

        except Exception as e:
            logger.error(f"GPU service connection test failed: {e}")
            raise ServiceError(f"Connection test failed: {e}")

    async def get_status(self, include_detailed_info: bool = True) -> dict[str, Any]:
        """Get GPU status and availability.

        Args:
            include_detailed_info: Whether to include detailed device information

        Returns:
            GPU status information

        """
        start_time = time.time()

        try:
            if not self.stub:
                raise ServiceError("Service stub not available")

            request = StatusRequest(include_detailed_info=include_detailed_info)
            response = await self.stub.GetStatus(request)

            # Convert response to dictionary
            result = {
                "gpu_available": response.gpu_available,
                "total_devices": response.total_devices,
                "available_devices": response.available_devices,
                "overall_status": (
                    response.overall_status.name if response.overall_status else "UNKNOWN"
                ),
                "devices": [],
            }

            # Convert devices
            for device in response.devices:
                device_info = {
                    "device_id": device.device_id,
                    "name": device.name,
                    "driver_version": device.driver_version,
                    "total_memory_mb": device.total_memory_mb,
                    "available_memory_mb": device.available_memory_mb,
                    "used_memory_mb": device.used_memory_mb,
                    "utilization_percent": device.utilization_percent,
                    "temperature_celsius": device.temperature_celsius,
                    "status": device.status.name if device.status else "UNKNOWN",
                    "active_allocations": [],
                }

                # Convert allocations
                for allocation in device.active_allocations:
                    allocation_info = {
                        "allocation_id": allocation.allocation_id,
                        "device_id": allocation.device_id,
                        "allocated_memory_mb": allocation.allocated_memory_mb,
                        "allocated_at": (
                            allocation.allocated_at.ToDatetime()
                            if allocation.allocated_at
                            else None
                        ),
                        "expires_at": (
                            allocation.expires_at.ToDatetime() if allocation.expires_at else None
                        ),
                        "metadata": dict(allocation.metadata),
                    }
                    device_info["active_allocations"].append(allocation_info)

                result["devices"].append(device_info)

            # Update statistics
            self._update_stats(True, start_time)

            return result

        except Exception as e:
            logger.error(f"Error getting GPU status: {e}")
            self._update_stats(False, start_time)
            raise self._handle_gpu_error(e)

    async def list_devices(self, include_usage_stats: bool = True) -> list[dict[str, Any]]:
        """List available GPU devices.

        Args:
            include_usage_stats: Whether to include usage statistics

        Returns:
            List of GPU device information

        """
        start_time = time.time()

        try:
            if not self.stub:
                raise ServiceError("Service stub not available")

            request = ListDevicesRequest(include_usage_stats=include_usage_stats)
            response = await self.stub.ListDevices(request)

            # Convert devices to list of dictionaries
            devices = []
            for device in response.devices:
                device_info = {
                    "device_id": device.device_id,
                    "name": device.name,
                    "driver_version": device.driver_version,
                    "total_memory_mb": device.total_memory_mb,
                    "available_memory_mb": device.available_memory_mb,
                    "used_memory_mb": device.used_memory_mb,
                    "utilization_percent": device.utilization_percent,
                    "temperature_celsius": device.temperature_celsius,
                    "status": device.status.name if device.status else "UNKNOWN",
                }
                devices.append(device_info)

            # Update statistics
            self._update_stats(True, start_time)

            return devices

        except Exception as e:
            logger.error(f"Error listing GPU devices: {e}")
            self._update_stats(False, start_time)
            raise self._handle_gpu_error(e)

    async def allocate_gpu(
        self,
        allocation_id: str,
        requested_memory_mb: int,
        preferred_device_id: str | None = None,
        timeout_seconds: int = 300,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Allocate GPU for processing.

        Args:
            allocation_id: Unique allocation identifier
            requested_memory_mb: Requested memory in MB
            preferred_device_id: Preferred GPU device ID
            timeout_seconds: Allocation timeout
            metadata: Additional metadata

        Returns:
            GPU allocation information

        """
        start_time = time.time()

        try:
            if not self.stub:
                raise ServiceError("Service stub not available")

            request = AllocateGPURequest(
                allocation_id=allocation_id,
                requested_memory_mb=requested_memory_mb,
                preferred_device_id=preferred_device_id or "",
                timeout_seconds=timeout_seconds,
                metadata=metadata or {},
            )

            response = await self.stub.AllocateGPU(request)

            if not response.success:
                raise ServiceError(f"GPU allocation failed: {response.error_message}")

            # Convert response to dictionary
            result = {
                "success": response.success,
                "device_id": response.device_id,
                "allocated_memory_mb": response.allocated_memory_mb,
                "allocation_id": response.allocation_id,
                "allocated_at": (
                    response.allocated_at.ToDatetime() if response.allocated_at else None
                ),
                "expires_at": response.expires_at.ToDatetime() if response.expires_at else None,
            }

            # Update statistics
            self._update_stats(True, start_time)

            return result

        except Exception as e:
            logger.error(f"Error allocating GPU: {e}")
            self._update_stats(False, start_time)
            raise self._handle_gpu_error(e)

    async def release_gpu(self, allocation_id: str, device_id: str) -> dict[str, Any]:
        """Release GPU allocation.

        Args:
            allocation_id: Allocation identifier
            device_id: GPU device ID

        Returns:
            Release result information

        """
        start_time = time.time()

        try:
            if not self.stub:
                raise ServiceError("Service stub not available")

            request = ReleaseGPURequest(
                allocation_id=allocation_id,
                device_id=device_id,
            )

            response = await self.stub.ReleaseGPU(request)

            if not response.success:
                raise ServiceError(f"GPU release failed: {response.message}")

            # Convert response to dictionary
            result = {
                "success": response.success,
                "message": response.message,
                "released_at": response.released_at.ToDatetime() if response.released_at else None,
            }

            # Update statistics
            self._update_stats(True, start_time)

            return result

        except Exception as e:
            logger.error(f"Error releasing GPU: {e}")
            self._update_stats(False, start_time)
            raise self._handle_gpu_error(e)

    def _handle_gpu_error(self, error: Exception) -> Exception:
        """Handle GPU errors and convert to appropriate exception types."""
        if isinstance(error, grpc.RpcError):
            if error.code() == grpc.StatusCode.UNAVAILABLE:
                return ServiceUnavailableError(f"GPU service unavailable: {error.details()}")
            elif error.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                return ServiceTimeoutError(f"GPU service timeout: {error.details()}")
            else:
                return ServiceError(f"gRPC error: {error.details()}")
        else:
            return ServiceError(f"GPU error: {error!s}")

    def _update_stats(self, success: bool, start_time: float) -> None:
        """Update processing statistics."""
        self._stats["total_requests"] += 1
        processing_time = time.time() - start_time
        self._stats["total_processing_time"] += processing_time

        if success:
            self._stats["successful_requests"] += 1
        else:
            self._stats["failed_requests"] += 1

        # Calculate average processing time
        if self._stats["total_requests"] > 0:
            self._stats["average_processing_time"] = (
                self._stats["total_processing_time"] / self._stats["total_requests"]
            )

        # Update circuit breaker state
        self._stats["circuit_breaker_state"] = self.circuit_breaker.state

    async def health_check(self) -> dict[str, Any]:
        """Check GPU service health.

        Returns:
            Health status information

        """
        try:
            if not self.stub:
                return {"status": "unhealthy", "error": "Service stub not available"}

            request = HealthRequest(service_name="gpu_service")
            response = await self.stub.GetHealth(request)

            return {
                "status": response.status,
                "message": response.message,
                "timestamp": response.timestamp.ToDatetime() if response.timestamp else None,
                "service_info": {
                    "version": response.service_info.version,
                    "capabilities": list(response.service_info.capabilities),
                },
                "resource_usage": {
                    "cpu_usage_percent": response.resource_usage.cpu_usage_percent,
                    "memory_usage_mb": response.resource_usage.memory_usage_mb,
                    "gpu_usage_percent": response.resource_usage.gpu_usage_percent,
                    "gpu_memory_usage_mb": response.resource_usage.gpu_memory_usage_mb,
                    "active_allocations": response.resource_usage.active_allocations,
                },
                "circuit_breaker_state": self.circuit_breaker.state.value,
            }

        except Exception as e:
            logger.error(f"GPU service health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_state": self.circuit_breaker.state.value,
            }

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        return {
            "client_stats": self._stats.copy(),
            "service_endpoint": self.service_endpoint,
            "circuit_breaker_state": self.circuit_breaker.state.value,
        }

    async def get_service_stats(self) -> dict[str, Any]:
        """Get service statistics from GPU service."""
        try:
            if not self.stub:
                return {"error": "Service stub not available"}

            request = StatsRequest()
            response = await self.stub.GetStats(request)

            return {
                "service_stats": [
                    {
                        "metric_name": metric.metric_name,
                        "value": metric.value,
                        "unit": metric.unit,
                        "timestamp": metric.timestamp.ToDatetime() if metric.timestamp else None,
                        "labels": dict(metric.labels),
                    }
                    for metric in response.metrics
                ],
                "generated_at": (
                    response.generated_at.ToDatetime() if response.generated_at else None
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get service stats: {e}")
            return {"error": str(e)}


class GPUClientManager:
    """Manager for multiple GPU client instances.

    Handles load balancing and failover between multiple GPU services.
    """

    def __init__(
        self,
        service_endpoints: list[str],
        mtls_manager: mTLSManager | None = None,
        service_name: str = "gpu-management",
    ):
        """Initialize the GPU client manager.

        Args:
            service_endpoints: List of gRPC endpoints for GPU services
            mtls_manager: Optional mTLS manager for secure communication
            service_name: Name of the service for mTLS

        """
        self.service_endpoints = service_endpoints
        self.mtls_manager = mtls_manager
        self.service_name = service_name
        self.clients: list[GPUClient] = []
        self.current_client_index = 0

    async def initialize(self) -> None:
        """Initialize all GPU clients."""
        for endpoint in self.service_endpoints:
            client = GPUClient(
                endpoint, mtls_manager=self.mtls_manager, service_name=self.service_name
            )
            await client.initialize()
            self.clients.append(client)

    async def close(self) -> None:
        """Close all GPU clients."""
        for client in self.clients:
            await client.close()
        self.clients.clear()

    async def get_status(self, include_detailed_info: bool = True) -> dict[str, Any]:
        """Get GPU status from available GPU client.

        Args:
            include_detailed_info: Whether to include detailed device information

        Returns:
            GPU status information

        """
        if not self.clients:
            raise ServiceError("No GPU clients available")

        # Try current client first
        client = self.clients[self.current_client_index]

        try:
            result = await client.get_status(include_detailed_info)
            return result

        except Exception as e:
            logger.warning(f"GPU client {self.current_client_index} failed: {e}")

        # Try other clients
        for i, client in enumerate(self.clients):
            if i == self.current_client_index:
                continue

            try:
                result = await client.get_status(include_detailed_info)
                # Update current client index on success
                self.current_client_index = i
                return result

            except Exception as e:
                logger.warning(f"GPU client {i} failed: {e}")

        # All clients failed
        raise ServiceError("All GPU clients failed")

    async def allocate_gpu(
        self,
        allocation_id: str,
        requested_memory_mb: int,
        preferred_device_id: str | None = None,
        timeout_seconds: int = 300,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Allocate GPU using available GPU client.

        Args:
            allocation_id: Unique allocation identifier
            requested_memory_mb: Requested memory in MB
            preferred_device_id: Preferred GPU device ID
            timeout_seconds: Allocation timeout
            metadata: Additional metadata

        Returns:
            GPU allocation information

        """
        if not self.clients:
            raise ServiceError("No GPU clients available")

        # Try current client first
        client = self.clients[self.current_client_index]

        try:
            result = await client.allocate_gpu(
                allocation_id, requested_memory_mb, preferred_device_id, timeout_seconds, metadata
            )
            return result

        except Exception as e:
            logger.warning(f"GPU client {self.current_client_index} failed: {e}")

        # Try other clients
        for i, client in enumerate(self.clients):
            if i == self.current_client_index:
                continue

            try:
                result = await client.allocate_gpu(
                    allocation_id,
                    requested_memory_mb,
                    preferred_device_id,
                    timeout_seconds,
                    metadata,
                )
                # Update current client index on success
                self.current_client_index = i
                return result

            except Exception as e:
                logger.warning(f"GPU client {i} failed: {e}")

        # All clients failed
        raise ServiceError("All GPU clients failed")

    async def health_check(self) -> dict[str, Any]:
        """Check health of all GPU clients."""
        health_results = []

        for i, client in enumerate(self.clients):
            try:
                health = await client.health_check()
                health_results.append(
                    {
                        "client_index": i,
                        "endpoint": client.service_endpoint,
                        "health": health,
                    }
                )
            except Exception as e:
                health_results.append(
                    {
                        "client_index": i,
                        "endpoint": client.service_endpoint,
                        "health": {"status": "unhealthy", "error": str(e)},
                    }
                )

        return {
            "total_clients": len(self.clients),
            "healthy_clients": len(
                [r for r in health_results if r["health"]["status"] == "healthy"]
            ),
            "client_health": health_results,
        }


def create_gpu_client(service_endpoint: str) -> GPUClient:
    """Create GPU client instance.

    Args:
        service_endpoint: gRPC endpoint for the GPU service

    Returns:
        GPUClient instance

    """
    return GPUClient(service_endpoint)


def create_gpu_client_manager(service_endpoints: list[str]) -> GPUClientManager:
    """Create GPU client manager.

    Args:
        service_endpoints: List of gRPC endpoints for GPU services

    Returns:
        GPUClientManager instance

    """
    return GPUClientManager(service_endpoints)
