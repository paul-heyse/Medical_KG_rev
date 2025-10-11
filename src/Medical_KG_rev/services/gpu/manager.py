"""GPU device detection and resource management utilities - Torch-free version."""

from __future__ import annotations

import structlog

from ..clients.gpu_client import GPUClientManager

logger = structlog.get_logger(__name__)


class GpuNotAvailableError(RuntimeError):
    """Raised when CUDA GPUs are unavailable for the microservices."""


class GpuDevice:
    """Representation of a single GPU device."""

    def __init__(self, index: int, name: str, total_memory_mb: int):
        self.index = index
        self.name = name
        self.total_memory_mb = total_memory_mb


class GpuServiceManager:
    """GPU service manager that uses gRPC to communicate with GPU services."""

    def __init__(self, min_memory_mb: int = 1000, preferred_device: int | None = None):
        self.min_memory_mb = min_memory_mb
        self.preferred_device = preferred_device
        self._client_manager: GPUClientManager | None = None
        self._device_cache: GpuDevice | None = None

    async def _get_client_manager(self) -> GPUClientManager:
        """Get or create GPU client manager."""
        if self._client_manager is None:
            self._client_manager = GPUClientManager()
            await self._client_manager.initialize()
        return self._client_manager

    async def get_available_device(self) -> GpuDevice:
        """Get an available GPU device via gRPC service."""
        try:
            client_manager = await self._get_client_manager()

            # Get GPU status from service
            status = await client_manager.get_status()
            if not status or not status.get("available", False):
                raise GpuNotAvailableError("No GPU devices available via gRPC service")

            # Get device list from service
            devices = await client_manager.list_devices()
            if not devices:
                raise GpuNotAvailableError("No GPU devices detected via gRPC service")

            # Select appropriate device
            for device_info in devices:
                if device_info.get("total_memory_mb", 0) >= self.min_memory_mb:
                    device = GpuDevice(
                        index=device_info.get("index", 0),
                        name=device_info.get("name", "Unknown"),
                        total_memory_mb=device_info.get("total_memory_mb", 0),
                    )
                    self._device_cache = device
                    return device

            raise GpuNotAvailableError(
                f"No GPU device with {self.min_memory_mb}MB+ memory available"
            )

        except Exception as e:
            logger.error("gpu.service.error", error=str(e))
            raise GpuNotAvailableError(f"GPU service communication failed: {e}")

    async def allocate_gpu(self, device: GpuDevice | None = None) -> GpuDevice:
        """Allocate a GPU device via gRPC service."""
        if device is None:
            device = await self.get_available_device()

        try:
            client_manager = await self._get_client_manager()
            allocation = await client_manager.allocate_gpu(device.index)

            if not allocation or not allocation.get("success", False):
                raise GpuNotAvailableError("Failed to allocate GPU via gRPC service")

            logger.info("gpu.allocated", device=device.name, index=device.index)
            return device

        except Exception as e:
            logger.error("gpu.allocation.error", error=str(e))
            raise GpuNotAvailableError(f"GPU allocation failed: {e}")

    async def deallocate_gpu(self, device: GpuDevice) -> None:
        """Deallocate a GPU device via gRPC service."""
        try:
            client_manager = await self._get_client_manager()
            success = await client_manager.deallocate_gpu(device.index)

            if not success:
                logger.warning("gpu.deallocation.failed", device=device.name, index=device.index)
            else:
                logger.info("gpu.deallocated", device=device.name, index=device.index)

        except Exception as e:
            logger.error("gpu.deallocation.error", error=str(e))

    async def close(self) -> None:
        """Close the GPU client manager."""
        if self._client_manager:
            await self._client_manager.close()


# Legacy compatibility - raise error for torch-dependent functionality
def ensure_torch():
    """Legacy function - GPU functionality moved to gRPC services."""
    raise GpuNotAvailableError(
        "GPU functionality moved to gRPC services. Use GpuServiceManager instead."
    )


def get_available_device() -> GpuDevice:
    """Legacy function - GPU functionality moved to gRPC services."""
    raise GpuNotAvailableError(
        "GPU functionality moved to gRPC services. Use GpuServiceManager instead."
    )


def allocate_gpu(device: GpuDevice | None = None) -> GpuDevice:
    """Legacy function - GPU functionality moved to gRPC services."""
    raise GpuNotAvailableError(
        "GPU functionality moved to gRPC services. Use GpuServiceManager instead."
    )


def deallocate_gpu(device: GpuDevice) -> None:
    """Legacy function - GPU functionality moved to gRPC services."""
    raise GpuNotAvailableError(
        "GPU functionality moved to gRPC services. Use GpuServiceManager instead."
    )


# Alias for backward compatibility
GpuManager = GpuServiceManager
