"""GPU device detection and resource management utilities."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)

try:  # pragma: no cover - optional dependency, exercised in tests via monkeypatch
    import torch
except Exception:  # pragma: no cover - torch is optional in unit tests
    torch = None  # type: ignore


class GpuNotAvailableError(RuntimeError):
    """Raised when CUDA GPUs are unavailable for the microservices."""


@dataclass(frozen=True)
class GpuDevice:
    """Representation of a single GPU device."""

    index: int
    name: str
    total_memory_mb: int


class GpuManager:
    """GPU device detection and resource management."""

    def __init__(self, min_memory_mb: int = 1000, preferred_device: int | None = None):
        self.min_memory_mb = min_memory_mb
        self.preferred_device = preferred_device
        self._device_cache: GpuDevice | None = None
        self._lock = asyncio.Lock()

    def _ensure_torch(self):
        if torch is None:
            raise GpuNotAvailableError("PyTorch with CUDA support is required for GPU services")
        if not torch.cuda.is_available():
            raise GpuNotAvailableError("CUDA is not available on this host")
        return torch

    def _select_device(self) -> GpuDevice:
        cached = self._device_cache
        if cached is not None:
            return cached

        with self._lock:
            if self._device_cache is not None:
                return self._device_cache

            lib = self._ensure_torch()
            device_count = lib.cuda.device_count()
            if device_count == 0:
                raise GpuNotAvailableError("No CUDA devices detected")

            indices = (
                [self.preferred_device]
                if self.preferred_device is not None
                else list(range(device_count))
            )

            for index in indices:
                if index is None or index < 0 or index >= device_count:
                    continue
                props = lib.cuda.get_device_properties(index)
                total_memory_mb = int(props.total_memory / (1024 * 1024))
                if total_memory_mb < self.min_memory_mb:
                    logger.warning(
                        "gpu.device.skipped",
                        device=index,
                        total_memory_mb=total_memory_mb,
                        required_mb=self.min_memory_mb,
                    )
                    continue
                device = GpuDevice(index=index, name=props.name, total_memory_mb=total_memory_mb)
                self._device_cache = device
                return device

            raise GpuNotAvailableError(
                f"No GPU device with {self.min_memory_mb}MB+ memory available"
            )

    def get_available_device(self) -> GpuDevice:
        """Get an available GPU device."""
        return self._select_device()

    def allocate_gpu(self, device: GpuDevice | None = None) -> GpuDevice:
        """Allocate a GPU device."""
        if device is None:
            device = self.get_available_device()

        logger.info("gpu.allocated", device=device.name, index=device.index)
        return device

    def deallocate_gpu(self, device: GpuDevice) -> None:
        """Deallocate a GPU device."""
        logger.info("gpu.deallocated", device=device.name, index=device.index)


# Legacy compatibility
def ensure_torch():
    """Legacy function for torch availability."""
    manager = GpuManager()
    return manager._ensure_torch()


def get_available_device() -> GpuDevice:
    """Legacy function for getting available device."""
    manager = GpuManager()
    return manager.get_available_device()


def allocate_gpu(device: GpuDevice | None = None) -> GpuDevice:
    """Legacy function for allocating GPU."""
    manager = GpuManager()
    return manager.allocate_gpu(device)


def deallocate_gpu(device: GpuDevice) -> None:
    """Legacy function for deallocating GPU."""
    manager = GpuManager()
    manager.deallocate_gpu(device)
