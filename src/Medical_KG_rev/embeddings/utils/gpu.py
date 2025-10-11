"""GPU utilities for embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

logger = logging.getLogger(__name__)

# Torch functionality moved to gRPC services
torch = None

_BYTES_IN_MEBIBYTE: Final[int] = 1024 * 1024


@dataclass(slots=True)
class GPUStatus:
    """Status of GPU availability."""

    available: bool
    device_name: str | None = None
    memory_total: int | None = None
    memory_used: int | None = None
    memory_free: int | None = None


@dataclass(slots=True)
class GPUMemoryInfo:
    """GPU memory information."""

    available: bool
    total_mb: int | None = None
    used_mb: int | None = None
    free_mb: int | None = None


def probe() -> GPUStatus:
    """Probe GPU availability and status."""
    # GPU functionality moved to gRPC services
    return GPUStatus(available=False)


def memory_info() -> GPUMemoryInfo:
    """Get GPU memory information."""
    # GPU functionality moved to gRPC services
    return GPUMemoryInfo(available=False)


def is_available() -> bool:
    """Check if GPU is available."""
    # GPU functionality moved to gRPC services
    return False


def get_device_count() -> int:
    """Get number of available GPU devices."""
    # GPU functionality moved to gRPC services
    return 0


def get_device_name(device_id: int = 0) -> str | None:
    """Get GPU device name."""
    # GPU functionality moved to gRPC services
    return None


def get_memory_info(device_id: int = 0) -> GPUMemoryInfo:
    """Get memory information for a specific GPU device."""
    # GPU functionality moved to gRPC services
    return GPUMemoryInfo(available=False)


def clear_cache(device_id: int | None = None) -> None:
    """Clear GPU memory cache."""
    # GPU functionality moved to gRPC services
    logger.info("GPU cache clear requested - handled by gRPC services")


def synchronize() -> None:
    """Synchronize GPU operations."""
    # GPU functionality moved to gRPC services
    logger.info("GPU synchronization requested - handled by gRPC services")


class GPUManager:
    """GPU resource manager."""

    def __init__(self) -> None:
        """Initialize GPU manager."""
        self._available = False
        self._devices: list[dict[str, any]] = []

    def initialize(self) -> None:
        """Initialize GPU resources."""
        # GPU functionality moved to gRPC services
        logger.info("GPU manager initialization - handled by gRPC services")

    def cleanup(self) -> None:
        """Cleanup GPU resources."""
        # GPU functionality moved to gRPC services
        logger.info("GPU manager cleanup - handled by gRPC services")

    def get_status(self) -> GPUStatus:
        """Get current GPU status."""
        return probe()

    def get_memory_info(self) -> GPUMemoryInfo:
        """Get GPU memory information."""
        return memory_info()

    def is_available(self) -> bool:
        """Check if GPU is available."""
        return is_available()

    def allocate_memory(self, size_mb: int) -> bool:
        """Allocate GPU memory."""
        # GPU functionality moved to gRPC services
        logger.info(f"GPU memory allocation requested ({size_mb}MB) - handled by gRPC services")
        return False

    def free_memory(self, size_mb: int) -> None:
        """Free GPU memory."""
        # GPU functionality moved to gRPC services
        logger.info(f"GPU memory free requested ({size_mb}MB) - handled by gRPC services")
