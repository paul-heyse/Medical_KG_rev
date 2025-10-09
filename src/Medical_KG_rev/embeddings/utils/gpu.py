"""GPU availability helpers for embedders that require CUDA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import structlog
from Medical_KG_rev.services import GpuNotAvailableError

logger = structlog.get_logger(__name__)


try:  # pragma: no cover - optional dependency guard
    import torch
except Exception:  # pragma: no cover - fallback when torch unavailable
    torch = None  # type: ignore[assignment]

_BYTES_IN_MEBIBYTE: Final[int] = 1024 * 1024


@dataclass(slots=True)
class GPUStatus:
    available: bool
    device_name: str | None = None
    device_count: int = 0


@dataclass(slots=True)
class GPUMemoryInfo:
    """Lightweight snapshot of GPU memory availability."""

    available: bool
    total_mb: int = 0
    free_mb: int = 0
    used_mb: int = 0

    @property
    def utilization(self) -> float:
        if not self.available or self.total_mb <= 0:
            return 0.0
        return min(1.0, max(0.0, self.used_mb / self.total_mb))


def probe() -> GPUStatus:
    if torch is None:
        return GPUStatus(available=False)
    available = bool(torch.cuda.is_available())
    device_count = torch.cuda.device_count() if available else 0
    name = torch.cuda.get_device_name(0) if available else None
    return GPUStatus(available=available, device_name=name, device_count=device_count)


def memory_info(device: int = 0) -> GPUMemoryInfo:
    """Return a snapshot of GPU memory usage when CUDA is available."""
    if torch is None or not torch.cuda.is_available():  # pragma: no cover - optional dependency
        return GPUMemoryInfo(available=False)
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except Exception:  # pragma: no cover - torch handles device validation
        logger.debug("embedding.gpu.memory_probe_failed", exc_info=True)
        return GPUMemoryInfo(available=True)
    used_bytes = total_bytes - free_bytes
    return GPUMemoryInfo(
        available=True,
        total_mb=int(total_bytes // _BYTES_IN_MEBIBYTE),
        free_mb=int(free_bytes // _BYTES_IN_MEBIBYTE),
        used_mb=int(used_bytes // _BYTES_IN_MEBIBYTE),
    )


def ensure_available(require_gpu: bool, *, operation: str) -> None:
    if not require_gpu:
        return
    status = probe()
    if not status.available:
        logger.error("embedding.gpu.missing", operation=operation)
        raise GpuNotAvailableError(
            f"GPU is required for {operation} but no CUDA device is available"
        )
    logger.debug(
        "embedding.gpu.available",
        operation=operation,
        device=status.device_name,
        devices=status.device_count,
    )


def ensure_memory_budget(
    require_gpu: bool,
    *,
    operation: str,
    fraction: float | None = None,
    reserve_mb: int | None = None,
) -> None:
    """Fail fast when the GPU memory budget for an operation would be exceeded."""
    if not require_gpu:
        return
    if fraction is None and reserve_mb is None:
        return
    info = memory_info()
    if not info.available:
        raise GpuNotAvailableError(
            f"GPU is required for {operation} but CUDA memory information is unavailable"
        )
    if fraction is not None and info.utilization >= max(0.0, min(1.0, fraction)):
        logger.warning(
            "embedding.gpu.memory_fraction_exceeded",
            operation=operation,
            utilization=info.utilization,
            limit=fraction,
        )
        raise GpuNotAvailableError(
            f"GPU memory utilization {info.utilization:.2f} exceeds limit {fraction:.2f}"
        )
    if reserve_mb is not None and info.free_mb < max(0, reserve_mb):
        logger.warning(
            "embedding.gpu.memory_reserve_exceeded",
            operation=operation,
            free_mb=info.free_mb,
            required_mb=reserve_mb,
        )
        raise GpuNotAvailableError(
            f"GPU free memory {info.free_mb}MB below reserve {reserve_mb}MB for {operation}"
        )


__all__ = [
    "GPUMemoryInfo",
    "GPUStatus",
    "ensure_available",
    "ensure_memory_budget",
    "memory_info",
    "probe",
]
