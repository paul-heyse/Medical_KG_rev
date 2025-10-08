"""GPU availability helpers for embedders that require CUDA."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from Medical_KG_rev.services import GpuNotAvailableError

logger = structlog.get_logger(__name__)


try:  # pragma: no cover - optional dependency guard
    import torch
except Exception:  # pragma: no cover - fallback when torch unavailable
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class GPUStatus:
    available: bool
    device_name: str | None = None
    device_count: int = 0


def probe() -> GPUStatus:
    if torch is None:
        return GPUStatus(available=False)
    available = bool(torch.cuda.is_available())
    device_count = torch.cuda.device_count() if available else 0
    name = torch.cuda.get_device_name(0) if available else None
    return GPUStatus(available=available, device_name=name, device_count=device_count)


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
