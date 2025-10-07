from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

from Medical_KG_rev.config.settings import MineruSettings
from Medical_KG_rev.services.gpu.manager import GpuDevice, GpuManager, GpuNotAvailableError

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class MineruGpuAllocation:
    """Represents an allocation decision for a MinerU worker."""

    device: GpuDevice
    vram_limit_mb: int


class MineruGpuManager:
    """Specialised GPU manager that enforces MinerU-specific constraints."""

    def __init__(self, base_manager: GpuManager, settings: MineruSettings) -> None:
        self._manager = base_manager
        self._settings = settings

    def ensure_cuda_version(self) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - torch optional in tests
            raise GpuNotAvailableError("PyTorch with CUDA support is required") from exc
        version = getattr(torch.version, "cuda", None)
        if version is None:
            raise GpuNotAvailableError("CUDA runtime version unknown")
        expected = self._settings.cuda_version
        if not version.startswith(str(expected)):
            raise GpuNotAvailableError(
                f"CUDA version mismatch: expected {expected}, found {version}"
            )
        logger.info("mineru.cuda.validated", expected=expected, detected=version)

    def allocate(self) -> MineruGpuAllocation:
        device = self._manager.get_device()
        required = self._settings.workers.vram_per_worker_mb
        if device.total_memory_mb < required:
            raise GpuNotAvailableError(
                f"GPU {device.index} has insufficient VRAM for MinerU (required {required} MB)"
            )
        return MineruGpuAllocation(device=device, vram_limit_mb=required)

    def wait_for_allocation(self, timeout: float = 10.0) -> MineruGpuAllocation:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return self.allocate()
            except GpuNotAvailableError as exc:
                last_error = exc
                time.sleep(0.5)
        raise GpuNotAvailableError(str(last_error) if last_error else "GPU unavailable")


def reserve_devices(manager: GpuManager, settings: MineruSettings) -> list[GpuDevice]:
    """Reserve GPU devices for MinerU workers (best-effort)."""

    allocated: list[GpuDevice] = []
    for _ in range(settings.workers.count):
        device = manager.get_device()
        allocated.append(device)
    logger.info("mineru.gpu.reserved", devices=[device.index for device in allocated])
    return allocated


__all__ = ["MineruGpuAllocation", "MineruGpuManager", "reserve_devices"]
