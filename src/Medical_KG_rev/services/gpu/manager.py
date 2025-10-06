"""GPU device detection and resource management utilities."""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import structlog

from .metrics import GPU_MEMORY_USED, GPU_UTILIZATION

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
    """Detects CUDA devices and enforces fail-fast semantics when unavailable."""

    def __init__(self, *, min_memory_mb: int = 0, preferred_device: Optional[int] = None) -> None:
        self.min_memory_mb = min_memory_mb
        self.preferred_device = preferred_device
        self._device_cache: Optional[GpuDevice] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Device detection helpers
    # ------------------------------------------------------------------
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
                logger.info("gpu.device.selected", device=index, name=props.name)
                return device

            raise GpuNotAvailableError(
                f"No CUDA devices satisfy minimum memory requirement ({self.min_memory_mb} MB)"
            )

    def get_device(self) -> GpuDevice:
        return self._select_device()

    # ------------------------------------------------------------------
    # Memory and utilization helpers
    # ------------------------------------------------------------------
    def _require_memory(self, lib, device: GpuDevice, required_mb: int) -> None:
        if required_mb <= 0:
            return
        with lib.cuda.device(device.index):
            free_bytes, total_bytes = self._memory_info(lib)
        free_mb = int(free_bytes / (1024 * 1024))
        if free_mb < required_mb:
            raise GpuNotAvailableError(
                f"Insufficient GPU memory: required {required_mb} MB, available {free_mb} MB"
            )

    def _memory_info(self, lib):  # pragma: no cover - exercised via monkeypatch in tests
        if hasattr(lib.cuda, "mem_get_info"):
            return lib.cuda.mem_get_info()  # type: ignore[misc]
        # Fallback: approximate using allocated memory
        device_index = lib.cuda.current_device()
        total = lib.cuda.get_device_properties(device_index).total_memory
        allocated = lib.cuda.memory_allocated(device_index)
        return total - allocated, total

    def _record_metrics(self, service_name: str, device: GpuDevice, lib) -> None:
        device_label = f"cuda:{device.index}"
        try:
            with lib.cuda.device(device.index):
                utilization = getattr(lib.cuda, "utilization", None)
                if callable(utilization):
                    util_value = float(utilization(device.index))  # pragma: no cover - depends on NVML
                else:
                    util_value = 0.0
                GPU_UTILIZATION.labels(service=service_name, device=device_label).set(util_value)
                allocated, total = lib.cuda.memory_allocated(device.index), lib.cuda.get_device_properties(
                    device.index
                ).total_memory
                GPU_MEMORY_USED.labels(service=service_name, device=device_label, state="allocated").set(
                    allocated / (1024 * 1024)
                )
                GPU_MEMORY_USED.labels(service=service_name, device=device_label, state="total").set(
                    total / (1024 * 1024)
                )
        except Exception as exc:  # pragma: no cover - metrics best-effort
            logger.warning("gpu.metrics.failed", error=str(exc))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def device_session(
        self,
        service_name: str,
        *,
        required_memory_mb: int = 0,
        warmup: bool = False,
    ) -> Iterator[GpuDevice]:
        """Context manager that reserves the GPU for a microservice call."""

        lib = self._ensure_torch()
        device = self.get_device()
        self._require_memory(lib, device, required_memory_mb)
        previous = lib.cuda.current_device() if lib.cuda.is_initialized() else None
        try:
            with lib.cuda.device(device.index):
                if warmup:
                    lib.cuda.synchronize()
                yield device
        finally:
            if previous is not None:
                lib.cuda.set_device(previous)
            self._record_metrics(service_name, device, lib)

    def clear_cache(self) -> None:
        with self._lock:
            self._device_cache = None

    def wait_for_gpu(self, timeout: float = 5.0, interval: float = 0.5) -> GpuDevice:
        """Poll until a GPU becomes available or timeout occurs."""

        deadline = time.monotonic() + timeout
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                return self.get_device()
            except GpuNotAvailableError as exc:
                last_error = exc
                time.sleep(interval)
        raise GpuNotAvailableError(str(last_error) if last_error else "GPU not available")


__all__ = ["GpuManager", "GpuDevice", "GpuNotAvailableError"]
