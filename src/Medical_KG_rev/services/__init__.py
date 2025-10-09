"""Lazy exports for service utilities to avoid optional dependency issues."""

from __future__ import annotations

import importlib.util
from typing import Any

__all__: list[str] = []

_PROMETHEUS_AVAILABLE = importlib.util.find_spec("prometheus_client") is not None
_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if _PROMETHEUS_AVAILABLE:
    from .gpu.metrics import GPU_MEMORY_USED, GPU_UTILIZATION  # type: ignore

    __all__.extend(["GPU_MEMORY_USED", "GPU_UTILIZATION"])
else:  # pragma: no cover - optional dependency fallback
    GPU_MEMORY_USED = None  # type: ignore[assignment]
    GPU_UTILIZATION = None  # type: ignore[assignment]


if _TORCH_AVAILABLE:
    from .gpu.manager import GpuManager, GpuNotAvailableError  # type: ignore

    __all__.extend(["GpuManager", "GpuNotAvailableError"])
else:  # pragma: no cover - optional dependency fallback

    class GpuNotAvailableError(RuntimeError):
        """Fallback error raised when GPU support is requested without torch."""

    class GpuManager:
        """Stub GPU manager used when torch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise GpuNotAvailableError("PyTorch with CUDA support is required for GPU services")

    __all__.extend(["GpuManager", "GpuNotAvailableError"])
