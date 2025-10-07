"""Lazy exports for service utilities to avoid optional dependency issues."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "GPU_MEMORY_USED",
    "GPU_UTILIZATION",
    "GpuManager",
    "GpuNotAvailableError",
]

_SERVICE_MAP = {
    "GpuManager": ("Medical_KG_rev.services.gpu.manager", "GpuManager"),
    "GpuNotAvailableError": ("Medical_KG_rev.services.gpu.manager", "GpuNotAvailableError"),
    "GPU_MEMORY_USED": ("Medical_KG_rev.services.gpu.metrics", "GPU_MEMORY_USED"),
    "GPU_UTILIZATION": ("Medical_KG_rev.services.gpu.metrics", "GPU_UTILIZATION"),
}


def __getattr__(name: str) -> Any:
    try:
        module_path, attribute = _SERVICE_MAP[name]
    except KeyError as exc:  # pragma: no cover
        raise AttributeError(name) from exc
    module = import_module(module_path)
    return getattr(module, attribute)


def __dir__() -> list[str]:  # pragma: no cover - tooling aid
    return sorted(__all__)
