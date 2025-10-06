"""GPU-enabled microservices for the Medical_KG_rev project."""

from .gpu.manager import GpuManager, GpuNotAvailableError
from .gpu.metrics import GPU_MEMORY_USED, GPU_UTILIZATION

__all__ = [
    "GpuManager",
    "GpuNotAvailableError",
    "GPU_MEMORY_USED",
    "GPU_UTILIZATION",
]
