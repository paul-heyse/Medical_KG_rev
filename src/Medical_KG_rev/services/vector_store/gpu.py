"""GPU availability helpers shared by vector store adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch may be absent in CI
    torch = None  # type: ignore[assignment]


def gpu_available() -> bool:
    """Return True when CUDA devices are visible."""

    if torch is None:
        return False
    return bool(torch.cuda.is_available())


@dataclass(slots=True)
class GPUStats:
    """Represents utilisation information for a CUDA device."""

    device: str
    total_memory: int
    allocated: int

    @property
    def utilisation(self) -> float:
        if not self.total_memory:
            return 0.0
        return float(self.allocated) / float(self.total_memory) * 100.0


@dataclass(slots=True)
class GPUResourceManager:
    """Tracks availability and records fallback decisions."""

    require_gpu: bool = False
    preferred_batch_size: int = 256

    def ensure(self) -> bool:
        """Ensure GPU availability, raising when required."""

        available = gpu_available()
        if self.require_gpu and not available:
            raise RuntimeError("GPU requested but not available")
        return available

    def fallback_message(self, *, operation: str) -> str | None:
        if self.require_gpu or gpu_available():
            return None
        return f"GPU not available for {operation}; falling back to CPU"

    def choose_batch_size(self, *, available: bool, total: int) -> int:
        if total <= self.preferred_batch_size:
            return total
        return self.preferred_batch_size


class GPUFallbackStrategy:
    """Coordinates logging and fallback decisions for GPU operations."""

    def __init__(self, *, logger: Callable[[str], None] | None = None) -> None:
        self.logger = logger

    def guard(self, *, operation: str, require_gpu: bool) -> bool:
        available = gpu_available()
        if require_gpu and not available:
            raise RuntimeError(f"{operation} requires a GPU but none are available")
        if not available and self.logger:
            self.logger(f"gpu_fallback:{operation}")
        return available


def get_gpu_stats() -> list[GPUStats]:
    """Return stats for each visible GPU device."""

    if torch is None or not getattr(torch, "cuda", None) or not torch.cuda.is_available():  # type: ignore[attr-defined]
        return []
    stats: list[GPUStats] = []
    for index in range(torch.cuda.device_count()):  # type: ignore[attr-defined]
        props = torch.cuda.get_device_properties(index)  # type: ignore[attr-defined]
        stats.append(
            GPUStats(
                device=str(index),
                total_memory=int(props.total_memory),
                allocated=int(torch.cuda.memory_allocated(index)),  # type: ignore[attr-defined]
            )
        )
    return stats


def plan_batches(
    total: int,
    *,
    manager: GPUResourceManager,
    logger: Callable[[str], None] | None = None,
) -> Sequence[range]:
    """Yield ranges describing how to batch work depending on GPU availability."""

    available = manager.ensure()
    batch_size = manager.choose_batch_size(available=available, total=total)
    if not available and logger:
        message = manager.fallback_message(operation="batching")
        if message:
            logger(message)
    ranges: list[range] = []
    for start in range(0, total, max(batch_size, 1)):
        ranges.append(range(start, min(start + batch_size, total)))
    return ranges


def summarise_stats(stats: Iterable[GPUStats]) -> dict[str, float]:
    """Summarise GPU stats for Prometheus or logging."""

    stats = list(stats)
    if not stats:
        return {"devices": 0, "max_utilisation": 0.0}
    return {
        "devices": float(len(stats)),
        "max_utilisation": max(stat.utilisation for stat in stats),
    }


__all__ = [
    "GPUFallbackStrategy",
    "GPUResourceManager",
    "GPUStats",
    "get_gpu_stats",
    "gpu_available",
    "plan_batches",
    "summarise_stats",
]

