"""GPU availability helpers shared by vector store adapters."""

from __future__ import annotations

from dataclasses import dataclass

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
class GPUResourceManager:
    """Tracks availability and records fallback decisions."""

    require_gpu: bool = False

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


__all__ = ["GPUResourceManager", "gpu_available"]

