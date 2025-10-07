"""GPU memory planning utilities for MinerU workers."""

from __future__ import annotations

from dataclasses import dataclass

from Medical_KG_rev.services.gpu.manager import GpuDevice


@dataclass(slots=True)
class GpuBudgetPlanner:
    """Determines a safe GPU memory reservation for MinerU CLI execution."""

    configured_requirement_mb: int
    safety_margin: float = 0.9

    def plan(self, device: GpuDevice) -> int:
        """Return the required memory after applying device-aware constraints."""

        if self.configured_requirement_mb <= 0:
            return 0
        margin = min(max(self.safety_margin, 0.0), 1.0)
        safe_limit = int(device.total_memory_mb * margin)
        if safe_limit <= 0:
            return max(0, self.configured_requirement_mb)
        return min(self.configured_requirement_mb, safe_limit)


__all__ = ["GpuBudgetPlanner"]

