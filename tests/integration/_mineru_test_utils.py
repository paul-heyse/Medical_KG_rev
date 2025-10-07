from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from Medical_KG_rev.config.settings import MineruSettings
from Medical_KG_rev.services.gpu.manager import GpuDevice


@dataclass
class FakeGpuManager:
    """GPU manager stub that always exposes a single simulated device."""

    device: GpuDevice = GpuDevice(index=0, name="simulated-gpu", total_memory_mb=32768)

    def wait_for_gpu(self, timeout: float | None = None):  # pragma: no cover - trivial helper
        return self.device

    def get_device(self) -> GpuDevice:  # pragma: no cover - trivial helper
        return self.device

    @contextmanager
    def device_session(
        self,
        service_name: str,
        *,
        required_memory_mb: int = 0,
        warmup: bool = False,
    ):
        yield self.device


def build_mineru_settings(**overrides: Any) -> MineruSettings:
    """Construct :class:`MineruSettings` with sensible defaults for tests."""

    workers_overrides = overrides.pop("workers", {})
    cache_overrides = overrides.pop("cache", None)

    count = overrides.pop("worker_count", workers_overrides.pop("count", 1))
    batch_default = min(count, workers_overrides.pop("batch_size", count))
    payload: dict[str, Any] = {
        "enabled": overrides.pop("enabled", True),
        "cli_command": overrides.pop("cli_command", "mineru"),
        "simulate_if_unavailable": overrides.pop("simulate_if_unavailable", True),
        "workers": {
            "count": count,
            "vram_per_worker_gb": overrides.pop(
                "vram_per_worker_gb", workers_overrides.pop("vram_per_worker_gb", 1)
            ),
            "timeout_seconds": overrides.pop(
                "timeout_seconds", workers_overrides.pop("timeout_seconds", 120)
            ),
            "batch_size": overrides.pop("batch_size", batch_default),
            "enable_prevalidation": overrides.pop(
                "enable_prevalidation", workers_overrides.pop("enable_prevalidation", False)
            ),
            **workers_overrides,
        },
    }
    if cache_overrides is not None:
        payload["cache"] = cache_overrides
    payload.update(overrides)
    return MineruSettings(**payload)


__all__ = ["FakeGpuManager", "build_mineru_settings"]
