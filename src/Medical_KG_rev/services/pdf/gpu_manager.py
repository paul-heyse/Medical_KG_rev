from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

import structlog

from Medical_KG_rev.observability.metrics import GPU_UTILISATION

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class GpuLease:
    """Represents a reserved GPU slot for MinerU processing."""

    gpu_id: str
    started_at: float

    def duration_seconds(self) -> float:
        return max(0.0, time.perf_counter() - self.started_at)


class GpuResourceManager:
    """Lightweight cooperative GPU scheduler for MinerU workers."""

    def __init__(
        self,
        *,
        max_concurrent: int | None = None,
        gpu_memory_mb: int | None = None,
    ) -> None:
        self._max_concurrent = max(1, max_concurrent or 1)
        self._semaphore = threading.Semaphore(self._max_concurrent)
        self._gpu_memory_mb = gpu_memory_mb
        self._active: dict[int, GpuLease] = {}

    @contextmanager
    def reserve(self, tenant_id: str, document_id: str) -> GpuLease:
        start = time.perf_counter()
        logger.debug(
            "gpu.manager.acquire",
            tenant_id=tenant_id,
            document_id=document_id,
            max_concurrent=self._max_concurrent,
        )
        self._semaphore.acquire()
        lease = GpuLease(gpu_id="gpu0", started_at=start)
        self._active[id(lease)] = lease
        self._record_utilisation()
        try:
            yield lease
        finally:
            self._active.pop(id(lease), None)
            self._record_utilisation()
            self._semaphore.release()
            duration = lease.duration_seconds()
            logger.debug(
                "gpu.manager.release",
                tenant_id=tenant_id,
                document_id=document_id,
                duration=round(duration, 3),
            )

    def _record_utilisation(self) -> None:
        active = len(self._active)
        utilisation = (active / self._max_concurrent) * 100.0 if self._max_concurrent else 0.0
        GPU_UTILISATION.labels("mineru").set(utilisation)


__all__ = ["GpuLease", "GpuResourceManager"]
