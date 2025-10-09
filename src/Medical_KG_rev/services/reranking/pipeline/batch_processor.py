"""Batch planning utilities for rerankers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from time import perf_counter

import structlog

from ..models import QueryDocumentPair

logger = structlog.get_logger(__name__)


try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore


class BatchProcessor:
    """Splits reranking jobs into batches based on heuristic limits."""

    def __init__(
        self,
        max_batch_size: int = 64,
        *,
        monitor_gpu: bool = True,
        batch_timeout: float = 0.5,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.monitor_gpu = monitor_gpu
        self.batch_timeout = batch_timeout

    def iter_batches(
        self,
        pairs: Sequence[QueryDocumentPair],
        *,
        preferred_size: int,
    ) -> Iterator[Sequence[QueryDocumentPair]]:
        batch_size = min(self.max_batch_size, max(1, preferred_size))
        available_memory = self.gpu_memory_snapshot() if self.monitor_gpu else None
        if available_memory is not None:
            batch_size = self.adjust_for_gpu(batch_size, available_memory)
        for index in range(0, len(pairs), batch_size):
            batch = pairs[index : index + batch_size]
            logger.debug(
                "rerank.batch",
                start=index,
                size=len(batch),
                configured=batch_size,
            )
            yield batch

    async def iter_batches_async(
        self,
        pairs: Sequence[QueryDocumentPair],
        *,
        preferred_size: int,
    ) -> AsyncIterator[Sequence[QueryDocumentPair]]:
        for batch in self.iter_batches(pairs, preferred_size=preferred_size):
            yield batch

    def adjust_for_gpu(self, requested: int, available_memory: float | None = None) -> int:
        if available_memory is None:
            return min(requested, self.max_batch_size)
        if available_memory < 1.0:
            return max(1, min(requested, self.max_batch_size // 4))
        if available_memory < 4.0:
            return max(1, min(requested, self.max_batch_size // 2))
        return min(requested, self.max_batch_size)

    def gpu_memory_snapshot(self) -> float | None:
        if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():  # type: ignore[attr-defined]
            return None
        try:  # pragma: no cover - depends on GPU runtime
            free, total = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
            free_gb = float(free) / (1024**3)
            logger.debug("rerank.gpu.memory", free_gb=free_gb)
            return free_gb
        except Exception:
            return None

    def split_on_timeout(
        self,
        batch: Sequence[QueryDocumentPair],
        duration_seconds: float,
    ) -> list[Sequence[QueryDocumentPair]]:
        if self.batch_timeout and duration_seconds > self.batch_timeout and len(batch) > 1:
            midpoint = max(1, len(batch) // 2)
            logger.warning(
                "rerank.batch.timeout",
                size=len(batch),
                duration=duration_seconds,
                timeout=self.batch_timeout,
            )
            return [batch[:midpoint], batch[midpoint:]]
        return []

    def time_batch(
        self,
        batch: Sequence[QueryDocumentPair],
        scorer: Callable[[Sequence[QueryDocumentPair]], object],
    ) -> tuple[object, float]:
        start = perf_counter()
        result = scorer(batch)
        duration = perf_counter() - start
        return result, duration
