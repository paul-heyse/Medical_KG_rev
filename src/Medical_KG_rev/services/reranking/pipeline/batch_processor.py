"""Batch planning utilities for rerankers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Sequence

import structlog

from ..models import QueryDocumentPair

logger = structlog.get_logger(__name__)


class BatchProcessor:
    """Splits reranking jobs into batches based on heuristic limits."""

    def __init__(self, max_batch_size: int = 64) -> None:
        self.max_batch_size = max_batch_size

    def iter_batches(
        self,
        pairs: Sequence[QueryDocumentPair],
        *,
        preferred_size: int,
    ) -> Iterator[Sequence[QueryDocumentPair]]:
        batch_size = min(self.max_batch_size, max(1, preferred_size))
        for index in range(0, len(pairs), batch_size):
            batch = pairs[index : index + batch_size]
            logger.debug(
                "rerank.batch",
                start=index,
                size=len(batch),
                configured=batch_size,
            )
            yield batch

    def adjust_for_gpu(self, requested: int, available_memory: float | None = None) -> int:
        if available_memory is None:
            return min(requested, self.max_batch_size)
        if available_memory < 1.0:
            return max(1, min(requested, self.max_batch_size // 4))
        if available_memory < 4.0:
            return max(1, min(requested, self.max_batch_size // 2))
        return min(requested, self.max_batch_size)
