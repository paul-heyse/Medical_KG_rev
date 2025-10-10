"""Batching utilities for embedding workloads."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class BatchProgress:
    """Tracks progress while iterating through batches."""

    total: int
    callback: Callable[[int, int], None] | None = None
    processed: int = 0

    def step(self, processed: int) -> None:
        self.processed += processed
        if self.callback:
            self.callback(self.processed, self.total)


def batched(items: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    """Yield batches of *batch_size* items from *items*."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def paired_batches(
    left: Sequence[T],
    right: Sequence[T],
    batch_size: int,
) -> Iterator[tuple[Sequence[T], Sequence[T]]]:
    """Yield aligned batches from two sequences of equal length."""
    if len(left) != len(right):
        raise ValueError("Sequences must be equal length")
    for left_batch, right_batch in zip(
        batched(left, batch_size), batched(right, batch_size), strict=True
    ):
        yield left_batch, right_batch


def iter_with_progress(
    items: Sequence[T],
    batch_size: int,
    *,
    progress: BatchProgress | None = None,
) -> Iterator[Sequence[T]]:
    """Iterate over *items* in batches while updating the provided progress tracker."""
    for batch in batched(items, batch_size):
        yield batch
        if progress:
            progress.step(len(batch))
