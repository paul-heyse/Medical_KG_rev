"""Batching utilities for embedding workloads."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")


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
    for left_batch, right_batch in zip(batched(left, batch_size), batched(right, batch_size), strict=True):
        yield left_batch, right_batch
