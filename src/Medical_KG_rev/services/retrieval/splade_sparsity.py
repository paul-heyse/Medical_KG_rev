"""SPLADE sparsity analysis placeholder."""

from __future__ import annotations

from typing import List


def sparsity(vector: List[float]) -> float:
    if not vector:
        return 0.0
    zeros = sum(1 for value in vector if value == 0.0)
    return zeros / len(vector)


__all__ = ["sparsity"]
