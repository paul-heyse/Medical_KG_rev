"""Batch size optimizer placeholder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OptimizationResult:
    batch_size: int
    confidence: float
    recommendation: str


class BatchSizeOptimizer:
    """Return default batch size recommendations."""

    def __init__(self, initial_batch_size: int = 4) -> None:
        self.initial_batch_size = initial_batch_size

    def recommend(self, metrics: Any | None = None) -> OptimizationResult:
        return OptimizationResult(
            batch_size=self.initial_batch_size,
            confidence=0.0,
            recommendation="insufficient-data",
        )


__all__ = ["BatchSizeOptimizer", "OptimizationResult"]
