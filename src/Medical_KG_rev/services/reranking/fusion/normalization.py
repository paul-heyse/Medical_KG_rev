"""Score normalization utilities for fusion algorithms."""

from __future__ import annotations

from collections.abc import Sequence
from math import exp
from statistics import mean, pstdev



def min_max(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.5 for _ in values]
    span = hi - lo
    return [(value - lo) / span for value in values]


def z_score(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    mu = mean(values)
    sigma = pstdev(values)
    if sigma == 0:
        return [0.0 for _ in values]
    return [(value - mu) / sigma for value in values]


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [exp(value - max_value) for value in values]
    denominator = sum(exps)
    if denominator == 0:
        return [0.0 for _ in values]
    return [value / denominator for value in exps]


def apply_normalization(strategy: str, values: Sequence[float], method: str) -> list[float]:
    match method:
        case "min_max":
            return min_max(values)
        case "z_score":
            return z_score(values)
        case "softmax":
            return softmax(values)
        case _:
            raise ValueError(f"Unknown normalization method '{method}'")
