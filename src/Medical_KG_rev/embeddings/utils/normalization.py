"""Normalization helpers for embeddings."""

from __future__ import annotations

from math import sqrt
from typing import Iterable, Sequence


def l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return list(vector)
    return [value / norm for value in vector]


def normalize_batch(vectors: Iterable[Sequence[float]]) -> list[list[float]]:
    return [l2_normalize(vector) for vector in vectors]
