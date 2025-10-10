"""Pooling strategies used by dense embedding adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

PoolingStrategy = Literal["mean", "max", "cls", "last_token", "none"]


def pool_hidden_states(
    hidden_states: Sequence[Sequence[float]], strategy: PoolingStrategy
) -> list[float]:
    if strategy == "none":
        return list(hidden_states[0]) if hidden_states else []
    if not hidden_states:
        return []
    matrix = np.asarray(hidden_states, dtype=float)
    if strategy == "mean":
        return matrix.mean(axis=0).astype(float).tolist()
    if strategy == "max":
        return matrix.max(axis=0).astype(float).tolist()
    if strategy == "cls":
        return matrix[0].astype(float).tolist()
    if strategy == "last_token":
        return matrix[-1].astype(float).tolist()
    raise ValueError(f"Unknown pooling strategy: {strategy}")
