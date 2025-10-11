"""Shared helpers promoting modular reranker implementations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any



def clamp(value: float, *, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp *value* between *lower* and *upper* bounds."""
    if lower > upper:
        lower, upper = upper, lower
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return lower
    return max(lower, min(upper, numeric))


def mean_or_default(values: Iterable[float], default: float = 0.0) -> float:
    """Compute the mean of *values* ignoring non-numeric inputs."""
    numeric: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            numeric.append(float(value))
    if not numeric:
        return default
    return float(sum(numeric) / len(numeric))


@dataclass(slots=True, frozen=True)
class FeatureView:
    """Light-weight accessor for metadata supplied with reranking pairs."""

    metadata: Mapping[str, Any]

    def get_float(self, key: str, default: float = 0.0) -> float:
        value = self.metadata.get(key)
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    def get_sequence(self, key: str) -> Sequence[Any]:
        value = self.metadata.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return value
        return ()

    def get_mapping(self, key: str) -> Mapping[str, Any]:
        value = self.metadata.get(key)
        if isinstance(value, Mapping):
            return value
        return {}

    def flag(self, key: str) -> bool:
        value = self.metadata.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "on"}
        return False
