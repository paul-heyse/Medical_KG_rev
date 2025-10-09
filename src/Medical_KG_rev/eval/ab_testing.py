"""Simple A/B testing runner for retrieval configurations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .metrics import evaluate_query


@dataclass(slots=True)
class ABTestOutcome:
    variant_a: str
    variant_b: str
    metrics: Mapping[str, float]
    confidence: float


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    variant_a: str
    variant_b: str
    traffic_split: float = 0.5


class ABTestRunner:
    def __init__(self, *, confidence_level: float = 0.95) -> None:
        self.confidence_level = confidence_level

    def run(
        self,
        *,
        dataset: Iterable[tuple[str, Mapping[str, float], Sequence[str], Sequence[str]]],
        variant_a: str | None = None,
        variant_b: str | None = None,
        config: ExperimentConfig | None = None,
    ) -> ABTestOutcome:
        if config is not None:
            variant_a = config.variant_a
            variant_b = config.variant_b
        if variant_a is None or variant_b is None:
            raise ValueError("variant_a and variant_b must be provided")
        improvements: list[float] = []
        for query_id, judgments, results_a, results_b in dataset:
            metrics_a = evaluate_query(results_a, judgments)
            metrics_b = evaluate_query(results_b, judgments)
            improvements.append(metrics_b["ndcg@10"] - metrics_a["ndcg@10"])
        mean = sum(improvements) / max(len(improvements), 1)
        variance = sum((value - mean) ** 2 for value in improvements) / max(len(improvements) - 1, 1)
        stderr = math.sqrt(variance / max(len(improvements), 1))
        confidence = max(0.0, min(1.0, 0.5 * (1 + math.erf(mean / (stderr * math.sqrt(2))))) if stderr else 1.0)
        metrics = {"mean_ndcg@10_delta": mean, "stderr": stderr}
        return ABTestOutcome(variant_a=variant_a, variant_b=variant_b, metrics=metrics, confidence=confidence)


__all__ = ["ABTestOutcome", "ABTestRunner", "ExperimentConfig"]
