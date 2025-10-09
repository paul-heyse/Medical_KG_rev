"""Utilities for evaluating embedding quality across namespaces."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class EvaluationDataset:
    """Lightweight description of an evaluation dataset."""

    name: str
    queries: Mapping[str, Sequence[str]]
    relevant: Mapping[str, set[str]]
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class MetricResult:
    metric: str
    value: float


@dataclass(slots=True)
class NamespaceLeaderboard:
    namespace: str
    metrics: list[MetricResult] = field(default_factory=list)

    def add(self, metric: str, value: float) -> None:
        self.metrics.append(MetricResult(metric=metric, value=value))


@dataclass(slots=True)
class ABTestResult:
    control_namespace: str
    variant_namespace: str
    control_score: float
    variant_score: float
    lift: float


class EmbeddingEvaluator:
    """Evaluates retrieval metrics for a provided search callable."""

    def __init__(
        self,
        dataset: EvaluationDataset,
        retrieve: Callable[[str, str, int], Sequence[Mapping[str, object]]],
    ) -> None:
        self.dataset = dataset
        self.retrieve = retrieve

    def evaluate(self, namespace: str, *, k: int = 10) -> NamespaceLeaderboard:
        leaderboard = NamespaceLeaderboard(namespace=namespace)
        recalls: list[float] = []
        ndcgs: list[float] = []
        mrrs: list[float] = []
        for query_id, texts in self.dataset.queries.items():
            for text in texts:
                results = list(self.retrieve(namespace, text, k))
                relevant = self.dataset.relevant.get(query_id, set())
                recalls.append(self._recall(results, relevant, k))
                ndcgs.append(self._ndcg(results, relevant, k))
                mrrs.append(self._mrr(results, relevant))
        leaderboard.add("recall@k", sum(recalls) / max(len(recalls), 1))
        leaderboard.add("ndcg@k", sum(ndcgs) / max(len(ndcgs), 1))
        leaderboard.add("mrr", sum(mrrs) / max(len(mrrs), 1))
        logger.info(
            "embedding.eval.completed",
            namespace=namespace,
            recall=leaderboard.metrics[0].value,
            ndcg=leaderboard.metrics[1].value,
            mrr=leaderboard.metrics[2].value,
        )
        return leaderboard

    def ab_test(
        self,
        control_namespace: str,
        variant_namespace: str,
        *,
        k: int = 10,
    ) -> ABTestResult:
        control = self.evaluate(control_namespace, k=k).metrics[0].value
        variant = self.evaluate(variant_namespace, k=k).metrics[0].value
        lift = variant - control
        logger.info(
            "embedding.eval.ab_test",
            control=control_namespace,
            variant=variant_namespace,
            lift=lift,
        )
        return ABTestResult(
            control_namespace=control_namespace,
            variant_namespace=variant_namespace,
            control_score=control,
            variant_score=variant,
            lift=lift,
        )

    def _recall(
        self,
        results: Sequence[Mapping[str, object]],
        relevant: set[str],
        k: int,
    ) -> float:
        if not relevant:
            return 0.0
        hits = sum(1 for result in results[:k] if result.get("_id") in relevant)
        return hits / len(relevant)

    def _ndcg(
        self,
        results: Sequence[Mapping[str, object]],
        relevant: set[str],
        k: int,
    ) -> float:
        dcg = 0.0
        for rank, result in enumerate(results[:k], start=1):
            if result.get("_id") in relevant:
                dcg += 1.0 / math.log2(rank + 1)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
        return dcg / ideal if ideal else 0.0

    def _mrr(self, results: Sequence[Mapping[str, object]], relevant: set[str]) -> float:
        for rank, result in enumerate(results, start=1):
            if result.get("_id") in relevant:
                return 1.0 / rank
        return 0.0
