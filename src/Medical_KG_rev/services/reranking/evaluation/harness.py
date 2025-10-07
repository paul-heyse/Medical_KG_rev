"""Evaluation harness for reranker comparison."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Mapping, Sequence

import math


@dataclass(slots=True)
class EvaluationResult:
    reranker_id: str
    ndcg_at_10: float
    recall_at_10: float
    mrr: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float

    def summary(self) -> dict[str, float]:
        return {
            "ndcg_at_10": self.ndcg_at_10,
            "recall_at_10": self.recall_at_10,
            "mrr": self.mrr,
            "latency_p95_ms": self.latency_p95_ms,
        }


class RerankerEvaluator:
    """Utility to compare rerankers based on relevance judgements."""

    def __init__(self, ground_truth: dict[str, set[str]]) -> None:
        self.ground_truth = ground_truth

    def evaluate(
        self,
        reranker_id: str,
        ranked_lists: Mapping[str, Sequence[str]],
        latencies_ms: Sequence[float],
    ) -> EvaluationResult:
        ndcg_scores = [
            self._ndcg(documents, 10, self.ground_truth.get(query, set()))
            for query, documents in ranked_lists.items()
        ]
        recall_scores = [
            self._recall(documents, 10, self.ground_truth.get(query, set()))
            for query, documents in ranked_lists.items()
        ]
        mrr_scores = [
            self._mrr(documents, self.ground_truth.get(query, set()))
            for query, documents in ranked_lists.items()
        ]
        return EvaluationResult(
            reranker_id=reranker_id,
            ndcg_at_10=mean(ndcg_scores) if ndcg_scores else 0.0,
            recall_at_10=mean(recall_scores) if recall_scores else 0.0,
            mrr=mean(mrr_scores) if mrr_scores else 0.0,
            latency_p50_ms=self._percentile(latencies_ms, 50),
            latency_p95_ms=self._percentile(latencies_ms, 95),
            latency_p99_ms=self._percentile(latencies_ms, 99),
        )

    def _dcg(self, ranked: Sequence[str], k: int, relevant: set[str]) -> float:
        score = 0.0
        for index, doc_id in enumerate(ranked[:k], start=1):
            if doc_id in relevant:
                score += 1.0 / (math.log2(index + 1))
        return score

    def _ndcg(self, ranked: Sequence[str], k: int, relevant: set[str]) -> float:
        ideal = self._dcg(sorted(relevant), k, relevant)
        if ideal == 0:
            return 0.0
        return self._dcg(ranked, k, relevant) / ideal

    def _recall(self, ranked: Sequence[str], k: int, relevant: set[str]) -> float:
        if not relevant:
            return 0.0
        hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
        return hits / len(relevant)

    def _mrr(self, ranked: Sequence[str], relevant: set[str]) -> float:
        for index, doc_id in enumerate(ranked, start=1):
            if doc_id in relevant:
                return 1.0 / index
        return 0.0

    def _percentile(self, values: Sequence[float], percentile: int) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(round((percentile / 100) * (len(sorted_values) - 1)))
        return sorted_values[index]

    # ------------------------------------------------------------------
    def build_tradeoff_curve(
        self, evaluations: Sequence[EvaluationResult]
    ) -> list[tuple[float, float]]:
        """Return (latency, ndcg) points for plotting accuracy vs latency."""

        return [
            (result.latency_p95_ms, result.ndcg_at_10)
            for result in sorted(evaluations, key=lambda item: item.latency_p95_ms)
        ]

    def ab_test(
        self, baseline: EvaluationResult, challenger: EvaluationResult
    ) -> dict[str, float]:
        """Compare two rerankers returning deltas for key metrics."""

        return {
            "ndcg_delta": challenger.ndcg_at_10 - baseline.ndcg_at_10,
            "recall_delta": challenger.recall_at_10 - baseline.recall_at_10,
            "mrr_delta": challenger.mrr - baseline.mrr,
            "latency_delta": challenger.latency_p95_ms - baseline.latency_p95_ms,
        }

    def leaderboard(
        self, evaluations: Sequence[EvaluationResult]
    ) -> list[EvaluationResult]:
        """Sort rerankers by nDCG@10 descending while favouring lower latency ties."""

        return sorted(
            evaluations,
            key=lambda result: (result.ndcg_at_10, -result.latency_p95_ms),
            reverse=True,
        )
