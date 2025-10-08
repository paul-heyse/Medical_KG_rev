"""Core retrieval metrics used across evaluation workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean
import numpy as np
from sklearn.metrics import ndcg_score


def recall_at_k(relevances: Sequence[float], total_relevant: int, k: int) -> float:
    """Return Recall@K for the given ranking.

    Args:
        relevances: Ordered sequence of graded relevance scores.
        total_relevant: Number of relevant documents for the query.
        k: Rank cutoff.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    hits = sum(1 for grade in relevances[:k] if grade > 0)
    if total_relevant <= 0:
        return 0.0
    return hits / float(total_relevant)


def precision_at_k(relevances: Sequence[float], k: int) -> float:
    """Return Precision@K for the given ranking."""

    if k <= 0:
        raise ValueError("k must be positive")
    if not relevances:
        return 0.0
    hits = sum(1 for grade in relevances[:k] if grade > 0)
    return hits / float(k)


def mean_reciprocal_rank(relevances: Sequence[float]) -> float:
    """Return the reciprocal of the rank of the first relevant item."""

    for index, grade in enumerate(relevances, start=1):
        if grade > 0:
            return 1.0 / float(index)
    return 0.0


def average_precision(relevances: Sequence[float]) -> float:
    """Return mean average precision for a ranked list."""

    hits = 0
    score = 0.0
    for index, grade in enumerate(relevances, start=1):
        if grade > 0:
            hits += 1
            score += hits / float(index)
    return score / float(hits) if hits else 0.0


def _to_numpy(values: Sequence[float]) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    return np.asarray(list(values), dtype=float)


def ndcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Return Normalised Discounted Cumulative Gain at rank *k*.

    The implementation delegates to :func:`sklearn.metrics.ndcg_score` to
    guarantee parity with widely used IR tooling.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if not relevances:
        return 0.0
    ground_truth = _to_numpy(relevances)
    # `ndcg_score` expects a 2D array of shape (n_samples, n_labels)
    ideal = ground_truth.reshape(1, -1)
    predicted = ground_truth.reshape(1, -1)
    return float(ndcg_score(ideal, predicted, k=k))


@dataclass(slots=True)
class RankingMetrics:
    """Container for per-query metric results."""

    metrics: Mapping[str, float]
    judgments: Mapping[str, float]

    def __getitem__(self, item: str) -> float:
        return self.metrics[item]


_DEFAULT_K_VALUES = (5, 10, 20)


def evaluate_ranking(
    retrieved_ids: Sequence[str],
    relevance_judgments: Mapping[str, float],
    *,
    k_values: Iterable[int] = _DEFAULT_K_VALUES,
    include_precision: bool = True,
) -> RankingMetrics:
    """Evaluate a ranked list against graded relevance judgements.

    Returns a mapping with Recall@K, Precision@K (optional), nDCG@K, MRR and MAP.
    """

    relevances = [relevance_judgments.get(doc_id, 0.0) for doc_id in retrieved_ids]
    total_relevant = sum(1 for value in relevance_judgments.values() if value > 0)
    metrics: dict[str, float] = {}
    for k in sorted(set(int(k) for k in k_values)):
        metrics[f"recall@{k}"] = recall_at_k(relevances, total_relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(relevances, k)
        if include_precision:
            metrics[f"precision@{k}"] = precision_at_k(relevances, k)
    metrics["mrr"] = mean_reciprocal_rank(relevances)
    metrics["map"] = average_precision(relevances)
    return RankingMetrics(metrics=metrics, judgments=dict(relevance_judgments))


def mean_metric(values: Iterable[Mapping[str, float]], metric: str) -> float:
    """Utility used by reporting helpers to average a given metric."""

    collected = [payload.get(metric, 0.0) for payload in values]
    return mean(collected) if collected else 0.0


__all__ = [
    "RankingMetrics",
    "average_precision",
    "evaluate_ranking",
    "mean_metric",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
