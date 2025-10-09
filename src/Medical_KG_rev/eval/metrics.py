"""Retrieval evaluation metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import log2


def ndcg_at_k(relevances: Sequence[float], k: int) -> float:
    gains = [rel for rel in relevances[:k]]
    if not gains:
        return 0.0
    dcg = sum(rel / log2(index + 2) for index, rel in enumerate(gains))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / log2(index + 2) for index, rel in enumerate(ideal[:k]))
    return dcg / idcg if idcg else 0.0


def recall_at_k(relevances: Sequence[float], total_relevant: int, k: int) -> float:
    hits = sum(1 for rel in relevances[:k] if rel > 0)
    return hits / total_relevant if total_relevant else 0.0


def mean_reciprocal_rank(relevances: Sequence[float]) -> float:
    for index, rel in enumerate(relevances, start=1):
        if rel > 0:
            return 1.0 / index
    return 0.0


def average_precision(relevances: Sequence[float]) -> float:
    hits = 0
    score = 0.0
    for index, rel in enumerate(relevances, start=1):
        if rel > 0:
            hits += 1
            score += hits / index
    return score / hits if hits else 0.0


def evaluate_query(
    retrieved_ids: Sequence[str],
    relevance_judgments: Mapping[str, float],
    k_values: Iterable[int] = (1, 5, 10, 20),
) -> dict[str, float]:
    relevances = [relevance_judgments.get(doc_id, 0.0) for doc_id in retrieved_ids]
    total_relevant = sum(1 for value in relevance_judgments.values() if value > 0)
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"ndcg@{k}"] = ndcg_at_k(relevances, k)
        metrics[f"recall@{k}"] = recall_at_k(relevances, total_relevant, k)
    metrics["mrr"] = mean_reciprocal_rank(relevances)
    metrics["map"] = average_precision(relevances)
    return metrics


__all__ = [
    "average_precision",
    "evaluate_query",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "recall_at_k",
]
