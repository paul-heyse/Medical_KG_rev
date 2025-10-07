"""Weighted linear fusion implementation."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..models import FusionResponse, ScoredDocument
from .normalization import apply_normalization


def weighted(
    ranked_lists: Mapping[str, Sequence[ScoredDocument]],
    *,
    weights: Mapping[str, float],
    normalization: str,
) -> FusionResponse:
    if not ranked_lists:
        return FusionResponse(documents=[], metrics={"weights": weights})
    normalised_weights = {
        strategy: float(weight) for strategy, weight in weights.items()
    }
    weight_sum = sum(normalised_weights.values())
    if weight_sum <= 0:
        raise ValueError("Fusion weights must sum to a positive value")
    for strategy in normalised_weights:
        normalised_weights[strategy] /= weight_sum

    aggregated: dict[str, ScoredDocument] = {}
    for strategy, documents in ranked_lists.items():
        if not documents:
            continue
        scores = [doc.score for doc in documents]
        normalised_scores = apply_normalization(strategy, scores, normalization)
        for document, score in zip(documents, normalised_scores, strict=False):
            base = aggregated.setdefault(document.doc_id, document.copy_for_rank())
            weighted_score = score * normalised_weights.get(strategy, 0.0)
            base.add_score(strategy, weighted_score)
            base.score += weighted_score
    fused = [doc for doc in aggregated.values()]
    fused.sort(key=lambda doc: doc.score, reverse=True)
    metrics = {"weights": normalised_weights, "normalization": normalization}
    return FusionResponse(documents=fused, metrics=metrics)
