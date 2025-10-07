"""Implementation of Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..models import FusionResponse, ScoredDocument


def rrf(
    ranked_lists: Mapping[str, Sequence[ScoredDocument]],
    *,
    k: int = 60,
) -> FusionResponse:
    aggregated: dict[str, ScoredDocument] = {}
    contributions: dict[str, float] = {}
    for strategy, documents in ranked_lists.items():
        for rank, document in enumerate(documents, start=1):
            base = aggregated.setdefault(
                document.doc_id,
                document.copy_for_rank(),
            )
            score = 1.0 / (k + rank)
            base.add_score(strategy, score)
            base.score += score
            contributions.setdefault(document.doc_id, 0.0)
            contributions[document.doc_id] += score
    fused = [doc for doc in aggregated.values()]
    fused.sort(
        key=lambda doc: (
            doc.score,
            float(doc.metadata.get("retrieval_score", 0.0)),
        ),
        reverse=True,
    )
    metrics = {"contributions": contributions, "strategy_count": len(ranked_lists)}
    return FusionResponse(documents=fused, metrics=metrics)
