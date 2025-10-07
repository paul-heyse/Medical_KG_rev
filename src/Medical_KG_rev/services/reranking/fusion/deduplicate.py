"""Utilities for deduplicating retrieval results."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from ..models import ScoredDocument


def deduplicate(
    documents: Sequence[ScoredDocument],
    *,
    aggregation: str = "max",
) -> list[ScoredDocument]:
    groups: dict[str, list[ScoredDocument]] = defaultdict(list)
    for document in documents:
        groups[document.doc_id].append(document)

    merged: list[ScoredDocument] = []
    for doc_id, items in groups.items():
        if not items:
            continue
        base = items[0].copy_for_rank()
        merged_metadata: dict[str, object] = {}
        merged_highlights: list[Mapping[str, object]] = []
        strategy_scores: dict[str, float] = {}
        scores = [item.score for item in items]
        for item in items:
            merged_metadata.update(item.metadata)
            merged_highlights.extend(item.highlights)
            strategy_scores.update(item.strategy_scores)
        match aggregation:
            case "max":
                base.score = max(scores)
            case "mean":
                base.score = sum(scores) / len(scores)
            case "sum":
                base.score = sum(scores)
            case _:
                raise ValueError(f"Unsupported aggregation '{aggregation}'")
        base.metadata.update(merged_metadata)
        if merged_highlights:
            base.highlights = merged_highlights
        if strategy_scores:
            base.strategy_scores.update(strategy_scores)
        merged.append(base)
    merged.sort(key=lambda doc: doc.score, reverse=True)
    return merged
