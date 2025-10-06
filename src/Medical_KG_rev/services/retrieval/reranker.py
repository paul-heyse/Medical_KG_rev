"""Simple deterministic cross-encoder style reranker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping


@dataclass(slots=True)
class CrossEncoderReranker:
    name: str = "cross-encoder-mini"

    def rerank(
        self,
        query: str,
        candidates: Iterable[Mapping[str, object]],
        text_field: str = "text",
    ) -> List[Mapping[str, object]]:
        scored: List[Mapping[str, object]] = []
        query_terms = set(query.lower().split())
        for candidate in candidates:
            text = str(candidate.get(text_field, ""))
            terms = set(text.lower().split())
            overlap = len(query_terms & terms)
            score = float(overlap) / max(len(query_terms), 1)
            enriched = dict(candidate)
            enriched["rerank_score"] = score
            scored.append(enriched)
        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored
