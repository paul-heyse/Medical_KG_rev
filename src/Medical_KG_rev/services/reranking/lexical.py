"""Lexical rerankers backed by OpenSearch style scores."""

from __future__ import annotations

from math import log1p

from .base import BaseReranker
from .models import QueryDocumentPair


class BM25Reranker(BaseReranker):
    def __init__(self, batch_size: int = 128) -> None:
        super().__init__(
            identifier="bm25-rerank",
            model_version="v1.0",
            batch_size=batch_size,
            requires_gpu=False,
        )

    def _score_pair(self, pair: QueryDocumentPair) -> float:
        bm25 = pair.metadata.get("bm25_score")
        if isinstance(bm25, (int, float)):
            return float(min(1.0, max(0.0, log1p(bm25) / 10)))
        overlap = len(set(pair.query.lower().split()) & set(pair.text.lower().split()))
        return float(min(1.0, max(0.0, overlap / 5)))


class BM25FReranker(BaseReranker):
    def __init__(self, field_weights: dict[str, float] | None = None) -> None:
        super().__init__(
            identifier="bm25f-rerank",
            model_version="v1.0",
            batch_size=64,
            requires_gpu=False,
        )
        self.field_weights = field_weights or {"title": 2.0, "body": 1.0}

    def _score_pair(self, pair: QueryDocumentPair) -> float:
        total = 0.0
        weight_sum = 0.0
        for field, weight in self.field_weights.items():
            field_score = pair.metadata.get(f"{field}_bm25")
            if isinstance(field_score, (int, float)):
                total += float(field_score) * weight
                weight_sum += weight
        if weight_sum:
            return float(min(1.0, max(0.0, log1p(total / weight_sum) / 5)))
        return float(min(1.0, max(0.0, len(pair.text.split()) / 1000)))
