"""Lexical rerankers backed by OpenSearch style scores."""

from __future__ import annotations

from collections.abc import Sequence
from math import log1p

from .base import BaseReranker
from .models import QueryDocumentPair, RerankingResponse
from .utils import FeatureView, clamp



class BM25Reranker(BaseReranker):
    def __init__(self, batch_size: int = 128) -> None:
        super().__init__(
            identifier="bm25-rerank",
            model_version="v1.0",
            batch_size=batch_size,
            requires_gpu=False,
        )

    def score_pairs(
        self,
        pairs: Sequence[QueryDocumentPair],
        *,
        top_k: int | None = None,
        normalize: bool = True,
        batch_size: int | None = None,
        explain: bool = False,
    ) -> RerankingResponse:
        response = super().score_pairs(
            pairs,
            top_k=top_k,
            normalize=normalize,
            batch_size=batch_size,
        )
        if explain:
            for result, pair in zip(response.results, pairs, strict=False):
                explanation = self._explain(pair)
                result.metadata = dict(result.metadata)
                result.metadata["bm25_explain"] = explanation
        return response

    def _score_pair(self, pair: QueryDocumentPair) -> float:
        view = FeatureView(pair.metadata)
        bm25 = view.get_float("bm25_score")
        if "bm25_score" in pair.metadata:
            return clamp(log1p(bm25) / 10)
        overlap = len(set(pair.query.lower().split()) & set(pair.text.lower().split()))
        return clamp(overlap / 5)

    def _explain(self, pair: QueryDocumentPair) -> dict[str, object]:
        tokens_query = pair.query.lower().split()
        tokens_doc = pair.text.lower().split()
        overlap = sorted(set(tokens_query) & set(tokens_doc))
        return {
            "bm25_score": pair.metadata.get("bm25_score", 0.0),
            "query_terms": tokens_query,
            "overlap": overlap,
            "document_length": len(tokens_doc),
        }


class BM25FReranker(BaseReranker):
    def __init__(self, field_weights: dict[str, float] | None = None) -> None:
        super().__init__(
            identifier="bm25f-rerank",
            model_version="v1.0",
            batch_size=64,
            requires_gpu=False,
        )
        self.field_weights = field_weights or {"title": 2.0, "body": 1.0}

    def score_pairs(
        self,
        pairs: Sequence[QueryDocumentPair],
        *,
        top_k: int | None = None,
        normalize: bool = True,
        batch_size: int | None = None,
        explain: bool = False,
    ) -> RerankingResponse:
        response = super().score_pairs(
            pairs,
            top_k=top_k,
            normalize=normalize,
            batch_size=batch_size,
        )
        if explain:
            for result, pair in zip(response.results, pairs, strict=False):
                result.metadata = dict(result.metadata)
                result.metadata["bm25f_explain"] = self._explain(pair)
        return response

    def _score_pair(self, pair: QueryDocumentPair) -> float:
        view = FeatureView(pair.metadata)
        total = 0.0
        weight_sum = 0.0
        for field, weight in self.field_weights.items():
            field_key = f"{field}_bm25"
            field_score = view.get_float(field_key)
            if field_key in pair.metadata:
                total += field_score * weight
                weight_sum += weight
        if weight_sum:
            return clamp(log1p(total / weight_sum) / 5)
        return clamp(len(pair.text.split()) / 1000)

    def _explain(self, pair: QueryDocumentPair) -> dict[str, object]:
        contributions: dict[str, float] = {}
        for field, weight in self.field_weights.items():
            score = pair.metadata.get(f"{field}_bm25")
            if isinstance(score, (int, float)):
                contributions[field] = float(score) * weight
        return {
            "field_weights": dict(self.field_weights),
            "contributions": contributions,
        }
