"""Late interaction reranking based on simplified ColBERT style scoring."""

from __future__ import annotations

from math import sqrt
from typing import Iterable

from .base import BaseReranker
from .models import QueryDocumentPair


def _cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    numerator = 0.0
    sum_a = 0.0
    sum_b = 0.0
    for value_a, value_b in zip(vec_a, vec_b, strict=False):
        numerator += value_a * value_b
        sum_a += value_a * value_a
        sum_b += value_b * value_b
    if sum_a == 0 or sum_b == 0:
        return 0.0
    return numerator / (sqrt(sum_a) * sqrt(sum_b))


class ColBERTReranker(BaseReranker):
    """Implements a lightweight MaxSim computation using token vectors from metadata."""

    def __init__(self, batch_size: int = 16) -> None:
        super().__init__(
            identifier="colbertv2-maxsim",
            model_version="v1.0",
            batch_size=batch_size,
            requires_gpu=False,
        )

    def _score_pair(self, pair: QueryDocumentPair) -> float:
        query_vectors = pair.metadata.get("query_vectors")
        doc_vectors = pair.metadata.get("doc_vectors")
        if not isinstance(query_vectors, list) or not isinstance(doc_vectors, list):
            return 0.0
        max_sim = 0.0
        for query_vector in query_vectors:
            if not isinstance(query_vector, list):
                continue
            similarities = [
                _cosine_similarity(query_vector, doc_vector)
                for doc_vector in doc_vectors
                if isinstance(doc_vector, list)
            ]
            if similarities:
                max_sim += max(similarities)
        if not query_vectors:
            return 0.0
        return float(min(1.0, max(0.0, max_sim / len(query_vectors))))
