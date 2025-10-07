"""Feature based rerankers inspired by OpenSearch LTR and Vespa."""

from __future__ import annotations

from math import tanh

from .base import BaseReranker
from .models import QueryDocumentPair


class OpenSearchLTRReranker(BaseReranker):
    def __init__(self, features: dict[str, float] | None = None) -> None:
        super().__init__(
            identifier="opensearch-ltr",
            model_version="v1.0",
            batch_size=32,
            requires_gpu=False,
        )
        self.features = features or {
            "bm25_score": 0.3,
            "splade_score": 0.25,
            "dense_score": 0.3,
            "recency_days": -0.05,
            "title_length": 0.1,
        }

    def _score_pair(self, pair: QueryDocumentPair) -> float:
        raw = 0.0
        for feature, weight in self.features.items():
            value = pair.metadata.get(feature)
            if isinstance(value, (int, float)):
                raw += float(value) * weight
        return float(min(1.0, max(0.0, tanh(raw / 5) * 0.5 + 0.5)))


class VespaRankProfileReranker(BaseReranker):
    def __init__(self, name: str = "biomedical_ranker_v1") -> None:
        super().__init__(
            identifier=f"vespa:{name}",
            model_version="v1.0",
            batch_size=16,
            requires_gpu=False,
        )
        self.name = name

    def _score_pair(self, pair: QueryDocumentPair) -> float:
        dense = pair.metadata.get("dense_score")
        lexical = pair.metadata.get("bm25_score")
        splade = pair.metadata.get("splade_score")
        values = [value for value in (dense, lexical, splade) if isinstance(value, (int, float))]
        if not values:
            return 0.0
        combined = sum(values) / len(values)
        return float(min(1.0, max(0.0, tanh(combined) * 0.6 + 0.4)))
