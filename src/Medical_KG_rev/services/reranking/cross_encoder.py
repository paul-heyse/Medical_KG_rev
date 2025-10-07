"""Cross-encoder style rerankers implemented with lightweight heuristics."""

from __future__ import annotations

from math import tanh

from .base import BaseReranker
from .models import QueryDocumentPair
from .utils import FeatureView, clamp, mean_or_default


def _lexical_overlap(query: str, document: str) -> float:
    query_terms = {term for term in query.lower().split() if term}
    doc_terms = {term for term in document.lower().split() if term}
    if not query_terms or not doc_terms:
        return 0.0
    intersection = len(query_terms & doc_terms)
    return intersection / max(len(query_terms), 1)


def _metadata_score(view: FeatureView, key: str) -> float:
    return view.get_float(key)


class BGEReranker(BaseReranker):
    """Heuristic implementation approximating the behaviour of BGE rerankers."""

    def __init__(self, batch_size: int = 32, precision: str = "fp16", device: str = "cpu") -> None:
        super().__init__(
            identifier="bge-reranker-v2-m3",
            model_version="v1.0",
            batch_size=batch_size,
            requires_gpu=device.startswith("cuda"),
        )
        self.precision = precision
        self.device = device

    def _score_pair(self, pair: QueryDocumentPair) -> float:
        view = FeatureView(pair.metadata)
        lexical = _lexical_overlap(pair.query, pair.text)
        dense = _metadata_score(view, "dense_score")
        splade = _metadata_score(view, "splade_score")
        recency = _metadata_score(view, "recency_days")
        recency_factor = 1.0 if recency <= 30 else max(0.2, 1.0 - (recency / 365))
        score = (lexical * 0.55) + (dense * 0.3) + (splade * 0.15)
        if self.precision == "fp16" and self.device.startswith("cuda"):
            score *= 1.05
        return clamp(score * recency_factor)


class MiniLMReranker(BaseReranker):
    """Fast cross encoder emulating MiniLM throughput."""

    def __init__(self, batch_size: int = 64, device: str = "cpu", quantization: str | None = None) -> None:
        super().__init__(
            identifier="ms-marco-MiniLM-L6-v2",
            model_version="v1.0",
            batch_size=batch_size,
            requires_gpu=device.startswith("cuda"),
        )
        self.device = device
        self.quantization = quantization

    def enable_int8(self) -> None:  # pragma: no cover - optional path
        self.quantization = "int8"

    def _score_pair(self, pair: QueryDocumentPair) -> float:
        view = FeatureView(pair.metadata)
        lexical = _lexical_overlap(pair.query, pair.text)
        bm25 = _metadata_score(view, "bm25_score")
        dense = _metadata_score(view, "dense_score")
        penalty = 0.05 if len(pair.text) > 2000 else 0.0
        score = (lexical * 0.6) + (bm25 * 0.25) + (dense * 0.2) - penalty
        return clamp(score)


class MonoT5Reranker(BaseReranker):
    """Heuristic monoT5 style reranker using prompt inspired features."""

    def __init__(self, batch_size: int = 8, device: str = "cpu") -> None:
        super().__init__(
            identifier="castorini/monot5-base-msmarco",
            model_version="v1.0",
            batch_size=batch_size,
            requires_gpu=device.startswith("cuda"),
        )
        self.device = device

    def _score_pair(self, pair: QueryDocumentPair) -> float:
        view = FeatureView(pair.metadata)
        lexical = _lexical_overlap(pair.query, pair.text)
        dense = _metadata_score(view, "dense_score")
        prompt_bias = 0.1 if "relevant" in pair.text.lower() else 0.0
        combined = lexical * 0.4 + dense * 0.4 + prompt_bias
        return clamp(tanh(combined) * 0.8 + 0.2)


class QwenReranker(BaseReranker):
    """LLM backed reranker that expects responses via an OpenAI compatible API."""

    def __init__(self, endpoint: str | None = None, batch_size: int = 4) -> None:
        super().__init__(
            identifier="qwen-reranker",
            model_version="v1.0",
            batch_size=batch_size,
            requires_gpu=False,
        )
        self.endpoint = endpoint

    def _score_pair(self, pair: QueryDocumentPair) -> float:
        view = FeatureView(pair.metadata)
        lexical = _lexical_overlap(pair.query, pair.text)
        semantic = mean_or_default(
            [value for key, value in pair.metadata.items() if key.endswith("_score")],
            default=0.0,
        )
        diversity_penalty = 0.1 if view.flag("is_duplicate") else 0.0
        score = (lexical * 0.5) + (semantic * 0.4) - diversity_penalty + 0.1
        return clamp(score)
