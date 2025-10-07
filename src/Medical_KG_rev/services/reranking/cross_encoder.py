"""Cross-encoder style rerankers implemented with lightweight heuristics."""

from __future__ import annotations

from math import tanh
from statistics import mean
from typing import Mapping, Sequence

from .base import BaseReranker
from .models import QueryDocumentPair


def _lexical_overlap(query: str, document: str) -> float:
    query_terms = {term for term in query.lower().split() if term}
    doc_terms = {term for term in document.lower().split() if term}
    if not query_terms or not doc_terms:
        return 0.0
    intersection = len(query_terms & doc_terms)
    return intersection / max(len(query_terms), 1)


def _metadata_score(metadata: Mapping[str, object], key: str) -> float:
    value = metadata.get(key)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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
        lexical = _lexical_overlap(pair.query, pair.text)
        dense = _metadata_score(pair.metadata, "dense_score")
        splade = _metadata_score(pair.metadata, "splade_score")
        recency = _metadata_score(pair.metadata, "recency_days")
        recency_factor = 1.0 if recency <= 30 else max(0.2, 1.0 - (recency / 365))
        score = (lexical * 0.55) + (dense * 0.3) + (splade * 0.15)
        if self.precision == "fp16" and self.device.startswith("cuda"):
            score *= 1.05
        return float(min(1.0, max(0.0, score * recency_factor)))


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
        lexical = _lexical_overlap(pair.query, pair.text)
        bm25 = _metadata_score(pair.metadata, "bm25_score")
        dense = _metadata_score(pair.metadata, "dense_score")
        penalty = 0.05 if len(pair.text) > 2000 else 0.0
        score = (lexical * 0.6) + (bm25 * 0.25) + (dense * 0.2) - penalty
        return float(min(1.0, max(0.0, score)))


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
        lexical = _lexical_overlap(pair.query, pair.text)
        dense = _metadata_score(pair.metadata, "dense_score")
        prompt_bias = 0.1 if "relevant" in pair.text.lower() else 0.0
        combined = lexical * 0.4 + dense * 0.4 + prompt_bias
        return float(min(1.0, max(0.0, tanh(combined) * 0.8 + 0.2)))


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
        lexical = _lexical_overlap(pair.query, pair.text)
        semantic = mean(
            value
            for key, value in pair.metadata.items()
            if isinstance(value, (int, float)) and key.endswith("_score")
        ) if pair.metadata else 0.0
        diversity_penalty = 0.1 if pair.metadata.get("is_duplicate") else 0.0
        score = (lexical * 0.5) + (semantic * 0.4) - diversity_penalty + 0.1
        return float(min(1.0, max(0.0, score)))
