"""Utilities for measuring semantic coherence between blocks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .provenance import BlockContext
from .tokenization import TokenCounter, default_token_counter


@dataclass(slots=True)
class CoherenceResult:
    """Represents coherence scores for a sequence of contexts."""

    scores: list[float]
    boundaries: list[int]


class CoherenceCalculator:
    """Calculates cosine similarity based coherence for contexts."""

    def __init__(
        self,
        *,
        embedding_fn: Callable[[Sequence[str]], np.ndarray],
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.embedding_fn = embedding_fn
        self.counter = token_counter or default_token_counter()

    def evaluate(self, contexts: Sequence[BlockContext]) -> CoherenceResult:
        if not contexts:
            return CoherenceResult(scores=[], boundaries=[0])
        sentences = [ctx.text for ctx in contexts]
        embeddings = self.embedding_fn(sentences)
        if embeddings.size == 0:
            return CoherenceResult(scores=[], boundaries=[len(contexts)])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
        sims = np.sum(normalized[1:] * normalized[:-1], axis=1).tolist()
        return CoherenceResult(scores=sims, boundaries=[len(contexts)])


class SemanticDriftDetector:
    """Detects semantic drift across block sequences."""

    def __init__(
        self,
        *,
        threshold: float = 0.8,
        min_tokens: int = 120,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.threshold = threshold
        self.min_tokens = min_tokens
        self.counter = token_counter or default_token_counter()

    def detect(
        self, contexts: Sequence[BlockContext], similarities: Iterable[float]
    ) -> list[int]:
        boundaries: list[int] = []
        token_budget = 0
        for idx, (ctx, sim) in enumerate(zip(contexts[1:], similarities, strict=False), start=1):
            token_budget += ctx.token_count
            if token_budget >= self.min_tokens and sim < self.threshold:
                boundaries.append(idx)
                token_budget = 0
        boundaries.append(len(contexts))
        return boundaries

