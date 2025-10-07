"""Protocol definitions for reranker implementations."""

from __future__ import annotations

from typing import Protocol, Sequence

from .models import NormalizationStrategy, QueryDocumentPair, RerankResult, RerankingResponse


class RerankerPort(Protocol):
    """Protocol every reranker implementation must follow."""

    identifier: str
    model_version: str
    supports_batch: bool
    requires_gpu: bool

    def score_pairs(
        self,
        pairs: Sequence[QueryDocumentPair],
        *,
        top_k: int | None = None,
        normalize: bool | NormalizationStrategy = True,
        batch_size: int | None = None,
        explain: bool = False,
    ) -> RerankingResponse:
        """Score the supplied query/document pairs."""

    def warm(self) -> None:
        """Optional hook allowing rerankers to pre-load models."""


class SupportsInt8Quantisation(Protocol):
    """Marker protocol for rerankers that can switch to INT8."""

    def enable_int8(self) -> None:  # pragma: no cover - optional capability
        """Enable INT8 execution for the reranker if supported."""
