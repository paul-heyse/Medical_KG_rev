"""High level fusion orchestrator selecting the correct algorithm."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from . import deduplicate
from ..models import FusionResponse, FusionSettings, FusionStrategy, ScoredDocument
from .rrf import rrf
from .weighted import weighted



class FusionService:
    """Coordinates fusion execution with deduplication and metrics."""

    def __init__(self, settings: FusionSettings | None = None) -> None:
        self.settings = settings or FusionSettings()

    def fuse(self, ranked_lists: Mapping[str, Sequence[ScoredDocument]]) -> FusionResponse:
        if self.settings.strategy is FusionStrategy.RRF:
            fused = rrf(ranked_lists, k=self.settings.rrf_k)
        elif self.settings.strategy is FusionStrategy.WEIGHTED:
            fused = weighted(
                ranked_lists,
                weights=self.settings.weights,
                normalization=self.settings.normalization.value,
            )
        else:  # Learned fusion proxies to weighted but keeps strategy metadata
            fused = weighted(
                ranked_lists,
                weights=self.settings.weights or dict.fromkeys(ranked_lists, 1.0),
                normalization=self.settings.normalization.value,
            )
            fused.metrics = {
                **fused.metrics,
                "strategy": "learned",
            }

        documents = list(fused.documents)
        if self.settings.deduplicate:
            documents = deduplicate.deduplicate(documents)
        fused.documents = documents
        return fused
