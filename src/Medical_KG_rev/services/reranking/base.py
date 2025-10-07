"""Shared helpers for reranker implementations."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import math
import structlog

from .errors import GPUUnavailableError
from .models import QueryDocumentPair, RerankResult, RerankingResponse
from .ports import RerankerPort

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class BaseReranker(RerankerPort):
    """Base implementation providing batching, logging and normalisation stubs."""

    identifier: str
    model_version: str
    batch_size: int
    requires_gpu: bool = False
    supports_batch: bool = True

    def score_pairs(
        self,
        pairs: Sequence[QueryDocumentPair],
        *,
        top_k: int | None = None,
        normalize: bool = True,
        batch_size: int | None = None,
    ) -> RerankingResponse:
        if self.requires_gpu and not self._gpu_available():
            raise GPUUnavailableError(self.identifier)

        started = perf_counter()
        evaluated: list[RerankResult] = []
        limit = len(pairs) if top_k is None else min(len(pairs), top_k)
        active_batch_size = max(1, batch_size or self.batch_size)

        for index, pair in enumerate(pairs[:limit]):
            score = self._score_pair(pair)
            evaluated.append(
                RerankResult(
                    doc_id=pair.doc_id,
                    score=float(score),
                    rank=index + 1,
                    metadata=dict(pair.metadata),
                )
            )

        if normalize and evaluated:
            scores = [result.score for result in evaluated]
            minimum = min(scores)
            maximum = max(scores)
            if not math.isclose(maximum, minimum):
                span = maximum - minimum
                for result in evaluated:
                    result.score = (result.score - minimum) / span
            else:
                for result in evaluated:
                    result.score = 0.5  # identical scores

        duration = perf_counter() - started
        metrics: Mapping[str, Any] = {
            "model": self.identifier,
            "version": self.model_version,
            "evaluated": len(evaluated),
            "batch_size": active_batch_size,
            "duration_ms": round(duration * 1000, 3),
        }
        logger.debug(
            "reranker.scored",
            reranker=self.identifier,
            evaluated=len(evaluated),
            duration_ms=metrics["duration_ms"],
        )
        return RerankingResponse(results=evaluated, metrics=metrics)

    # ------------------------------------------------------------------
    def warm(self) -> None:  # pragma: no cover - optional for subclasses
        logger.debug("reranker.warm", reranker=self.identifier)

    # ------------------------------------------------------------------
    def _gpu_available(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - torch optional
            return False

    def _score_pair(self, pair: QueryDocumentPair) -> float:  # pragma: no cover - abstract
        raise NotImplementedError
