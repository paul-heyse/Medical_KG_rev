"""Shared helpers for reranker implementations."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import structlog

from .errors import GPUUnavailableError
from .fusion import normalization
from .models import NormalizationStrategy, QueryDocumentPair, RerankResult, RerankingResponse
from .ports import RerankerPort

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class BatchScore:
    """Container returned by batch scorers with optional metadata."""

    scores: Sequence[float]
    extra_metadata: Sequence[Mapping[str, Any]] | None = None


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
        normalize: bool | NormalizationStrategy = True,
        batch_size: int | None = None,
        explain: bool = False,
    ) -> RerankingResponse:
        if self.requires_gpu and not self._gpu_available():
            raise GPUUnavailableError(self.identifier)

        started = perf_counter()
        evaluated: list[RerankResult] = []
        limit = len(pairs) if top_k is None else min(len(pairs), top_k)
        active_batch_size = max(1, batch_size or self.batch_size)

        batches: Iterable[Sequence[QueryDocumentPair]] = self._iter_batches(
            pairs[:limit], active_batch_size
        )
        offset = 0
        for batch in batches:
            batch_result = self._score_batch(batch, explain=explain)
            scores = list(batch_result.scores)
            metadata_overrides = list(batch_result.extra_metadata or [])
            for index, pair in enumerate(batch):
                metadata = dict(pair.metadata)
                if metadata_overrides:
                    override = metadata_overrides[index] if index < len(metadata_overrides) else {}
                    metadata.update(override)
                evaluated.append(
                    RerankResult(
                        doc_id=pair.doc_id,
                        score=float(scores[index]),
                        rank=offset + index + 1,
                        metadata=metadata,
                    )
                )
            offset += len(batch)

        strategy: NormalizationStrategy | None
        if isinstance(normalize, NormalizationStrategy):
            strategy = normalize
        elif normalize:
            strategy = NormalizationStrategy.MIN_MAX
        else:
            strategy = None

        if strategy is not None and evaluated:
            scores = [result.score for result in evaluated]
            match strategy:
                case NormalizationStrategy.MIN_MAX:
                    normalised = normalization.min_max(scores)
                case NormalizationStrategy.Z_SCORE:
                    normalised = normalization.z_score(scores)
                case NormalizationStrategy.SOFTMAX:
                    normalised = normalization.softmax(scores)
                case _:
                    normalised = scores
            for result, value in zip(evaluated, normalised):
                result.score = float(value)

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
    def _iter_batches(
        self, pairs: Sequence[QueryDocumentPair], batch_size: int
    ) -> Iterable[Sequence[QueryDocumentPair]]:
        for start in range(0, len(pairs), batch_size):
            yield pairs[start : start + batch_size]

    def _score_batch(
        self, batch: Sequence[QueryDocumentPair], *, explain: bool = False
    ) -> BatchScore:
        scores = [self._score_pair(pair) for pair in batch]
        return BatchScore(scores=scores)

    def _gpu_available(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - torch optional
            return False

    def _score_pair(self, pair: QueryDocumentPair) -> float:  # pragma: no cover - abstract
        raise NotImplementedError
