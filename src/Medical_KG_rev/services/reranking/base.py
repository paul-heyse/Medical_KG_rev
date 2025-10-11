"""Shared helpers for reranker implementations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import structlog

from .errors import GPUUnavailableError
from .fusion import normalization
from .models import (
    QueryDocumentPair,
    RerankResult,
    ScoredDocument,
)
from .ports import RerankerPort


@dataclass
class BaseReranker:
    """Base class for reranker implementations."""

    identifier: str
    model_version: str
    batch_size: int
    requires_gpu: bool

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for reranking."""
        try:
            return False  # GPU functionality moved to gRPC services
        except Exception:  # pragma: no cover - torch optional
            return False

    def _score_pair(self, pair: QueryDocumentPair) -> float:  # pragma: no cover - abstract
        raise NotImplementedError
