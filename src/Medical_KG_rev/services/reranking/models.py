"""Typed models shared across the reranking and fusion system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, MutableMapping, Sequence


class FusionStrategy(str, Enum):
    """Supported fusion algorithms."""

    RRF = "rrf"
    WEIGHTED = "weighted"
    LEARNED = "learned"


class NormalizationStrategy(str, Enum):
    """Score normalization approaches supported by the fusion layer."""

    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    SOFTMAX = "softmax"


@dataclass(slots=True)
class QueryDocumentPair:
    """Light-weight representation of a query/document pair for reranking."""

    tenant_id: str
    doc_id: str
    query: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RerankResult:
    """Result of reranking for a single document."""

    doc_id: str
    score: float
    rank: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScoredDocument:
    """Intermediate representation used between retrieval, fusion and reranking."""

    doc_id: str
    content: str
    tenant_id: str
    source: str
    strategy_scores: MutableMapping[str, float] = field(default_factory=dict)
    metadata: MutableMapping[str, Any] = field(default_factory=dict)
    highlights: Sequence[Mapping[str, Any]] = field(default_factory=list)
    score: float = 0.0

    def add_score(self, strategy: str, score: float) -> None:
        self.strategy_scores[strategy] = float(score)

    def copy_for_rank(self) -> "ScoredDocument":
        return ScoredDocument(
            doc_id=self.doc_id,
            content=self.content,
            tenant_id=self.tenant_id,
            source=self.source,
            strategy_scores=dict(self.strategy_scores),
            metadata=dict(self.metadata),
            highlights=list(self.highlights),
            score=self.score,
        )


@dataclass(slots=True)
class RerankingResponse:
    """Envelope returned by rerankers including metrics for observability."""

    results: Sequence[RerankResult]
    metrics: Mapping[str, Any]


@dataclass(slots=True)
class FusionResponse:
    """Container returned by fusion algorithms."""

    documents: Sequence[ScoredDocument]
    metrics: Mapping[str, Any]


@dataclass(slots=True)
class RerankerConfig:
    """Configuration describing how rerankers should be initialised."""

    method: str
    model: str
    batch_size: int = 16
    precision: str = "fp16"
    device: str = "cpu"
    onnx_optimize: bool = False
    quantization: str | None = None
    cache_ttl: int = 3600
    requires_gpu: bool = False
    normalization: NormalizationStrategy = NormalizationStrategy.MIN_MAX


@dataclass(slots=True)
class PipelineSettings:
    """Settings controlling the two stage retrieval pipeline."""

    retrieve_candidates: int = 1000
    rerank_candidates: int = 100
    return_top_k: int = 10


@dataclass(slots=True)
class FusionSettings:
    """Configuration for the fusion layer."""

    strategy: FusionStrategy = FusionStrategy.RRF
    rrf_k: int = 60
    weights: Mapping[str, float] = field(default_factory=dict)
    normalization: NormalizationStrategy = NormalizationStrategy.MIN_MAX
    deduplicate: bool = True


@dataclass(slots=True)
class CacheMetrics:
    """Metrics emitted by the cache manager for observability."""

    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
