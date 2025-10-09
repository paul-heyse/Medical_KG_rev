"""Runtime coordinating staged reranking execution."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import TypeVar

import structlog
from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.observability.metrics import record_pipeline_stage

from ..errors import InvalidPairFormatError, RerankingError
from ..models import QueryDocumentPair, RerankingResponse, RerankResult, ScoredDocument
from ..ports import RerankerPort
from .batch_processor import BatchProcessor
from .cache import RerankCacheManager

logger = structlog.get_logger(__name__)

T = TypeVar("T")


def _normalise_label(label: str) -> str:
    return label.replace(" ", "_").replace(":", ".")


@dataclass(slots=True)
class PipelineRuntime:
    """Reusable helper capturing stage timings and metadata."""

    name: str
    stage_prefix: str | None = None
    emit_stage_metric: Callable[[str, float], None] | None = None
    timings: MutableMapping[str, float] = field(default_factory=dict)
    metadata: MutableMapping[str, object] = field(default_factory=dict)

    def annotate(self, key: str, value: object) -> None:
        self.metadata[key] = value

    def _label(self, stage: str) -> str:
        prefix = self.stage_prefix
        if prefix:
            return f"{_normalise_label(prefix)}.{_normalise_label(stage)}"
        return _normalise_label(stage)

    def record(self, stage: str, duration_seconds: float) -> None:
        self.timings[f"{stage}_ms"] = round(duration_seconds * 1000, 3)
        if self.emit_stage_metric is not None:
            self.emit_stage_metric(self._label(stage), duration_seconds)

    def mark(self, stage: str, duration_seconds: float = 0.0) -> None:
        self.record(stage, duration_seconds)

    @contextmanager
    def stage(self, stage: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.record(stage, perf_counter() - start)

    def run(self, stage: str, func: Callable[..., T], *args, **kwargs) -> T:
        start = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            self.record(stage, perf_counter() - start)


@dataclass(slots=True)
class RuntimeResult:
    """Container describing the outcome of a staged reranking run."""

    response: RerankingResponse
    cached_results: Sequence[RerankResult]
    pending_pairs: Sequence[QueryDocumentPair]
    fresh_results: Sequence[RerankResult]
    timings: Mapping[str, float]
    duration_seconds: float
    hit_rate: float
    cache_snapshot: Mapping[str, float | int]
    gpu_floor_gb: float | None
    metadata: Mapping[str, object]


@dataclass(slots=True)
class RerankRuntime:
    """Execute reranking in well defined stages for observability and reuse."""

    alias: str
    reranker: RerankerPort
    context: SecurityContext
    query: str
    documents: Sequence[ScoredDocument]
    top_k: int
    explain: bool
    cache: RerankCacheManager
    batch_processor: BatchProcessor

    pipeline: PipelineRuntime = field(init=False)

    def __post_init__(self) -> None:
        self.pipeline = PipelineRuntime(
            name=self.alias,
            stage_prefix=f"rerank.{self.alias}",
            emit_stage_metric=record_pipeline_stage,
        )
        self.pipeline.annotate("alias", self.alias)
        self.pipeline.annotate("tenant", self.context.tenant_id)
        self.pipeline.annotate("documents", len(self.documents))

    def execute(self) -> RuntimeResult:
        total_start = perf_counter()
        logger.debug(
            "rerank.runtime.start",
            alias=self.alias,
            reranker=self.reranker.identifier,
            tenant=self.context.tenant_id,
            documents=len(self.documents),
        )
        pairs = self.pipeline.run("prepare", self._prepare_pairs)
        cached, pending = self.pipeline.run("cache", self._partition_cache, pairs)
        fresh: list[RerankResult] = []
        gpu_floor: float | None = None
        if pending:
            fresh, gpu_floor = self.pipeline.run("score", self._score_pending, pending)
            if fresh:
                self.pipeline.run("store", self._store_results, fresh)
            else:
                self.pipeline.mark("store")
        combined = list(cached) + fresh
        limit = self.top_k if self.top_k else len(combined)
        ordered = list(enumerate(combined))
        ordered.sort(key=lambda item: (-item[1].score, item[0]))
        trimmed: list[RerankResult] = []
        for rank, (_, result) in enumerate(ordered[:limit], start=1):
            result.rank = rank
            trimmed.append(result)
        duration = perf_counter() - total_start
        cache_snapshot = asdict(self.cache.metrics())
        hit_rate = (len(cached) / len(pairs)) if pairs else 0.0
        metrics = {
            "model": self.reranker.identifier,
            "alias": self.alias,
            "version": getattr(self.reranker, "model_version", "unknown"),
            "evaluated": len(pending),
            "cached": len(cached),
            "fresh": len(fresh),
            "duration_ms": round(duration * 1000, 3),
            "timing": dict(self.pipeline.timings),
            "cache": cache_snapshot,
            "batch_size": getattr(self.reranker, "batch_size", 0),
        }
        logger.debug(
            "rerank.runtime.complete",
            alias=self.alias,
            reranker=self.reranker.identifier,
            duration_ms=metrics["duration_ms"],
            cached=len(cached),
            fresh=len(fresh),
        )
        response = RerankingResponse(results=trimmed, metrics=metrics)
        return RuntimeResult(
            response=response,
            cached_results=cached,
            pending_pairs=pending,
            fresh_results=fresh,
            timings=dict(self.pipeline.timings),
            duration_seconds=duration,
            hit_rate=hit_rate,
            cache_snapshot=cache_snapshot,
            gpu_floor_gb=gpu_floor,
            metadata=dict(self.pipeline.metadata),
        )

    # ------------------------------------------------------------------
    def _prepare_pairs(self) -> list[QueryDocumentPair]:
        pairs: list[QueryDocumentPair] = []
        for document in self.documents:
            tenant = document.tenant_id or self.context.tenant_id
            if tenant != self.context.tenant_id:
                raise RerankingError(
                    title="Tenant isolation violation",
                    status=403,
                    detail=(
                        f"Document '{document.doc_id}' belongs to tenant '{tenant}'"
                    ),
                )
            if not document.content:
                raise InvalidPairFormatError(
                    f"Document '{document.doc_id}' is missing textual content"
                )
            metadata = dict(document.metadata)
            metadata.setdefault("retrieval_source", document.source)
            pairs.append(
                QueryDocumentPair(
                    tenant_id=tenant,
                    doc_id=document.doc_id,
                    query=self.query,
                    text=document.content,
                    metadata=metadata,
                )
            )
        return pairs

    # ------------------------------------------------------------------
    def _partition_cache(
        self, pairs: Sequence[QueryDocumentPair]
    ) -> tuple[list[RerankResult], list[QueryDocumentPair]]:
        cached: list[RerankResult] = []
        pending: list[QueryDocumentPair] = []
        for pair in pairs:
            cached_result = self.cache.lookup(
                self.reranker.identifier,
                pair.tenant_id,
                pair.doc_id,
                getattr(self.reranker, "model_version", "v1"),
            )
            if cached_result is not None:
                cached.append(cached_result)
            else:
                pending.append(pair)
        return cached, pending

    # ------------------------------------------------------------------
    def _score_pending(
        self, pending: Sequence[QueryDocumentPair]
    ) -> tuple[list[RerankResult], float | None]:
        queue = list(
            self.batch_processor.iter_batches(
                pending, preferred_size=getattr(self.reranker, "batch_size", len(pending))
            )
        )
        scored: list[RerankResult] = []
        gpu_floor: float | None = None
        while queue:
            batch = queue.pop(0)
            response, duration = self.batch_processor.time_batch(
                batch,
                lambda items: self.reranker.score_pairs(
                    items,
                    explain=self.explain,
                ),
            )
            extra = self.batch_processor.split_on_timeout(batch, duration)
            if extra:
                queue = list(extra) + queue
                continue
            scored.extend(response.results)
            if getattr(self.reranker, "requires_gpu", False):
                snapshot = self.batch_processor.gpu_memory_snapshot()
                if snapshot is not None:
                    gpu_floor = snapshot if gpu_floor is None else min(gpu_floor, snapshot)
        return scored, gpu_floor

    # ------------------------------------------------------------------
    def _store_results(self, results: Sequence[RerankResult]) -> None:
        if not results:
            return
        self.cache.store(
            self.reranker.identifier,
            self.context.tenant_id,
            getattr(self.reranker, "model_version", "v1"),
            results,
        )
