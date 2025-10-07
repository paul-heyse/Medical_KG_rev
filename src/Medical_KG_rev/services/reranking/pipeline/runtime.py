"""Runtime coordinating staged reranking execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Mapping, MutableMapping, Sequence

import structlog

from Medical_KG_rev.auth.context import SecurityContext

from ..errors import InvalidPairFormatError, RerankingError
from ..models import QueryDocumentPair, RerankResult, RerankingResponse, ScoredDocument
from ..ports import RerankerPort
from .batch_processor import BatchProcessor
from .cache import RerankCacheManager

logger = structlog.get_logger(__name__)


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

    timings: MutableMapping[str, float] = field(default_factory=dict)

    def execute(self) -> RuntimeResult:
        total_start = perf_counter()
        logger.debug(
            "rerank.runtime.start",
            alias=self.alias,
            reranker=self.reranker.identifier,
            tenant=self.context.tenant_id,
            documents=len(self.documents),
        )
        pairs = self._prepare_pairs()
        cached, pending = self._partition_cache(pairs)
        fresh: list[RerankResult] = []
        gpu_floor: float | None = None
        if pending:
            fresh, gpu_floor = self._score_pending(pending)
            self._store_results(fresh)
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
            "timing": dict(self.timings),
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
            timings=dict(self.timings),
            duration_seconds=duration,
            hit_rate=hit_rate,
            cache_snapshot=cache_snapshot,
            gpu_floor_gb=gpu_floor,
        )

    # ------------------------------------------------------------------
    def _prepare_pairs(self) -> list[QueryDocumentPair]:
        stage_start = perf_counter()
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
        self.timings["prepare_ms"] = round((perf_counter() - stage_start) * 1000, 3)
        return pairs

    # ------------------------------------------------------------------
    def _partition_cache(
        self, pairs: Sequence[QueryDocumentPair]
    ) -> tuple[list[RerankResult], list[QueryDocumentPair]]:
        stage_start = perf_counter()
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
        self.timings["cache_ms"] = round((perf_counter() - stage_start) * 1000, 3)
        return cached, pending

    # ------------------------------------------------------------------
    def _score_pending(
        self, pending: Sequence[QueryDocumentPair]
    ) -> tuple[list[RerankResult], float | None]:
        stage_start = perf_counter()
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
        self.timings["score_ms"] = round((perf_counter() - stage_start) * 1000, 3)
        return scored, gpu_floor

    # ------------------------------------------------------------------
    def _store_results(self, results: Sequence[RerankResult]) -> None:
        if not results:
            self.timings.setdefault("store_ms", 0.0)
            return
        stage_start = perf_counter()
        self.cache.store(
            self.reranker.identifier,
            self.context.tenant_id,
            getattr(self.reranker, "model_version", "v1"),
            results,
        )
        self.timings["store_ms"] = round((perf_counter() - stage_start) * 1000, 3)
