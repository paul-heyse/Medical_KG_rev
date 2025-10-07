"""High level reranking orchestration with caching and circuit breaking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Iterable, Mapping, Sequence

import structlog

from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.observability.metrics import (
    record_cache_hit_rate,
    record_gpu_memory_alert,
    record_latency_alert,
    record_reranking_error,
    record_reranking_operation,
)

from .errors import CircuitBreakerOpenError, InvalidPairFormatError, RerankingError
from .factory import RerankerFactory
from .models import QueryDocumentPair, RerankResult, RerankerConfig, RerankingResponse, ScoredDocument
from .pipeline.batch_processor import BatchProcessor
from .pipeline.cache import RerankCacheManager
from .pipeline.circuit import CircuitBreaker
from .ports import RerankerPort

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class RerankingEngine:
    factory: RerankerFactory
    cache: RerankCacheManager
    batch_processor: BatchProcessor
    circuit_breaker: CircuitBreaker

    def rerank(
        self,
        *,
        context: SecurityContext,
        query: str,
        documents: Sequence[ScoredDocument],
        reranker_id: str | None,
        top_k: int | None = None,
        explain: bool = False,
    ) -> RerankingResponse:
        reranker_key = reranker_id or "cross_encoder:bge"
        if not context.has_scope("retrieve:read"):
            error = RerankingError(
                title="Missing retrieve scope",
                status=403,
                detail="Scope 'retrieve:read' is required to rerank documents",
            )
            record_reranking_error(reranker_key, "scope")
            raise error
        reranker = self.factory.resolve(reranker_key)
        if not self.circuit_breaker.can_execute(reranker.identifier):
            raise CircuitBreakerOpenError(reranker.identifier)

        pairs = self._build_pairs(context, query, documents)
        top_k = top_k or len(pairs)
        start = perf_counter()
        cached: list[RerankResult] = []
        pending: list[QueryDocumentPair] = []
        for pair in pairs:
            cached_result = self.cache.lookup(
                reranker.identifier,
                pair.tenant_id,
                pair.doc_id,
                reranker.model_version,
            )
            if cached_result is not None:
                cached.append(cached_result)
            else:
                pending.append(pair)

        scored: list[RerankResult] = list(cached)
        if pending:
            try:
                queue: list[Sequence[QueryDocumentPair]] = list(
                    self.batch_processor.iter_batches(
                        pending,
                        preferred_size=reranker.batch_size,
                    )
                )
                while queue:
                    batch = queue.pop(0)
                    def _score(items: Sequence[QueryDocumentPair]) -> RerankingResponse:
                        if explain:
                            try:
                                return reranker.score_pairs(items, top_k=top_k, explain=True)
                            except TypeError:
                                return reranker.score_pairs(items, top_k=top_k)
                        return reranker.score_pairs(items, top_k=top_k)

                    response, batch_duration = self.batch_processor.time_batch(
                        batch,
                        _score,
                    )
                    extra = self.batch_processor.split_on_timeout(batch, batch_duration)
                    if extra:
                        queue = list(extra) + queue
                        continue
                    if reranker.requires_gpu:
                        available = self.batch_processor.gpu_memory_snapshot()
                        if available is not None and available < 0.5:
                            record_gpu_memory_alert(reranker.identifier)
                    scored.extend(response.results)
            except RerankingError as err:
                self.circuit_breaker.record_failure(reranker.identifier)
                record_reranking_error(reranker.identifier, err.__class__.__name__)
                raise
            except Exception as exc:  # pragma: no cover - defensive logging
                self.circuit_breaker.record_failure(reranker.identifier)
                logger.exception("rerank.failed", reranker=reranker.identifier)
                record_reranking_error(reranker.identifier, exc.__class__.__name__)
                raise RerankingError(
                    title="Reranking failed",
                    status=500,
                    detail=str(exc),
                ) from exc
            else:
                self.circuit_breaker.record_success(reranker.identifier)
                self.cache.store(
                    reranker.identifier,
                    context.tenant_id,
                    reranker.model_version,
                    scored,
                )
        else:
            self.circuit_breaker.record_success(reranker.identifier)

        scored.sort(key=lambda result: result.score, reverse=True)
        trimmed = scored[:top_k]
        duration = perf_counter() - start
        metrics: Mapping[str, object] = {
            "model": reranker.identifier,
            "version": reranker.model_version,
            "evaluated": len(pending),
            "cached": len(cached),
            "duration_ms": round(duration * 1000, 3),
            "circuit_state": self.circuit_breaker.state(reranker.identifier),
            "cache": asdict(self.cache.metrics()),
        }
        cache_metrics = metrics["cache"]
        if isinstance(cache_metrics, Mapping):
            hit_rate = float(cache_metrics.get("hit_rate", 0.0))
            record_cache_hit_rate(reranker.identifier, hit_rate)
        record_latency_alert(reranker.identifier, duration, slo_seconds=0.25)
        record_reranking_operation(
            reranker.identifier,
            context.tenant_id,
            reranker.batch_size,
            duration,
            pairs=len(pairs),
            circuit_state=self.circuit_breaker.state(reranker.identifier),
        )
        return RerankingResponse(results=trimmed, metrics=metrics)

    # ------------------------------------------------------------------
    def _build_pairs(
        self,
        context: SecurityContext,
        query: str,
        documents: Sequence[ScoredDocument],
    ) -> list[QueryDocumentPair]:
        pairs: list[QueryDocumentPair] = []
        for document in documents:
            tenant = document.tenant_id or context.tenant_id
            if tenant != context.tenant_id:
                raise RerankingError(
                    title="Tenant isolation violation",
                    status=403,
                    detail=f"Document '{document.doc_id}' belongs to tenant '{tenant}'",
                )
            if not document.content:
                raise InvalidPairFormatError(
                    f"Document '{document.doc_id}' is missing textual content"
                )
            pairs.append(
                QueryDocumentPair(
                    tenant_id=tenant,
                    doc_id=document.doc_id,
                    query=query,
                    text=document.content,
                    metadata=dict(document.metadata),
                )
            )
        return pairs

    # ------------------------------------------------------------------
    def warm_cache(
        self,
        reranker_id: str,
        tenant_id: str,
        version: str,
        results: Iterable[RerankResult],
    ) -> None:
        reranker = self.factory.resolve(reranker_id)
        cache_version = version or reranker.model_version
        self.cache.warm(reranker.identifier, tenant_id, cache_version, results)

    # ------------------------------------------------------------------
    def health(self) -> Mapping[str, bool]:
        status: dict[str, bool] = {}
        for reranker_id in self.factory.available:
            try:
                reranker = self.factory.resolve(reranker_id)
            except RerankingError:
                status[reranker_id] = False
            else:
                status[reranker_id] = self.circuit_breaker.state(reranker.identifier) != "open"
        return status
