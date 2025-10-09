"""High level reranking orchestration with caching and circuit breaking."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import structlog
from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.observability.metrics import (
    record_cache_hit_rate,
    record_gpu_memory_alert,
    record_latency_alert,
    record_reranking_error,
    record_reranking_operation,
)

from .errors import CircuitBreakerOpenError, RerankingError
from .factory import RerankerFactory
from .models import RerankingResponse, RerankResult, ScoredDocument
from .pipeline.batch_processor import BatchProcessor
from .pipeline.cache import RerankCacheManager
from .pipeline.circuit import CircuitBreaker
from .pipeline.runtime import RerankRuntime

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

        runtime = RerankRuntime(
            alias=reranker_key,
            reranker=reranker,
            context=context,
            query=query,
            documents=documents,
            top_k=top_k or len(documents),
            explain=explain,
            cache=self.cache,
            batch_processor=self.batch_processor,
        )
        try:
            result = runtime.execute()
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

        response_metrics = dict(result.response.metrics)
        response_metrics["circuit_state"] = self.circuit_breaker.state(reranker.identifier)
        response_metrics.setdefault("gpu_floor_gb", result.gpu_floor_gb)
        result.response.metrics = response_metrics

        record_cache_hit_rate(reranker_key, result.hit_rate)
        record_latency_alert(reranker.identifier, result.duration_seconds, slo_seconds=0.25)
        if reranker.requires_gpu and result.gpu_floor_gb is not None and result.gpu_floor_gb < 0.5:
            record_gpu_memory_alert(reranker.identifier)
        record_reranking_operation(
            reranker.identifier,
            context.tenant_id,
            getattr(reranker, "batch_size", len(result.response.results)),
            result.duration_seconds,
            pairs=len(result.pending_pairs) + len(result.cached_results),
            circuit_state=self.circuit_breaker.state(reranker.identifier),
            gpu_utilisation=None,
        )
        return result.response

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
    def health(self) -> Mapping[str, Mapping[str, object]]:
        status: dict[str, Mapping[str, object]] = {}
        for reranker_id in self.factory.available:
            entry: dict[str, object] = {
                "available": False,
                "circuit_state": self.circuit_breaker.state(reranker_id),
            }
            try:
                reranker = self.factory.resolve(reranker_id)
            except RerankingError:
                entry["error"] = "resolution_failed"
            else:
                entry.update(
                    {
                        "available": True,
                        "identifier": reranker.identifier,
                        "model_version": getattr(reranker, "model_version", "unknown"),
                        "requires_gpu": getattr(reranker, "requires_gpu", False),
                        "batch_size": getattr(reranker, "batch_size", None),
                        "circuit_state": self.circuit_breaker.state(reranker.identifier),
                    }
                )
            status[reranker_id] = entry
        return status
