"""Hybrid retrieval orchestration combining lexical, sparse, and dense signals."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.config import RerankingSettings
from Medical_KG_rev.services.reranking import (
    BatchProcessor,
    CircuitBreaker,
    FusionService,
    FusionSettings,
    FusionStrategy,
    NormalizationStrategy,
    PipelineSettings,
    RerankCacheManager,
    RerankerFactory,
    RerankingEngine,
    ScoredDocument,
)
from Medical_KG_rev.services.vector_store.errors import VectorStoreError
from Medical_KG_rev.services.vector_store.models import VectorQuery
from Medical_KG_rev.services.vector_store.service import VectorStoreService

from .faiss_index import FAISSIndex
from .opensearch_client import OpenSearchClient
from .reranker import CrossEncoderReranker
from .router import RetrievalRouter, RetrievalStrategy, RouterMatch, RoutingRequest


logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class RetrievalResult:
    id: str
    text: str
    retrieval_score: float
    rerank_score: float | None
    highlights: Sequence[Mapping[str, object]]
    metadata: Mapping[str, object]
    granularity: str


class RetrievalService:
    """Coordinates hybrid retrieval across lexical, sparse, and dense namespaces."""

    def __init__(
        self,
        opensearch: OpenSearchClient,
        faiss: FAISSIndex | None = None,
        reranker: Callable[..., Any] | None = None,
        *,
        vector_store: VectorStoreService | None = None,
        vector_namespace: str = "default",
        context_factory: Callable[[], SecurityContext] | None = None,
        router: RetrievalRouter | None = None,
        namespace_map: Mapping[str, str] | None = None,
    ) -> None:
        self.opensearch = opensearch
        self.faiss = faiss
        self.vector_store = vector_store
        self.vector_namespace = vector_namespace
        self._context_factory = context_factory
        self.router = router or RetrievalRouter()
        self._namespace_map = dict(namespace_map or {})

    # ------------------------------------------------------------------
    def search(
        self,
        index: str,
        query: str,
        filters: Mapping[str, object] | None = None,
        k: int = 10,
        rerank: bool = False,
        embedding_kind: str | None = None,
        *,
        reranker_id: str | None = None,
        context: SecurityContext | None = None,
        explain: bool = False,
    ) -> list[RetrievalResult]:
        security_context = context or (
            self._context_factory()
            if self._context_factory
            else SecurityContext(subject="system", tenant_id="system", scopes={"*"})
        )
        namespace = self._resolve_namespace(embedding_kind)
        request = RoutingRequest(
            query=query,
            top_k=k,
            filters=filters or {},
            namespace=namespace,
            context=security_context,
        )
        strategies = self._build_strategies(index, security_context, namespace)
        fused_matches = self.router.execute(request, strategies)
        fused = [
            RetrievalResult(
                id=match.id,
                text=str(match.metadata.get("text", "")),
                retrieval_score=match.score,
                rerank_score=None,
                highlights=list(match.metadata.get("highlights", [])),
                metadata={key: value for key, value in match.metadata.items() if key != "highlights"},
            )
            for match in fused_matches
        ]
        if rerank:
            fused = self._apply_rerank(query, fused)
        fused.sort(key=lambda item: item.rerank_score or item.retrieval_score, reverse=True)
        return fused

    def _dense_strategy(
        self, namespace: str, query: str, k: int, context: SecurityContext
    ) -> list[RouterMatch]:
        pseudo_query = [float(hash(token) % 100) for token in query.split()]
        results: list[RouterMatch] = []
        if self.vector_store is not None:
            try:
                dimension = self.vector_store.registry.get(
                    tenant_id=context.tenant_id, namespace=namespace
                ).params.dimension
                if len(pseudo_query) < dimension:
                    pseudo_query.extend([0.0] * (dimension - len(pseudo_query)))
                elif len(pseudo_query) > dimension:
                    pseudo_query = pseudo_query[:dimension]
                matches = self.vector_store.query(
                    context=context,
                    namespace=namespace,
                    query=VectorQuery(values=pseudo_query, top_k=k),
                )
            except VectorStoreError as exc:
                logger.warning(
                    "retrieval.vector_search.failed",
                    namespace=namespace,
                    error=str(exc),
                )
                return []
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "retrieval.vector_search.failed",
                    namespace=namespace,
                    error=str(exc),
                )
                return []
            for match in matches:
                metadata = dict(match.metadata)
                metadata.setdefault("namespace", namespace)
                results.append(
                    RouterMatch(
                        id=match.vector_id,
                        score=float(match.score),
                        metadata=metadata,
                        source="vector",
                    )
                )
            return results
        if not self.faiss or not self.faiss.ids:
            return []
        if len(pseudo_query) < self.faiss.dimension:
            pseudo_query.extend([0.0] * (self.faiss.dimension - len(pseudo_query)))
        elif len(pseudo_query) > self.faiss.dimension:
            pseudo_query = pseudo_query[: self.faiss.dimension]
        hits = self.faiss.search(pseudo_query, k=k)
        for chunk_id, score, metadata in hits:
            meta = {"text": metadata.get("text", ""), **metadata}
            results.append(
                RouterMatch(
                    id=chunk_id,
                    score=float(score),
                    metadata=meta,
                    source="faiss",
                )
            )
        return results

    def _build_strategies(
        self, index: str, context: SecurityContext, namespace: str
    ) -> list[RetrievalStrategy]:
        def bm25_handler(request: RoutingRequest) -> list[RouterMatch]:
            hits = self.opensearch.search(
                index, request.query, strategy="bm25", filters=request.filters, size=request.top_k
            )
            return [self._opensearch_to_match(hit, "bm25") for hit in hits]

        def splade_handler(request: RoutingRequest) -> list[RouterMatch]:
            hits = self.opensearch.search(
                index, request.query, strategy="splade", filters=request.filters, size=request.top_k
            )
            return [self._opensearch_to_match(hit, "splade") for hit in hits]

        def dense_handler(request: RoutingRequest) -> list[RouterMatch]:
            return self._dense_strategy(namespace, request.query, request.top_k, context)

        strategies = [
            RetrievalStrategy(name="bm25", handler=bm25_handler),
            RetrievalStrategy(name="splade", handler=splade_handler),
            RetrievalStrategy(name="dense", handler=dense_handler, fusion="linear", weight=2.0),
        ]
        return strategies

    def _resolve_namespace(self, embedding_kind: str | None) -> str:
        if embedding_kind and embedding_kind in self._namespace_map:
            return self._namespace_map[embedding_kind]
        return self.vector_namespace

    def _opensearch_to_match(self, hit: Mapping[str, object], source: str) -> RouterMatch:
        metadata = dict(hit.get("_source", {}))
        metadata.setdefault("highlights", hit.get("highlight", []))
        metadata.setdefault("text", metadata.get("text", ""))
        return RouterMatch(
            id=str(hit.get("_id")),
            score=float(hit.get("_score", 0.0)),
            metadata=metadata,
            source=source,
        )

    def _apply_rerank_stub(
        self, results: Sequence[RetrievalResult], reranker_id: str | None
    ) -> list[RetrievalResult]:
        reranked: list[RetrievalResult] = []
        model_name = reranker_id or "cross-encoder:stub"
        for result in results:
            metadata = dict(result.metadata)
            metadata.setdefault("reranking", {"model": model_name, "source": "stub"})
            reranked.append(
                RetrievalResult(
                    id=result.id,
                    text=result.text,
                    retrieval_score=result.retrieval_score,
                    rerank_score=result.retrieval_score * 1.1,
                    highlights=result.highlights,
                    metadata=metadata,
                    granularity=result.granularity,
                )
            )
        return reranked

    # ------------------------------------------------------------------
    def _query_cache_get(self, query: str) -> list[Mapping[str, object]] | None:
        key = (query, tuple(sorted(self.active_namespaces)))
        if key in self._query_cache:
            value = self._query_cache.pop(key)
            self._query_cache[key] = value
            return value
        return None

    def _query_cache_set(self, query: str, results: list[Mapping[str, object]]) -> None:
        key = (query, tuple(sorted(self.active_namespaces)))
        self._query_cache[key] = results
        if len(self._query_cache) > self._cache_size:
            self._query_cache.popitem(last=False)
