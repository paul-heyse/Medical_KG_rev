"""Multi-strategy retrieval service combining sparse and dense search."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import structlog

from Medical_KG_rev.auth.context import SecurityContext
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


class RetrievalService:
    def __init__(
        self,
        opensearch: OpenSearchClient,
        faiss: FAISSIndex | None = None,
        reranker: CrossEncoderReranker | None = None,
        *,
        vector_store: VectorStoreService | None = None,
        vector_namespace: str = "default",
        context_factory: Callable[[], SecurityContext] | None = None,
        router: RetrievalRouter | None = None,
    ) -> None:
        self.opensearch = opensearch
        self.faiss = faiss
        self.reranker = reranker or CrossEncoderReranker()
        self.vector_store = vector_store
        self.vector_namespace = vector_namespace
        self._context_factory = context_factory
        self.router = router or RetrievalRouter()

    def search(
        self,
        index: str,
        query: str,
        filters: Mapping[str, object] | None = None,
        k: int = 10,
        rerank: bool = False,
        *,
        context: SecurityContext | None = None,
    ) -> list[RetrievalResult]:
        security_context = context or (
            self._context_factory()
            if self._context_factory
            else SecurityContext(subject="system", tenant_id="system", scopes={"*"})
        )
        request = RoutingRequest(
            query=query,
            top_k=k,
            filters=filters or {},
            namespace=self.vector_namespace,
            context=security_context,
        )
        strategies = self._build_strategies(index, security_context)
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
        self, query: str, k: int, context: SecurityContext
    ) -> list[RouterMatch]:
        pseudo_query = [float(hash(token) % 100) for token in query.split()]
        results: list[RouterMatch] = []
        if self.vector_store is not None:
            namespace = self.vector_namespace
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
        self, index: str, context: SecurityContext
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
            return self._dense_strategy(request.query, request.top_k, context)

        strategies = [
            RetrievalStrategy(name="bm25", handler=bm25_handler),
            RetrievalStrategy(name="splade", handler=splade_handler),
            RetrievalStrategy(name="dense", handler=dense_handler, fusion="linear", weight=2.0),
        ]
        return strategies

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

    def _apply_rerank(
        self, query: str, results: Iterable[RetrievalResult]
    ) -> list[RetrievalResult]:
        materialised = list(results)
        candidates = [
            {"id": result.id, "text": result.text, **result.metadata} for result in materialised
        ]
        scored, _metrics = self.reranker.rerank(query, candidates)
        score_map = {item.get("id"): item.get("rerank_score", 0.0) for item in scored}
        reranked: list[RetrievalResult] = []
        for result in materialised:
            reranked.append(
                RetrievalResult(
                    id=result.id,
                    text=result.text,
                    retrieval_score=result.retrieval_score,
                    rerank_score=score_map.get(result.id),
                    highlights=result.highlights,
                    metadata=result.metadata,
                )
            )
        return reranked
