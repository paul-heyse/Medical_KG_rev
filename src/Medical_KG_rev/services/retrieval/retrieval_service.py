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
        granularity_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.opensearch = opensearch
        self.faiss = faiss
        self.vector_store = vector_store
        self.vector_namespace = vector_namespace
        self._context_factory = context_factory
        self.embedding_worker = embedding_worker
        self._rrf_k = 60
        if active_namespaces is not None:
            self.active_namespaces = list(active_namespaces)
        elif embedding_worker is not None:
            self.active_namespaces = embedding_worker.active_namespaces
        else:
            self.active_namespaces = [vector_namespace]
        if vector_namespace not in self.active_namespaces:
            self.active_namespaces.append(vector_namespace)
        if namespace_weights is not None:
            self.namespace_weights = dict(namespace_weights)
        else:
            self.namespace_weights = {namespace: 1.0 for namespace in self.active_namespaces}
        self._reranker = reranker
        self._query_cache: OrderedDict[tuple[str, tuple[str, ...]], list[Mapping[str, object]]] = OrderedDict()
        self._cache_size = 32

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
        filters = filters or {}
        lexical_results = self.opensearch.search(index, query, filters=filters, size=k)
        sparse_results = self.opensearch.search(
            index,
            query,
            strategy="splade",
            filters=filters,
            size=k,
        )
        dense_results = self._dense_search(query, k, security_context)
        fused = self._fuse_rrf({"bm25": lexical_results, "splade": sparse_results, "dense": dense_results}, k)
        results = [self._build_result(entry) for entry in fused]
        if rerank and results:
            results = self._apply_rerank_stub(results, reranker_id)
        return results

    # ------------------------------------------------------------------
    def _dense_search(
        self, query: str, k: int, context: SecurityContext
    ) -> list[Mapping[str, object]]:
        if not self.vector_store or not self.embedding_worker:
            return []
        return self._vector_store_search(query, k, context)

    def _vector_store_search(
        self, query: str, k: int, context: SecurityContext
    ) -> list[Mapping[str, object]]:
        cached = self._query_cache_get(query)
        if cached is not None:
            return cached
        embeddings_response = self._encode_query(query, context)
        if not embeddings_response:
            return []
        aggregated: dict[str, Mapping[str, object]] = {}
        try:
            for embedding in embeddings_response:
                if not embedding.vectors:
                    continue
                values = list(embedding.vectors[0])
                registry_entry = self.vector_store.registry.get(
                    tenant_id=context.tenant_id, namespace=embedding.namespace
                )
                dimension = registry_entry.params.dimension
                if len(values) < dimension:
                    values.extend([0.0] * (dimension - len(values)))
                elif len(values) > dimension:
                    values = values[:dimension]
                matches = self.vector_store.query(
                    context=context,
                    namespace=embedding.namespace,
                    query=VectorQuery(values=values, top_k=k),
                )
                for match in matches:
                    payload = dict(match.metadata)
                    payload.setdefault("namespace", embedding.namespace)
                    payload.setdefault("text", str(payload.get("text", "")))
                    score = match.score * self.namespace_weights.get(embedding.namespace, 1.0)
                    current = aggregated.get(match.vector_id)
                    if current is None or score > current["_score"]:
                        aggregated[match.vector_id] = {
                            "_id": match.vector_id,
                            "_score": score,
                            "_source": payload,
                            "highlight": [],
                        }
        except VectorStoreError as exc:
            logger.warning(
                "retrieval.vector_search.failed",
                namespace=self.vector_namespace,
                error=str(exc),
            )
            return []
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "retrieval.vector_search.failed",
                namespace=self.vector_namespace,
                error=str(exc),
            )
            return []
        results = sorted(aggregated.values(), key=lambda item: item["_score"], reverse=True)[:k]
        if results:
            self._query_cache_set(query, results)
        return results

    def _encode_query(self, query: str, context: SecurityContext) -> Sequence[EmbeddingVector]:
        if not self.embedding_worker:
            return []
        request = QueryEmbeddingRequest(
            tenant_id=context.tenant_id,
            chunk_ids=[f"query:{index}" for index in range(len(self.active_namespaces))],
            texts=[query] * len(self.active_namespaces),
            namespaces=self.active_namespaces,
            batch_size=1,
            actor=context.subject,
        )
        response = self.embedding_worker.encode_queries(request)
        return response.vectors

    # ------------------------------------------------------------------
    def _fuse_rrf(
        self,
        candidates: Mapping[str, Sequence[Mapping[str, object]]],
        k: int,
    ) -> list[dict[str, Any]]:
        aggregated: dict[str, dict[str, Any]] = {}
        for strategy, results in candidates.items():
            for rank, result in enumerate(results, start=1):
                doc_id = str(result.get("_id"))
                if not doc_id:
                    continue
                entry = aggregated.setdefault(
                    doc_id,
                    {
                        "doc_id": doc_id,
                        "score": 0.0,
                        "metadata": {},
                        "highlight": [],
                        "strategy_scores": {},
                    },
                )
                increment = 1.0 / (self._rrf_k + rank)
                entry["score"] += increment
                entry["strategy_scores"][strategy] = increment
                source = result.get("_source", {})
                if isinstance(source, Mapping):
                    if not entry["metadata"]:
                        entry["metadata"] = dict(source)
                    else:
                        entry["metadata"].setdefault("text", source.get("text", ""))
                        entry["metadata"].setdefault("document_id", source.get("document_id"))
                if not entry["highlight"] and result.get("highlight"):
                    entry["highlight"] = list(result.get("highlight", []))
        fused = sorted(aggregated.values(), key=lambda item: item["score"], reverse=True)
        return fused[:k]

    def _build_result(self, payload: Mapping[str, Any]) -> RetrievalResult:
        metadata = dict(payload.get("metadata", {}))
        metadata.setdefault("retrieval_score", float(payload.get("score", 0.0)))
        metadata.setdefault("strategy_scores", payload.get("strategy_scores", {}))
        text = str(metadata.get("text", ""))
        granularity = str(metadata.get("granularity", "chunk"))
        return RetrievalResult(
            id=str(payload.get("doc_id")),
            text=text,
            retrieval_score=float(payload.get("score", 0.0)),
            rerank_score=None,
            highlights=list(payload.get("highlight", [])),
            metadata=metadata,
            granularity=granularity,
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
