"""Multi-strategy retrieval service combining sparse and dense search."""

from __future__ import annotations

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.services.embedding.service import (
    EmbeddingRequest as QueryEmbeddingRequest,
    EmbeddingVector,
    EmbeddingWorker,
)
from Medical_KG_rev.services.vector_store.errors import VectorStoreError
from Medical_KG_rev.services.vector_store.models import VectorQuery
from Medical_KG_rev.services.vector_store.service import VectorStoreService

from .faiss_index import FAISSIndex
from .opensearch_client import OpenSearchClient
from .reranker import CrossEncoderReranker


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
        embedding_worker: EmbeddingWorker | None = None,
        active_namespaces: Sequence[str] | None = None,
        namespace_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.opensearch = opensearch
        self.faiss = faiss
        self.reranker = reranker or CrossEncoderReranker()
        self.vector_store = vector_store
        self.vector_namespace = vector_namespace
        self._context_factory = context_factory
        self.embedding_worker = embedding_worker
        self.active_namespaces = list(active_namespaces or [vector_namespace])
        self.namespace_weights = dict(namespace_weights or {vector_namespace: 1.0})
        self._query_cache: OrderedDict[tuple[str, tuple[str, ...]], list[Mapping[str, Any]]] = OrderedDict()
        self._cache_size = 32

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
        bm25_results = self.opensearch.search(
            index, query, strategy="bm25", filters=filters, size=k
        )
        splade_results = self.opensearch.search(
            index, query, strategy="splade", filters=filters, size=k
        )
        dense_results = self._dense_search(query, k, security_context)
        fused = self._fuse_results([bm25_results, splade_results, dense_results])
        if rerank:
            fused = self._apply_rerank(query, fused)
        fused.sort(key=lambda item: item.rerank_score or item.retrieval_score, reverse=True)
        return fused

    def _dense_search(
        self, query: str, k: int, context: SecurityContext
    ) -> list[Mapping[str, object]]:
        if self.vector_store is not None and self.embedding_worker is not None:
            return self._vector_store_search(query, k, context)
        if not self.faiss or not self.faiss.ids:
            return []
        pseudo_query = [float(hash(token) % 100) for token in query.split()]
        if len(pseudo_query) < self.faiss.dimension:
            pseudo_query.extend([0.0] * (self.faiss.dimension - len(pseudo_query)))
        elif len(pseudo_query) > self.faiss.dimension:
            pseudo_query = pseudo_query[: self.faiss.dimension]
        hits = self.faiss.search(pseudo_query, k=k)
        results: list[Mapping[str, object]] = []
        for chunk_id, score, metadata in hits:
            results.append(
                {
                    "_id": chunk_id,
                    "_score": score,
                    "_source": {"text": metadata.get("text", ""), **metadata},
                    "highlight": [],
                }
            )
        return results

    def _vector_store_search(
        self, query: str, k: int, context: SecurityContext
    ) -> list[Mapping[str, object]]:
        cached = self._query_cache_get(query)
        if cached is not None:
            return cached
        embeddings = self._encode_query(query, context)
        namespace = self.vector_namespace
        try:
            results: list[Mapping[str, object]] = []
            for embedding in embeddings:
                if not embedding.vectors:
                    continue
                dimension = self.vector_store.registry.get(
                    tenant_id=context.tenant_id, namespace=embedding.namespace
                ).params.dimension
                values = list(embedding.vectors[0])
                if len(values) < dimension:
                    values.extend([0.0] * (dimension - len(values)))
                elif len(values) > dimension:
                    values = values[:dimension]
                matches = self.vector_store.query(
                    context=context,
                    namespace=embedding.namespace,
                    query=VectorQuery(values=values, top_k=k),
                )
                results.extend(
                    {
                        "_id": match.vector_id,
                        "_score": match.score * self.namespace_weights.get(embedding.namespace, 1.0),
                        "_source": {"text": str(match.metadata.get("text", "")), **match.metadata},
                        "highlight": [],
                    }
                    for match in matches
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
        if results:
            self._query_cache_set(query, results)
        return results

    def _fuse_results(
        self, result_sets: Sequence[Sequence[Mapping[str, object]]]
    ) -> list[RetrievalResult]:
        aggregated: dict[str, dict[str, object]] = {}
        for results in result_sets:
            for rank, result in enumerate(results, start=1):
                chunk_id = result["_id"]
                data = aggregated.setdefault(
                    chunk_id,
                    {
                        "text": result["_source"].get("text", ""),
                        "metadata": result["_source"],
                        "highlights": list(result.get("highlight", [])),
                        "rrf": 0.0,
                    },
                )
                data["rrf"] += 1.0 / (50 + rank)
        fused: list[RetrievalResult] = []
        for chunk_id, payload in aggregated.items():
            fused.append(
                RetrievalResult(
                    id=chunk_id,
                    text=str(payload["text"]),
                    retrieval_score=float(payload["rrf"]),
                    rerank_score=None,
                    highlights=list(payload["highlights"]),
                    metadata=dict(payload["metadata"]),
                )
            )
        fused.sort(key=lambda item: item.retrieval_score, reverse=True)
        return fused

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

    def _encode_query(self, query: str, context: SecurityContext) -> Sequence[EmbeddingVector]:
        if not self.embedding_worker:
            return []
        namespaces = self.active_namespaces or [self.vector_namespace]
        request = QueryEmbeddingRequest(
            tenant_id=context.tenant_id,
            chunk_ids=[f"query:{index}" for index in range(len(namespaces))],
            texts=[query] * len(namespaces),
            namespaces=namespaces,
            batch_size=1,
        )
        response = self.embedding_worker.encode_queries(request)
        return response.vectors

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
