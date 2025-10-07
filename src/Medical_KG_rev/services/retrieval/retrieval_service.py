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
    ) -> list[Mapping[str, object]]:
        if self.vector_store is not None and self.embedding_worker is not None:
            return self._vector_store_search(query, k, context)
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

    def _merge_neighbors(self, results: Iterable[RetrievalResult]) -> list[RetrievalResult]:
        merged: list[RetrievalResult] = []
        buffer: dict[tuple[str, str], RetrievalResult] = {}
        for result in results:
            if result.granularity != "window":
                merged.append(result)
                continue
            doc_key, chunker, index = self._parse_chunk_id(result.id)
            key = (doc_key, chunker)
            existing = buffer.get(key)
            if existing is None:
                buffer[key] = result
                continue
            prev_doc, prev_chunker, prev_index = self._parse_chunk_id(existing.id)
            if prev_doc == doc_key and prev_chunker == chunker and index == prev_index + 1:
                combined_text = existing.text + "\n" + result.text
                combined_score = max(existing.retrieval_score, result.retrieval_score)
                metadata = dict(existing.metadata)
                metadata.setdefault("merged_ids", [existing.id])
                metadata["merged_ids"].append(result.id)
                metadata["text"] = combined_text
                merged_result = RetrievalResult(
                    id=result.id,
                    text=combined_text,
                    retrieval_score=combined_score,
                    rerank_score=None,
                    highlights=existing.highlights,
                    metadata=metadata,
                    granularity=result.granularity,
                )
                buffer[key] = merged_result
            else:
                merged.append(existing)
                buffer[key] = result
        merged.extend(buffer.values())
        return merged

    def _parse_chunk_id(self, chunk_id: str) -> tuple[str, str, int]:
        parts = chunk_id.split(":")
        if len(parts) < 4:
            return chunk_id, "unknown", 0
        try:
            return parts[0], parts[1], int(parts[-1])
        except ValueError:
            return parts[0], parts[1], 0

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
                    granularity=result.granularity,
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
