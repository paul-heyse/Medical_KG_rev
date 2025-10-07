"""Multi-strategy retrieval service combining sparse and dense search."""

from __future__ import annotations

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
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
from Medical_KG_rev.services.reranking.pipeline.two_stage import TwoStagePipeline


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
        granularity_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.opensearch = opensearch
        self.faiss = faiss
        self.vector_store = vector_store
        self.vector_namespace = vector_namespace
        self._context_factory = context_factory
        self._granularity_weights = {
            "document": 0.6,
            "section": 1.0,
            "paragraph": 1.2,
            "window": 0.9,
            "table": 0.8,
        }
        if granularity_weights:
            self._granularity_weights.update(granularity_weights)

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
        dense_results = self._dense_search(query, k, security_context)
        fused = self._fuse_results([bm25_results, splade_results, dense_results])
        if filters and "granularity" in filters:
            allowed = filters["granularity"]
            if isinstance(allowed, str):
                allowed_set = {allowed}
            elif isinstance(allowed, Sequence):
                allowed_set = {str(value) for value in allowed}
            else:
                allowed_set = {str(allowed)}
            fused = [result for result in fused if result.granularity in allowed_set]
        fused = self._merge_neighbors(fused)
        if rerank:
            for result in results:
                result.metadata.setdefault("reranking", metrics.get("reranking", {}))
        if explain:
            for result, document in zip(results, fused, strict=False):
                result.metadata.setdefault("pipeline_metrics", metrics)
                result.metadata.setdefault("fusion", metrics.get("fusion", {}))
                result.metadata.setdefault("timing", metrics.get("timing", {}))
                result.metadata.setdefault(
                    "strategy_scores",
                    dict(document.strategy_scores),
                )
        return results

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
                weight = self._granularity_weights.get(
                    str(result["_source"].get("granularity", "paragraph")),
                    1.0,
                )
                data["rrf"] += weight * (1.0 / (50 + rank))
        fused: list[RetrievalResult] = []
        for chunk_id, payload in aggregated.items():
            metadata = dict(payload["metadata"])
            granularity = str(metadata.get("granularity", "paragraph"))
            fused.append(
                RetrievalResult(
                    id=chunk_id,
                    text=str(payload["text"]),
                    retrieval_score=float(payload["rrf"]),
                    rerank_score=None,
                    highlights=list(payload["highlights"]),
                    metadata=metadata,
                    granularity=granularity,
                )
            )
            documents.append(document)
        return documents

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
            actor=context.subject,
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
