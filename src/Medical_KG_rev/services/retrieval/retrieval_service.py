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
        fusion_service: FusionService | None = None,
        pipeline_settings: PipelineSettings | None = None,
        reranking_engine: RerankingEngine | None = None,
        reranking_settings: RerankingSettings | None = None,
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

        fusion_cfg = reranking_settings.fusion if reranking_settings else None
        fusion_settings = FusionSettings(
            strategy=FusionStrategy(fusion_cfg.strategy)
            if fusion_cfg
            else FusionStrategy.RRF,
            rrf_k=fusion_cfg.rrf_k if fusion_cfg else 60,
            weights=fusion_cfg.weights if fusion_cfg else {},
            normalization=NormalizationStrategy(fusion_cfg.normalization)
            if fusion_cfg
            else NormalizationStrategy.MIN_MAX,
            deduplicate=fusion_cfg.deduplicate if fusion_cfg else True,
        )
        self._fusion = fusion_service or FusionService(fusion_settings)

        ttl = reranking_settings.cache_ttl if reranking_settings else 3600
        failure_threshold = (
            reranking_settings.circuit_breaker_failures if reranking_settings else 5
        )
        reset_timeout = (
            reranking_settings.circuit_breaker_reset if reranking_settings else 30.0
        )
        batch_size = (
            reranking_settings.model.batch_size if reranking_settings else 64
        )
        self._reranking_engine = reranking_engine or RerankingEngine(
            factory=RerankerFactory(),
            cache=RerankCacheManager(ttl_seconds=ttl),
            batch_processor=BatchProcessor(max_batch_size=batch_size),
            circuit_breaker=CircuitBreaker(
                failure_threshold=failure_threshold, reset_timeout=reset_timeout
            ),
        )
        pipeline_cfg = reranking_settings.pipeline if reranking_settings else None
        pipeline_settings = pipeline_settings or PipelineSettings(
            retrieve_candidates=pipeline_cfg.retrieve_candidates if pipeline_cfg else 1000,
            rerank_candidates=pipeline_cfg.rerank_candidates if pipeline_cfg else 100,
            return_top_k=pipeline_cfg.return_top_k if pipeline_cfg else 10,
        )
        self._pipeline = TwoStagePipeline(
            fusion=self._fusion,
            reranking=self._reranking_engine,
            settings=pipeline_settings,
        )
        # Backwards compatible attribute
        self.reranker = reranker or CrossEncoderReranker()
        self._default_reranker = (
            reranking_settings.model.reranker_id
            if reranking_settings
            else "cross_encoder:bge"
        )

    def search(
        self,
        index: str,
        query: str,
        filters: Mapping[str, object] | None = None,
        k: int = 10,
        rerank: bool = False,
        *,
        reranker_id: str | None = None,
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
        dense_results = self._dense_search(query, k, security_context)

        default_reranker = reranker_id or self._default_reranker
        candidate_lists = {
            "bm25": self._materialise_documents(
                bm25_results, security_context, strategy="bm25"
            ),
            "splade": self._materialise_documents(
                splade_results, security_context, strategy="splade"
            ),
            "dense": self._materialise_documents(
                dense_results, security_context, strategy="dense"
            ),
        }
        fused, metrics = self._pipeline.execute(
            security_context,
            query,
            candidate_lists,
            reranker_id=default_reranker,
            top_k=k,
            rerank=rerank,
        )
        results: list[RetrievalResult] = []
        for rank, document in enumerate(fused, start=1):
            retrieval_score = float(document.metadata.get("retrieval_score", document.score))
            results.append(
                RetrievalResult(
                    id=document.doc_id,
                    text=document.content,
                    retrieval_score=retrieval_score,
                    rerank_score=document.score if rerank else None,
                    highlights=list(document.highlights),
                    metadata=dict(document.metadata),
                )
            )
        if rerank:
            for result in results:
                result.metadata.setdefault("reranking", metrics.get("reranking", {}))
        return results

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

    def _materialise_documents(
        self,
        results: Sequence[Mapping[str, object]],
        context: SecurityContext,
        *,
        strategy: str,
    ) -> list[ScoredDocument]:
        documents: list[ScoredDocument] = []
        for result in results:
            doc_id = str(result.get("_id"))
            source = result.get("_source", {})
            if not isinstance(source, Mapping):
                source = {}
            metadata = dict(source)
            metadata.setdefault("strategy", strategy)
            tenant = str(metadata.get("tenant_id", context.tenant_id))
            text = str(metadata.get("text", ""))
            score = float(result.get("_score", 0.0))
            document = ScoredDocument(
                doc_id=doc_id,
                content=text,
                tenant_id=tenant,
                source=str(metadata.get("source", strategy)),
                strategy_scores={strategy: score},
                metadata=metadata,
                highlights=list(result.get("highlight", [])),
                score=score,
            )
            documents.append(document)
        return documents

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
