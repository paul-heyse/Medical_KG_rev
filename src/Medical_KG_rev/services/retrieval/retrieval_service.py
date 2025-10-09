"""Hybrid retrieval service coordinating lexical, sparse, and dense signals."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping, Sequence

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
    RerankerModelRegistry,
    ModelHandle,
    ModelDownloadError,
    ScoredDocument,
)
from Medical_KG_rev.services.reranking.errors import RerankingError
from Medical_KG_rev.services.reranking.pipeline.two_stage import TwoStagePipeline
from Medical_KG_rev.services.vector_store.errors import VectorStoreError
from Medical_KG_rev.services.vector_store.models import VectorQuery
from Medical_KG_rev.services.vector_store.service import VectorStoreService

from .faiss_index import FAISSIndex
from .hybrid import HybridComponentSettings, HybridSearchCoordinator
from .opensearch_client import OpenSearchClient
from .rerank_policy import TenantRerankPolicy
from .reranker import CrossEncoderReranker
from .router import RetrievalRouter, RouterMatch

logger = structlog.get_logger(__name__)

DEFAULT_POLICY_PATH = Path("config/retrieval/reranking.yaml")
DEFAULT_COMPONENT_CONFIG_PATH = Path("config/retrieval/components.yaml")


@dataclass(slots=True)
class RetrievalResult:
    id: str
    text: str
    retrieval_score: float
    rerank_score: float | None
    highlights: Sequence[Mapping[str, object]]
    metadata: Mapping[str, object]
    granularity: str = "chunk"


class RetrievalService:
    """Coordinates fan-out to retrieval components, fusion and reranking."""

    def __init__(
        self,
        opensearch: OpenSearchClient,
        faiss: FAISSIndex | None = None,
        reranker: Callable[..., object] | None = None,
        *,
        vector_store: VectorStoreService | None = None,
        vector_namespace: str = "default",
        context_factory: Callable[[], SecurityContext] | None = None,
        fusion_service: FusionService | None = None,
        pipeline_settings: PipelineSettings | None = None,
        reranking_engine: RerankingEngine | None = None,
        reranking_settings: RerankingSettings | None = None,
        router: RetrievalRouter | None = None,
        namespace_map: Mapping[str, str] | None = None,
        rerank_policy: TenantRerankPolicy | None = None,
        model_registry: RerankerModelRegistry | None = None,
        hybrid_coordinator: HybridSearchCoordinator | None = None,
        hybrid_settings: HybridComponentSettings | None = None,
    ) -> None:
        self.opensearch = opensearch
        self.faiss = faiss
        self.vector_store = vector_store
        self.vector_namespace = vector_namespace
        self._context_factory = context_factory
        self.router = router or RetrievalRouter()
        self._namespace_map = dict(namespace_map or {})
        self._rerank_policy = rerank_policy or TenantRerankPolicy.from_file(DEFAULT_POLICY_PATH)
        self._model_registry = model_registry or RerankerModelRegistry()
        self._model_handles: dict[str, ModelHandle] = {}
        self._hybrid = hybrid_coordinator or self._build_hybrid_coordinator(hybrid_settings)

        fusion_cfg = reranking_settings.fusion if reranking_settings else None
        fusion_settings = FusionSettings(
            strategy=FusionStrategy(fusion_cfg.strategy) if fusion_cfg else FusionStrategy.RRF,
            rrf_k=fusion_cfg.rrf_k if fusion_cfg else 60,
            weights=fusion_cfg.weights if fusion_cfg else {},
            normalization=NormalizationStrategy(fusion_cfg.normalization)
            if fusion_cfg
            else NormalizationStrategy.MIN_MAX,
            deduplicate=fusion_cfg.deduplicate if fusion_cfg else True,
        )
        self._fusion = fusion_service or FusionService(fusion_settings)

        ttl = reranking_settings.cache_ttl if reranking_settings else 3600
        failure_threshold = reranking_settings.circuit_breaker_failures if reranking_settings else 5
        reset_timeout = reranking_settings.circuit_breaker_reset if reranking_settings else 30.0
        batch_size = reranking_settings.model.batch_size if reranking_settings else 64
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
        self._candidate_pool = max(100, pipeline_settings.retrieve_candidates)
        self.reranker = reranker or CrossEncoderReranker()
        configured_model_key: str | None = None
        if reranking_settings and reranking_settings.model.model:
            configured_model_key = reranking_settings.model.model
        self._default_model_key = self._resolve_model_key(configured_model_key)
        self._default_model_handle = self._ensure_model(self._default_model_key)
        configured_reranker = (
            reranking_settings.model.reranker_id
            if reranking_settings and reranking_settings.model.reranker_id
            else None
        )
        self._default_reranker = configured_reranker or self._default_model_handle.model.reranker_id

    def search(
        self,
        index: str,
        query: str,
        filters: Mapping[str, object] | None = None,
        k: int = 10,
        rerank: bool | None = None,
        embedding_kind: str | None = None,
        *,
        reranker_id: str | None = None,
        rerank_model: str | None = None,
        context: SecurityContext | None = None,
        explain: bool = False,
    ) -> list[RetrievalResult]:
        security_context = context or (
            self._context_factory()
            if self._context_factory
            else SecurityContext(subject="system", tenant_id="system", scopes={"*"})
        )
        namespace = self._resolve_namespace(embedding_kind)
        component_k = max(k, self._candidate_pool)
        component_results, component_errors, component_timings = self._execute_components(
            index=index,
            query=query,
            filters=filters or {},
            namespace=namespace,
            top_k=component_k,
            context=security_context,
        )
        candidate_lists = {
            component: self._materialise_documents(results, security_context, strategy=component)
            for component, results in component_results.items()
        }

        decision = self._rerank_policy.decide(security_context.tenant_id, query, rerank)
        model_handle, model_key, requested_model, model_fallback = self._resolve_model(rerank_model)
        reranker_key = reranker_id or model_handle.model.reranker_id or self._default_reranker
        metrics: dict[str, object]
        rerank_applied = decision.enabled
        try:
            documents, metrics = self._pipeline.execute(
                security_context,
                query,
                candidate_lists,
                reranker_id=reranker_key,
                top_k=k,
                rerank=decision.enabled,
                explain=explain,
            )
        except RerankingError as exc:
            logger.warning(
                "retrieval.rerank_failed",
                tenant=security_context.tenant_id,
                reranker=reranker_key,
                error=str(exc),
            )
            rerank_applied = False
            documents, metrics = self._pipeline.execute(
                security_context,
                query,
                candidate_lists,
                reranker_id=reranker_key,
                top_k=k,
                rerank=False,
                explain=explain,
            )
            metrics = dict(metrics)
            rerank_metrics = dict(metrics.get("reranking", {}))
            rerank_metrics.update(
                {
                    "error": exc.__class__.__name__,
                    "message": str(exc),
                    "fallback": "fusion",
                }
            )
            metrics["reranking"] = rerank_metrics
        else:
            metrics = dict(metrics)

        metrics.setdefault("components", {})
        metrics["components"]["errors"] = list(component_errors)
        if component_timings:
            metrics["components"]["timings_ms"] = dict(component_timings)
        rerank_metadata = dict(metrics.get("reranking", {}))
        rerank_metadata.update(decision.as_metadata())
        rerank_metadata.setdefault("requested", rerank)
        rerank_metadata["applied"] = rerank_applied
        model_meta = {
            "key": model_key,
            "model_id": model_handle.model.model_id,
            "version": model_handle.model.version,
            "provider": model_handle.model.provider,
            "requires_gpu": model_handle.model.requires_gpu,
        }
        if requested_model:
            rerank_metadata["requested_model"] = requested_model
        if model_fallback:
            rerank_metadata["fallback_model"] = model_key
            rerank_metadata.setdefault("warnings", []).append("model_fallback")
        rerank_metadata["model"] = model_meta
        rerank_metadata.setdefault("reranker_id", reranker_key)
        metrics["reranking"] = rerank_metadata

        results: list[RetrievalResult] = []
        for document in documents:
            metadata = dict(document.metadata)
            metadata.setdefault("component_scores", dict(document.strategy_scores))
            metadata.setdefault("components", {})
            metadata["components"].setdefault("errors", list(component_errors))
            if component_timings:
                metadata["components"].setdefault("timings_ms", dict(component_timings))
            if rerank_metadata:
                metadata.setdefault("reranking", rerank_metadata)
            result = RetrievalResult(
                id=document.doc_id,
                text=document.content,
                retrieval_score=float(metadata.get("retrieval_score", document.score)),
                rerank_score=document.score if rerank_applied else None,
                highlights=list(document.highlights),
                metadata=metadata,
                granularity=str(metadata.get("granularity", "chunk")),
            )
            if explain:
                metadata.setdefault("pipeline_metrics", metrics)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    def _execute_components(
        self,
        *,
        index: str,
        query: str,
        filters: Mapping[str, object],
        namespace: str,
        top_k: int,
        context: SecurityContext,
    ) -> tuple[
        dict[str, Sequence[Mapping[str, object]]],
        list[str],
        dict[str, float],
    ]:
        if self._hybrid:
            outcome = self._hybrid.search_sync(
                index=index,
                query=query,
                k=top_k,
                filters=filters,
                correlation_id=context.identity,
                context=context,
                cache_scope=context.tenant_id,
            )
            component_results = outcome.component_results
            errors = outcome.component_errors
            for component in component_results:
                component_results.setdefault(component, [])
            return component_results, errors, dict(outcome.timings_ms)

        tasks: dict[str, Callable[[], Sequence[Mapping[str, object]]]] = {
            "bm25": lambda: self.opensearch.search(
                index,
                query,
                strategy="bm25",
                filters=filters,
                size=top_k,
            ),
            "splade": lambda: self.opensearch.search(
                index,
                query,
                strategy="splade",
                filters=filters,
                size=top_k,
            ),
        }
        if self.faiss or self.vector_store:
            tasks["dense"] = lambda: self._dense_search(query, top_k, context)

        results: dict[str, Sequence[Mapping[str, object]]] = {}
        errors: list[str] = []
        timings: dict[str, float] = {}
        start_times: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_map = {}
            for name, func in tasks.items():
                start_times[name] = perf_counter()
                future_map[executor.submit(func)] = name
            for future in as_completed(future_map):
                component = future_map[future]
                try:
                    results[component] = list(future.result())
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "retrieval.component_failed",
                        component=component,
                        error=str(exc),
                    )
                    errors.append(f"{component}:{exc.__class__.__name__}")
                    results[component] = []
                finally:
                    started = start_times.get(component, perf_counter())
                    timings[component] = (perf_counter() - started) * 1000.0
        for component in tasks:
            results.setdefault(component, [])
        return results, errors, timings

    def _build_hybrid_coordinator(
        self, settings: HybridComponentSettings | None
    ) -> HybridSearchCoordinator:
        resolved_settings = settings
        if resolved_settings is None:
            try:
                resolved_settings = HybridComponentSettings.from_file(DEFAULT_COMPONENT_CONFIG_PATH)
            except FileNotFoundError:
                resolved_settings = HybridComponentSettings()
        components = {
            "bm25": self._bm25_component,
            "splade": self._splade_component,
            "dense": self._dense_component,
        }
        return HybridSearchCoordinator(components, settings=resolved_settings)

    async def _bm25_component(
        self,
        *,
        index: str,
        query: str,
        k: int,
        filters: Mapping[str, object],
        context: SecurityContext | None = None,
    ) -> Sequence[Mapping[str, object]]:
        return await asyncio.to_thread(
            self.opensearch.search,
            index,
            query,
            "bm25",
            filters,
            True,
            k,
        )

    async def _splade_component(
        self,
        *,
        index: str,
        query: str,
        k: int,
        filters: Mapping[str, object],
        context: SecurityContext | None = None,
    ) -> Sequence[Mapping[str, object]]:
        return await asyncio.to_thread(
            self.opensearch.search,
            index,
            query,
            "splade",
            filters,
            True,
            k,
        )

    async def _dense_component(
        self,
        *,
        index: str,
        query: str,
        k: int,
        filters: Mapping[str, object],
        context: SecurityContext | None = None,
    ) -> Sequence[Mapping[str, object]]:
        if context is None:
            raise ValueError("Security context required for dense retrieval")
        return await asyncio.to_thread(self._dense_search, query, k, context)

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
            score = float(result.get("_score", 0.0))
            metadata.setdefault("retrieval_score", score)
            tenant = str(metadata.get("tenant_id", context.tenant_id))
            text = str(metadata.get("text", ""))
            highlights = list(result.get("highlight", []))
            document = ScoredDocument(
                doc_id=doc_id,
                content=text,
                tenant_id=tenant,
                source=str(metadata.get("source", strategy)),
                strategy_scores={strategy: score},
                metadata=metadata,
                highlights=highlights,
                score=score,
            )
            documents.append(document)
        return documents

    def _dense_search(
        self, query: str, k: int, context: SecurityContext
    ) -> list[Mapping[str, object]]:
        matches = self._dense_strategy(self.vector_namespace, query, k, context)
        results: list[Mapping[str, object]] = []
        for match in matches:
            metadata = dict(match.metadata)
            metadata.setdefault("text", metadata.get("text", ""))
            results.append(
                {
                    "_id": match.id,
                    "_score": match.score,
                    "_source": metadata,
                    "highlight": metadata.get("highlights", []),
                }
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

    # ------------------------------------------------------------------
    def _resolve_model_key(self, identifier: str | None) -> str:
        if not identifier:
            return self._default_model_key
        try:
            return self._model_registry.resolve_key(identifier)
        except KeyError:
            logger.warning(
                "retrieval.rerank_model.unknown",
                requested=identifier,
            )
            return self._default_model_key

    def _ensure_model(self, key: str) -> ModelHandle:
        handle = self._model_handles.get(key)
        if handle is not None:
            return handle
        try:
            handle = self._model_registry.ensure(key)
        except ModelDownloadError as exc:
            logger.warning(
                "retrieval.rerank_model.ensure_failed",
                model=key,
                error=str(exc),
            )
            model = self._model_registry.get(key)
            cache_dir = getattr(self._model_registry, "cache_dir", None)
            cache_root = cache_dir if cache_dir else DEFAULT_POLICY_PATH.parent
            handle = ModelHandle(model=model, path=model.cache_path(cache_root))
        self._model_handles[key] = handle
        return handle

    def _resolve_model(self, identifier: str | None) -> tuple[ModelHandle, str, str | None, bool]:
        key = self._resolve_model_key(identifier)
        fallback = (
            bool(identifier)
            and key == self._default_model_key
            and (identifier != self._default_model_key)
        )
        handle = self._ensure_model(key)
        if handle is not self._default_model_handle and key not in self._model_handles:
            self._model_handles[key] = handle
        if identifier and key != identifier:
            fallback = True
        return handle, key, identifier, fallback

    def _resolve_namespace(self, embedding_kind: str | None) -> str:
        if embedding_kind and embedding_kind in self._namespace_map:
            return self._namespace_map[embedding_kind]
        return self.vector_namespace


__all__ = ["RetrievalResult", "RetrievalService"]
