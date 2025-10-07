"""Query pipeline orchestration including retrieval, fusion, and reranking."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import structlog
from Medical_KG_rev.observability.metrics import (
    observe_query_latency,
    observe_query_stage_latency,
    record_query_operation,
)
from Medical_KG_rev.utils.errors import ProblemDetail

if TYPE_CHECKING:  # pragma: no cover - typing only
    from Medical_KG_rev.services.retrieval.router import RetrievalRouter, RetrievalStrategy

from .pipeline import (
    ParallelExecutor,
    PipelineContext,
    PipelineExecutor,
    PipelineStage,
    StageFailure,
)
from .profiles import ProfileDetector, apply_profile_overrides
from .resilience import CircuitBreaker

logger = structlog.get_logger(__name__)


ResultList = Sequence[Mapping[str, Any]]


@dataclass(slots=True)
class ConfigurableStage(PipelineStage):
    """Base class for stages that honour per-request configuration overrides."""

    name: str
    timeout_ms: int | None = None
    base_options: Mapping[str, Any] = field(default_factory=dict)
    config_key: str | None = None

    def resolve_options(self, context: PipelineContext) -> dict[str, Any]:
        options = dict(self.base_options)
        config = context.data.get("config")
        key = self.config_key or self.name
        if isinstance(config, Mapping):
            overrides = config.get(key)
            if isinstance(overrides, Mapping):
                options.update(overrides)
        return options

    def effective_timeout(self, options: Mapping[str, Any]) -> int | None:
        override = options.get("timeout_ms")
        if isinstance(override, (int, float)):
            return int(override)
        return self.timeout_ms


@dataclass(slots=True)
class StrategySpec:
    """Declarative strategy binding used by the retrieval stage."""

    name: str
    runner: Callable[[PipelineContext, Mapping[str, Any]], ResultList]
    timeout_ms: int | None = None

    def execute(self, context: PipelineContext, options: Mapping[str, Any]) -> ResultList:
        return self.runner(context, options)

    @classmethod
    def from_router(
        cls,
        router: "RetrievalRouter",
        strategy: "RetrievalStrategy",
        *,
        timeout_ms: int | None = None,
    ) -> StrategySpec:
        """Adapt a retrieval router strategy into a pipeline-aware spec."""

        from Medical_KG_rev.services.retrieval.router import RoutingRequest

        def _run(context: PipelineContext, options: Mapping[str, Any]) -> list[dict[str, Any]]:
            top_k = int(options.get("top_k", context.data.get("top_k", 10)))
            request = RoutingRequest(
                query=str(context.data.get("query", "")),
                top_k=top_k,
                filters=context.data.get("filters"),
                namespace=options.get("namespace"),
                context=context.data.get("metadata"),
            )
            matches = router.execute(request, [strategy])
            results: list[dict[str, Any]] = []
            for match in matches:
                metadata = dict(getattr(match, "metadata", {}) or {})
                document = metadata.get("document")
                if isinstance(document, Mapping):
                    document_payload = dict(document)
                else:
                    document_payload = {
                        "id": match.id,
                        "title": metadata.get("title", match.id),
                        "summary": metadata.get("summary"),
                        "source": metadata.get("source", strategy.name),
                    }
                document_payload.setdefault("id", match.id)
                document_payload.setdefault("title", metadata.get("title", document_payload["id"]))
                document_payload.setdefault("source", metadata.get("source", strategy.name))
                if metadata:
                    document_payload.setdefault("metadata", metadata)
                results.append(
                    {
                        "id": match.id,
                        "score": float(getattr(match, "score", 0.0)),
                        "document": document_payload,
                    }
                )
            return results

        return cls(name=strategy.name, runner=_run, timeout_ms=timeout_ms)


@dataclass(slots=True)
class RetrievalOrchestrator(ConfigurableStage):
    """Fan out to registered retrieval strategies and collect candidates."""

    strategies: Mapping[str, StrategySpec] = field(default_factory=dict)
    fanout: ParallelExecutor = field(default_factory=ParallelExecutor)

    def execute(self, context: PipelineContext) -> PipelineContext:
        options = self.resolve_options(context)
        selections = self._selected_strategies(options)
        if not selections:
            raise StageFailure(
                "No retrieval strategies configured",
                status=500,
                stage=self.name,
                error_type="configuration",
            )
        stage_timeout = self.effective_timeout(options)
        overrides_map = {definition.name: overrides for definition, overrides in selections}
        tasks = {
            definition.name: (lambda d=definition, o=overrides_map[definition.name]: self._run_strategy(context, d, o))
            for definition, _ in selections
        }
        results = self.fanout.run(
            tasks,
            timeout_ms=stage_timeout,
            correlation_id=context.correlation_id,
        )
        aggregated: list[dict[str, Any]] = []
        for name, result in results.items():
            if result.timed_out:
                problem = ProblemDetail(
                    title="Strategy timeout",
                    status=504,
                    detail=f"Retrieval strategy '{name}' timed out",
                )
                context.errors.append(problem)
                context.partial = True
                context.degraded = True
                continue
            if result.error:
                context.errors.append(result.error)
                context.partial = True
                context.degraded = True
                continue
            aggregated.append(
                {
                    "strategy": name,
                    "results": list(result.value or []),
                    "duration_ms": round(result.duration * 1000, 3),
                    "options": overrides_map.get(name, {}),
                }
            )
        if not aggregated:
            raise StageFailure(
                "All retrieval strategies failed",
                status=504,
                stage=self.name,
                error_type="timeout",
                retriable=True,
            )
        durations = [entry.get("duration_ms", 0.0) for entry in aggregated]
        if durations:
            observe_query_stage_latency(self.name, max(durations) / 1000.0)
        context.data["retrieval_candidates"] = aggregated
        return context

    def _selected_strategies(
        self, options: Mapping[str, Any]
    ) -> list[tuple[StrategySpec, Mapping[str, Any]]]:
        configured = options.get("strategies")
        selections: list[tuple[str, Mapping[str, Any]]] = []
        if isinstance(configured, Mapping):
            for name, value in configured.items():
                if name in self.strategies:
                    if isinstance(value, Mapping):
                        selections.append((name, dict(value)))
                    else:
                        selections.append((name, {}))
        elif isinstance(configured, Sequence):
            for name in configured:
                if name in self.strategies:
                    selections.append((name, {}))
        else:
            selections = [(name, {}) for name in self.strategies]
        defaults = options.get("strategy_options")
        if not isinstance(defaults, Mapping):
            defaults = {}
        timeouts = options.get("timeouts") or options.get("strategy_timeouts")
        if not isinstance(timeouts, Mapping):
            timeouts = {}
        global_timeout = options.get("strategy_timeout_ms")
        resolved: list[tuple[StrategySpec, Mapping[str, Any]]] = []
        for name, overrides in selections:
            definition = self.strategies.get(name)
            if not definition:
                continue
            strategy_options: dict[str, Any] = {}
            if isinstance(defaults.get(name), Mapping):
                strategy_options.update(defaults[name])
            strategy_options.update(overrides)
            if name in timeouts and "timeout_ms" not in strategy_options:
                strategy_options["timeout_ms"] = timeouts[name]
            if "timeout_ms" not in strategy_options and definition.timeout_ms is not None:
                strategy_options["timeout_ms"] = definition.timeout_ms
            if global_timeout and "timeout_ms" not in strategy_options:
                strategy_options["timeout_ms"] = int(global_timeout)
            resolved.append((definition, strategy_options))
        return resolved

    def _run_strategy(
        self,
        context: PipelineContext,
        definition: StrategySpec,
        options: Mapping[str, Any],
    ) -> ResultList:
        started = time.perf_counter()
        results = definition.execute(context, options)
        duration = time.perf_counter() - started
        context.stage_timings[f"{self.name}.{definition.name}"] = duration
        timeout_ms = options.get("timeout_ms")
        if isinstance(timeout_ms, (int, float)) and duration * 1000 > float(timeout_ms):
            raise StageFailure(
                f"Strategy '{definition.name}' exceeded timeout",
                status=504,
                stage=self.name,
                error_type="timeout",
                retriable=True,
            )
        return results


@dataclass(slots=True)
class FusionOrchestrator(ConfigurableStage):
    """Merge candidate lists into a fused ranking."""

    def execute(self, context: PipelineContext) -> PipelineContext:
        candidates: Sequence[Mapping[str, Any]] = context.data.get("retrieval_candidates", [])
        if not candidates:
            raise StageFailure(
                "No candidates available for fusion",
                status=400,
                stage=self.name,
                error_type="validation",
            )
        options = self.resolve_options(context)
        method = str(options.get("method", "rrf")).lower()
        weights = options.get("weights") or options.get("strategy_weights") or {}
        normalization = options.get("normalization")
        fused = _fuse_candidates(
            candidates,
            method=method,
            weights=weights,
            normalization=normalization,
        )
        context.data["fusion_results"] = fused
        return context


def _fuse_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    method: str,
    weights: Mapping[str, float] | None,
    normalization: str | None,
) -> list[dict[str, Any]]:
    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    metadata: dict[str, dict[str, Any]] = {}
    for entry in candidates:
        strategy = str(entry.get("strategy"))
        results = entry.get("results") or []
        weight = float(weights.get(strategy, 1.0))
        for rank, item in enumerate(results, start=1):
            doc_id = str(item.get("id"))
            score = float(item.get("score", 0.0))
            if method == "weighted":
                contribution = score * weight
            else:
                contribution = weight / (rank + 60)
            scores[doc_id] += contribution
            metadata.setdefault(doc_id, {}).setdefault("strategies", {})[strategy] = {
                "score": score,
                "rank": rank,
            }
            metadata[doc_id].setdefault("document", item)
    if normalization == "max" and scores:
        max_score = max(scores.values()) or 1.0
        for key in list(scores):
            scores[key] /= max_score
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused: list[dict[str, Any]] = []
    for doc_id, score in ranked:
        document = metadata[doc_id]["document"]
        fused.append(
            {
                "id": doc_id,
                "score": score,
                "document": document,
                "strategies": metadata[doc_id].get("strategies", {}),
            }
        )
    return fused


@dataclass(slots=True)
class RerankCache:
    ttl_seconds: float = 300.0
    _store: dict[str, tuple[float, list[dict[str, Any]]]] = field(default_factory=dict)

    def get(self, key: str) -> list[dict[str, Any]] | None:
        item = self._store.get(key)
        if not item:
            return None
        timestamp, value = item
        if time.time() - timestamp > self.ttl_seconds:
            self._store.pop(key, None)
            return None
        return value

    def put(self, key: str, value: list[dict[str, Any]]) -> None:
        self._store[key] = (time.time(), value)


@dataclass(slots=True)
class RerankOrchestrator(ConfigurableStage):
    """Apply reranking to fused candidates with optional caching."""

    rerank: Callable[[PipelineContext, Sequence[dict[str, Any]], Mapping[str, Any]], Sequence[dict[str, Any]]] = field(default=None)
    cache: RerankCache = field(default_factory=RerankCache)
    circuit_breaker: CircuitBreaker | None = field(default=None)

    def execute(self, context: PipelineContext) -> PipelineContext:
        candidates: list[dict[str, Any]] = context.data.get("fusion_results", [])
        if not candidates:
            raise StageFailure(
                "No fusion results available for reranking",
                status=400,
                stage=self.name,
                error_type="validation",
            )
        options = self.resolve_options(context)
        timeout_ms = self.effective_timeout(options)
        if "cache_ttl_seconds" in options:
            try:
                self.cache.ttl_seconds = float(options["cache_ttl_seconds"])
            except (TypeError, ValueError):
                logger.debug("rerank.cache.invalid_ttl", value=options.get("cache_ttl_seconds"))
        if not bool(options.get("enabled", True)):
            context.data["reranked_results"] = list(candidates)
            return context
        rerank_candidates = int(options.get("rerank_candidates", min(len(candidates), 100)))
        allow_overflow = bool(options.get("allow_overflow", False))
        top_candidates = candidates if allow_overflow else candidates[:rerank_candidates]
        cache_key = _rerank_cache_key(context, top_candidates, options)
        cached = self.cache.get(cache_key)
        if cached is not None:
            context.data["reranked_results"] = cached
            return context
        started = time.perf_counter()
        try:
            if self.circuit_breaker:
                with self.circuit_breaker.guard(self.name):
                    results = self.rerank(context, top_candidates, options)
            else:
                results = self.rerank(context, top_candidates, options)
        except StageFailure:
            raise
        except Exception as exc:  # pragma: no cover - guardrail
            raise StageFailure(
                "Reranking failed",
                status=500,
                stage=self.name,
                detail=str(exc),
                retriable=True,
            ) from exc
        duration = time.perf_counter() - started
        context.stage_timings[f"{self.name}:duration"] = duration
        if timeout_ms and duration * 1000 > timeout_ms:
            raise StageFailure(
                "Reranking exceeded timeout",
                status=504,
                stage=self.name,
                error_type="timeout",
                retriable=True,
            )
        self.cache.put(cache_key, list(results))
        context.data["reranked_results"] = list(results)
        observe_query_stage_latency(self.name, duration)
        return context


def _rerank_cache_key(
    context: PipelineContext,
    candidates: Sequence[dict[str, Any]],
    options: Mapping[str, Any],
) -> str:
    doc_ids = ",".join(str(candidate.get("id")) for candidate in candidates)
    query = str(context.data.get("query", ""))
    tenant = context.tenant_id
    reranker_id = str(options.get("reranker_id", "default"))
    return f"{tenant}:{reranker_id}:{query}:{hash(doc_ids)}"


@dataclass(slots=True)
class FinalSelectorOrchestrator(ConfigurableStage):
    """Prepare the final response payload."""

    def execute(self, context: PipelineContext) -> PipelineContext:
        options = self.resolve_options(context)
        results: Sequence[dict[str, Any]] = (
            context.data.get("reranked_results")
            or context.data.get("fusion_results")
            or []
        )
        if not results:
            raise StageFailure(
                "No results available for final selection",
                status=400,
                stage=self.name,
                error_type="validation",
            )
        top_k = int(options.get("top_k", context.data.get("top_k", 10)))
        explain = bool(options.get("explain", context.data.get("explain", False)))
        include_metadata = bool(options.get("include_metadata", True))
        final_results: list[dict[str, Any]] = []
        for rank, item in enumerate(results[:top_k], start=1):
            document = dict(item.get("document", {}))
            payload: dict[str, Any] = {
                "id": item.get("id") or document.get("id"),
                "rank": rank,
                "score": item.get("score"),
                "document": document if include_metadata else {k: document.get(k) for k in ("title", "summary", "source")},
            }
            if explain:
                payload["strategies"] = item.get("strategies", {})
            final_results.append(payload)
        context.data["results"] = final_results
        return context


class QueryPipelineExecutor:
    """Executes the retrieval pipeline with total timeout enforcement."""

    def __init__(
        self,
        stages: Sequence[PipelineStage],
        *,
        operation: str = "retrieve",
        pipeline_name: str = "query",
        pipeline_version: str | None = None,
        total_timeout_ms: int = 100,
        profile_detector: ProfileDetector | None = None,
    ) -> None:
        self.executor = PipelineExecutor(stages, operation=operation, pipeline=pipeline_name)
        self.total_timeout = total_timeout_ms / 1000
        self.pipeline_name = pipeline_name
        self.pipeline_version = pipeline_version
        self.profile_detector = profile_detector

    def run(self, context: PipelineContext) -> PipelineContext:
        context.data.setdefault("config", {})
        if self.profile_detector:
            context = self._apply_profile(context)
            if context.errors and context.partial:
                return context
        started = time.perf_counter()
        record_query_operation(self.pipeline_name, context.tenant_id, "started")
        try:
            result = self.executor.run(context)
        except StageFailure as failure:
            context.errors.append(failure.problem)
            context.partial = True
            context.degraded = True
            context.data.setdefault("degradation_events", []).append(
                {"stage": failure.stage or "pipeline", "reason": failure.problem.title}
            )
            record_query_operation(self.pipeline_name, context.tenant_id, "failed")
            return context
        duration = time.perf_counter() - started
        if duration > self.total_timeout:
            timeout_problem = ProblemDetail(
                title="Query pipeline timeout",
                status=504,
                detail=f"Pipeline exceeded {self.total_timeout * 1000:.0f}ms",
            )
            context.errors.append(timeout_problem)
            context.partial = True
            context.degraded = True
            context.data.setdefault("degradation_events", []).append(
                {"stage": "pipeline", "reason": timeout_problem.title}
            )
            record_query_operation(self.pipeline_name, context.tenant_id, "degraded")
        else:
            record_query_operation(self.pipeline_name, context.tenant_id, "success")
        context.stage_timings.setdefault("total", duration)
        observe_query_latency(self.pipeline_name, duration)
        if self.pipeline_version and not context.pipeline_version:
            context.pipeline_version = self.pipeline_version
        context.data["degraded"] = context.degraded or context.partial
        context.data.setdefault("pipeline_version", context.pipeline_version or self.pipeline_version)
        return result

    def _apply_profile(self, context: PipelineContext) -> PipelineContext:
        metadata = {}
        meta_payload = context.data.get("metadata")
        if isinstance(meta_payload, Mapping):
            metadata.update({k: v for k, v in meta_payload.items() if isinstance(k, str)})
        explicit = context.data.get("profile")
        try:
            profile = self.profile_detector.detect(
                explicit=str(explicit) if explicit else None,
                metadata=metadata,
            )
        except KeyError as exc:
            problem = ProblemDetail(
                title="Unknown profile",
                status=400,
                detail=str(exc),
            )
            context.errors.append(problem)
            context.partial = True
            return context
        context.data = apply_profile_overrides(context.data, profile)
        if not context.pipeline_version:
            if self.pipeline_version and ":" in self.pipeline_version:
                context.pipeline_version = self.pipeline_version
            elif self.pipeline_version:
                context.pipeline_version = f"{profile.query}:{self.pipeline_version}".rstrip(":")
            else:
                context.pipeline_version = profile.query
        logger.info(
            "orchestration.profile.applied",
            operation=context.operation,
            profile=profile.name,
            pipeline=context.pipeline_version,
        )
        return context


__all__ = [
    "ConfigurableStage",
    "FinalSelectorOrchestrator",
    "FusionOrchestrator",
    "QueryPipelineExecutor",
    "RerankCache",
    "RerankOrchestrator",
    "RetrievalOrchestrator",
    "StrategySpec",
]
