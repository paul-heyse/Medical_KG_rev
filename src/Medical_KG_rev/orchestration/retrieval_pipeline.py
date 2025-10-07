"""Query pipeline orchestration including retrieval, fusion, and reranking."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import structlog

from Medical_KG_rev.utils.errors import ProblemDetail

from .pipeline import ParallelExecutor, PipelineContext, PipelineExecutor, PipelineStage, StageFailure
from .profiles import ProfileDetector, apply_profile_overrides


logger = structlog.get_logger(__name__)


ResultList = Sequence[Mapping[str, Any]]


@dataclass(slots=True)
class RetrievalStrategy:
    name: str
    executor: Callable[[PipelineContext], ResultList]
    timeout_ms: int | None = None


@dataclass(slots=True)
class RetrievalOrchestrator(PipelineStage):
    strategies: Sequence[RetrievalStrategy]
    fanout: ParallelExecutor
    timeout_ms: int | None = 50
    name: str = "retrieval"

    def execute(self, context: PipelineContext) -> PipelineContext:
        if not self.strategies:
            raise StageFailure(
                "No retrieval strategies configured",
                status=500,
                stage=self.name,
                error_type="configuration",
            )
        retrieval_config = _stage_config(context.data, "retrieval")
        enabled = retrieval_config.get("strategies")
        if enabled:
            selected = [s for s in self.strategies if s.name in enabled]
        else:
            selected = list(self.strategies)
        if not selected:
            raise StageFailure(
                "No retrieval strategies enabled",
                status=500,
                stage=self.name,
                error_type="configuration",
            )
        timeout_override = retrieval_config.get("timeout_ms")
        if isinstance(timeout_override, (int, float)):
            self_timeout = int(timeout_override)
        else:
            self_timeout = self.timeout_ms
        tasks = {
            strategy.name: _strategy_callable(
                context,
                _apply_strategy_overrides(strategy, retrieval_config),
            )
            for strategy in selected
        }
        per_strategy_timeouts = [
            strategy.timeout_ms for strategy in selected if strategy.timeout_ms
        ]
        timeout_ms = self_timeout
        if per_strategy_timeouts:
            timeout_ms = min(per_strategy_timeouts)
        results = self.fanout.run(
            tasks,
            timeout_ms=timeout_ms,
            correlation_id=context.correlation_id,
        )
        aggregated: list[dict[str, Any]] = []
        for name, result in results.items():
            if result.timed_out:
                context.errors.append(
                    ProblemDetail(
                        title="Strategy timeout",
                        status=504,
                        detail=f"Retrieval strategy '{name}' timed out",
                    )
                )
                context.partial = True
                continue
            if result.error:
                context.errors.append(result.error)
                context.partial = True
                continue
            aggregated.append(
                {
                    "strategy": name,
                    "results": list(result.value or []),
                    "duration_ms": round(result.duration * 1000, 3),
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
        context.data["retrieval_candidates"] = aggregated
        return context


def _strategy_callable(
    context: PipelineContext, strategy: RetrievalStrategy
) -> Callable[[], ResultList]:
    def runner() -> ResultList:
        started = time.perf_counter()
        value = strategy.executor(context)
        elapsed = (time.perf_counter() - started) * 1000
        context.stage_timings[f"{strategy.name}:retrieve"] = elapsed / 1000
        if strategy.timeout_ms and elapsed > strategy.timeout_ms:
            raise StageFailure(
                f"Strategy '{strategy.name}' exceeded timeout",
                status=504,
                stage="retrieval",
                error_type="timeout",
                retriable=True,
            )
        return value

    return runner


@dataclass(slots=True)
class FusionOrchestrator(PipelineStage):
    name: str = "fusion"
    timeout_ms: int | None = 5

    def execute(self, context: PipelineContext) -> PipelineContext:
        candidates: Sequence[Mapping[str, Any]] = context.data.get("retrieval_candidates", [])
        if not candidates:
            raise StageFailure(
                "No candidates available for fusion",
                status=400,
                stage=self.name,
                error_type="validation",
            )
        fusion_config = _stage_config(context.data, "fusion")
        method = str(fusion_config.get("method", "rrf")).lower()
        weights = fusion_config.get("weights") or fusion_config.get("strategy_weights", {})
        fused = _fuse_candidates(candidates, method=method, weights=weights)
        context.data["fusion_results"] = fused
        return context


def _fuse_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    method: str,
    weights: Mapping[str, float] | None,
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
            else:  # reciprocal rank fusion
                contribution = weight * (1.0 / (rank + 60))
            scores[doc_id] += contribution
            metadata.setdefault(doc_id, {}).setdefault("strategies", {})[strategy] = {
                "score": score,
                "rank": rank,
            }
            metadata[doc_id].setdefault("document", item)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused: list[dict[str, Any]] = []
    for doc_id, score in ranked:
        document = metadata[doc_id]["document"]
        fused.append(
            {
                "id": doc_id,
                "score": score,
                "document": document,
                "strategies": metadata[doc_id]["strategies"],
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
class RerankOrchestrator(PipelineStage):
    rerank: Callable[[PipelineContext, Sequence[dict[str, Any]], int], list[dict[str, Any]]]
    cache: RerankCache = field(default_factory=RerankCache)
    name: str = "rerank"
    timeout_ms: int | None = 50

    def execute(self, context: PipelineContext) -> PipelineContext:
        candidates: list[dict[str, Any]] = context.data.get("fusion_results", [])
        if not candidates:
            raise StageFailure(
                "No fusion results available for reranking",
                status=400,
                stage=self.name,
                error_type="validation",
            )
        rerank_config = _stage_config(context.data, "rerank")
        if "cache_ttl_seconds" in rerank_config:
            try:
                self.cache.ttl_seconds = float(rerank_config["cache_ttl_seconds"])
            except (TypeError, ValueError):
                pass
        timeout_override = rerank_config.get("timeout_ms")
        timeout_ms = self.timeout_ms
        if isinstance(timeout_override, (int, float)):
            timeout_ms = int(timeout_override)
        rerank_candidates = int(
            rerank_config.get("rerank_candidates", min(len(candidates), 100))
        )
        top_candidates = candidates[:rerank_candidates]
        cache_key = _rerank_cache_key(context, top_candidates)
        cached = self.cache.get(cache_key)
        if cached is not None:
            context.data["reranked_results"] = cached
            return context
        started = time.perf_counter()
        results = self.rerank(context, top_candidates, rerank_candidates)
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
        self.cache.put(cache_key, results)
        context.data["reranked_results"] = results
        return context


def _rerank_cache_key(context: PipelineContext, candidates: Sequence[dict[str, Any]]) -> str:
    doc_ids = ",".join(str(candidate.get("id")) for candidate in candidates)
    query = str(context.data.get("query", ""))
    tenant = context.tenant_id
    return f"{tenant}:{query}:{hash(doc_ids)}"


def _stage_config(data: Mapping[str, Any], stage: str) -> dict[str, Any]:
    config = data.get("config")
    if isinstance(config, Mapping):
        stage_config = config.get(stage)
        if isinstance(stage_config, Mapping):
            return dict(stage_config)
    return {}


def _apply_strategy_overrides(strategy: RetrievalStrategy, config: Mapping[str, Any]) -> RetrievalStrategy:
    timeouts = config.get("strategy_timeouts") or config.get("timeouts")
    timeout_ms = strategy.timeout_ms
    if isinstance(timeouts, Mapping) and strategy.name in timeouts:
        override = timeouts[strategy.name]
        if isinstance(override, (int, float)):
            timeout_ms = int(override)
    return RetrievalStrategy(name=strategy.name, executor=strategy.executor, timeout_ms=timeout_ms)


@dataclass(slots=True)
class FinalSelectorOrchestrator(PipelineStage):
    name: str = "final"
    timeout_ms: int | None = 5

    def execute(self, context: PipelineContext) -> PipelineContext:
        results: Sequence[dict[str, Any]] = (
            context.data.get("reranked_results")
            or context.data.get("fusion_results")
            or []
        )
        final_config = _stage_config(context.data, "final")
        top_k = int(final_config.get("top_k", 10))
        explain = bool(final_config.get("explain", context.data.get("explain", False)))
        final_results = []
        for rank, item in enumerate(results[:top_k], start=1):
            document = dict(item.get("document", {}))
            payload = {
                "id": item.get("id"),
                "rank": rank,
                "score": item.get("score"),
                "document": document,
            }
            if explain:
                payload["strategies"] = item.get("strategies", {})
            final_results.append(payload)
        context.data["results"] = final_results
        context.data["pipeline_version"] = context.pipeline_version
        return context


class QueryPipelineExecutor:
    """Executes the retrieval pipeline with total timeout enforcement."""

    def __init__(
        self,
        stages: Sequence[PipelineStage],
        *,
        operation: str = "retrieve",
        pipeline_name: str = "query",
        total_timeout_ms: int = 100,
        profile_detector: ProfileDetector | None = None,
    ) -> None:
        self.executor = PipelineExecutor(stages, operation=operation, pipeline=pipeline_name)
        self.total_timeout = total_timeout_ms / 1000
        self.pipeline_name = pipeline_name
        self.profile_detector = profile_detector

    def run(self, context: PipelineContext) -> PipelineContext:
        if self.profile_detector:
            context = self._apply_profile(context)
            if context.errors and context.partial:
                return context
        started = time.perf_counter()
        try:
            result = self.executor.run(context)
        except StageFailure as failure:
            context.errors.append(failure.problem)
            context.partial = True
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
            pipeline = profile.query_definition(self.profile_detector.manager.config)
            context.pipeline_version = (
                f"{pipeline.name}:{self.profile_detector.manager.config.version}"
            )
        logger.info(
            "orchestration.profile.applied",
            operation=context.operation,
            profile=profile.name,
            pipeline=context.pipeline_version,
        )
        return context


__all__ = [
    "FinalSelectorOrchestrator",
    "FusionOrchestrator",
    "QueryPipelineExecutor",
    "RerankCache",
    "RerankOrchestrator",
    "RetrievalOrchestrator",
    "RetrievalStrategy",
]
