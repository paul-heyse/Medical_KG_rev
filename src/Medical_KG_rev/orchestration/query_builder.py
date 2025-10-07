"""Helpers for constructing retrieval pipelines from configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .config_manager import PipelineConfigManager
from .pipeline import (
    ParallelExecutor,
    PipelineContext,
    PipelineDefinition,
    PipelineStage,
    StageConfig,
)
from .profiles import PipelineProfile, ProfileManager
from .resilience import CircuitBreaker
from .retrieval_pipeline import (
    FinalSelectorOrchestrator,
    FusionOrchestrator,
    QueryPipelineExecutor,
    RerankCache,
    RerankOrchestrator,
    RetrievalOrchestrator,
    StrategySpec,
)


def _default_cache_factory() -> RerankCache:
    return RerankCache()


Runner = Callable[[PipelineContext, Sequence[dict[str, Any]], Mapping[str, Any]], Sequence[dict[str, Any]]]


@dataclass
class QueryPipelineBuilder:
    """Materialises query pipeline executors for configured profiles."""

    config_manager: PipelineConfigManager
    profile_manager: ProfileManager
    parallel_executor: ParallelExecutor
    strategies: Mapping[str, StrategySpec]
    rerank_runner: Runner
    circuit_breaker: CircuitBreaker | None = None
    rerank_cache_factory: Callable[[], RerankCache] = _default_cache_factory
    total_timeout_ms: int = 100
    _executors: dict[str, QueryPipelineExecutor] = field(default_factory=dict, init=False)

    def invalidate(self) -> None:
        """Clear cached executors so the next request rebuilds them."""

        self._executors.clear()

    def executor_for_profile(self, profile: PipelineProfile) -> QueryPipelineExecutor:
        """Return a cached executor for the supplied profile."""

        version = self.config_manager.config.version
        cache_key = f"{profile.query}:{version}"
        if cache_key in self._executors:
            return self._executors[cache_key]
        definition = profile.query_definition(self.config_manager.config)
        stages = self._build_stages(definition, profile)
        executor = QueryPipelineExecutor(
            stages,
            pipeline_name=definition.name,
            pipeline_version=f"{definition.name}:{version}",
            total_timeout_ms=self.total_timeout_ms,
        )
        self._executors[cache_key] = executor
        return executor

    def executor_for_profile_name(self, profile_name: str) -> QueryPipelineExecutor:
        """Resolve a profile by name and return its executor."""

        profile = self.profile_manager.get(profile_name)
        return self.executor_for_profile(profile)

    def _build_stages(
        self, definition: PipelineDefinition, profile: PipelineProfile
    ) -> list[PipelineStage]:
        stages: list[PipelineStage] = []
        for stage_config in definition.stages:
            base_options = dict(stage_config.options)
            overrides = profile.overrides.get(stage_config.name)
            if isinstance(overrides, Mapping):
                base_options.update(overrides)
            stage = self._build_stage(stage_config, base_options)
            stages.append(stage)
        return stages

    def _build_stage(self, stage: StageConfig, options: Mapping[str, Any]):
        if stage.kind == "retrieval":
            return RetrievalOrchestrator(
                name=stage.name,
                timeout_ms=stage.timeout_ms,
                base_options=options,
                strategies=self.strategies,
                fanout=self.parallel_executor,
            )
        if stage.kind == "fusion":
            return FusionOrchestrator(
                name=stage.name,
                timeout_ms=stage.timeout_ms,
                base_options=options,
            )
        if stage.kind == "rerank":
            return RerankOrchestrator(
                name=stage.name,
                timeout_ms=stage.timeout_ms,
                base_options=options,
                rerank=self.rerank_runner,
                cache=self.rerank_cache_factory(),
                circuit_breaker=self.circuit_breaker,
            )
        if stage.kind == "final":
            return FinalSelectorOrchestrator(
                name=stage.name,
                timeout_ms=stage.timeout_ms,
                base_options=options,
            )
        raise ValueError(f"Unsupported stage kind '{stage.kind}' in query pipeline")


__all__ = ["QueryPipelineBuilder", "Runner"]
