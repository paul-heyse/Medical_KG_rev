"""Configuration models and loaders for Dagster-based orchestration."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from Medical_KG_rev.observability.metrics import (
    record_resilience_circuit_state,
    record_resilience_rate_limit_wait,
    record_resilience_retry,
)
from Medical_KG_rev.utils.logging import get_logger

logger = get_logger(__name__)


class BackoffStrategy(str, Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    NONE = "none"


class GateCondition(BaseModel):
    """Predicate evaluated against Job Ledger entries to resume a pipeline."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    equals: Any
    timeout_seconds: int | None = Field(default=None, ge=1, le=3600)
    poll_interval_seconds: float = Field(default=5.0, ge=0.5, le=60.0)


class GateDefinition(BaseModel):
    """Declarative definition for a pipeline gate."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    condition: GateCondition
    resume_stage: str = Field(pattern=r"^[A-Za-z0-9_-]+$")


class StageDefinition(BaseModel):
    """Declarative stage specification for topology YAML files."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    stage_type: str = Field(alias="type", pattern=r"^[A-Za-z0-9_-]+$")
    policy: str | None = Field(default=None, alias="policy")
    depends_on: list[str] = Field(default_factory=list, alias="depends_on")
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("depends_on")
    @classmethod
    def _unique_dependencies(cls, value: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in value:
            if item in seen:
                raise ValueError(f"duplicate dependency '{item}' declared for stage")
            seen.add(item)
            result.append(item)
        return result


class PipelineMetadata(BaseModel):
    """Optional metadata about the pipeline."""

    owner: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)


class PipelineTopologyConfig(BaseModel):
    """Complete topology definition for a pipeline."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    version: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}(-[A-Za-z0-9]+)?$")
    applicable_sources: list[str] = Field(default_factory=list)
    stages: list[StageDefinition]
    gates: list[GateDefinition] = Field(default_factory=list)
    metadata: PipelineMetadata | None = None

    @model_validator(mode="after")
    def _validate_dependencies(self) -> PipelineTopologyConfig:
        stage_names = [stage.name for stage in self.stages]
        if len(stage_names) != len(set(stage_names)):
            duplicates = {name for name in stage_names if stage_names.count(name) > 1}
            raise ValueError(f"duplicate stage names detected: {sorted(duplicates)}")

        stage_set = set(stage_names)
        for stage in self.stages:
            missing = [dep for dep in stage.depends_on if dep not in stage_set]
            if missing:
                raise ValueError(
                    f"stage '{stage.name}' declares unknown dependencies: {', '.join(sorted(missing))}"
                )

        order = _topological_sort({stage.name: stage.depends_on for stage in self.stages})
        if order is None:
            raise ValueError("cycle detected in pipeline dependencies")

        gate_stage_set = {stage.name for stage in self.stages}
        for gate in self.gates:
            if gate.resume_stage not in gate_stage_set:
                raise ValueError(
                    f"gate '{gate.name}' references unknown resume_stage '{gate.resume_stage}'"
                )
        return self


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = Field(ge=3, le=10)
    recovery_timeout: float = Field(ge=1.0, le=600.0)
    expected_exception: str | None = None


class RateLimitConfig(BaseModel):
    rate_limit_per_second: float = Field(ge=0.1, le=100.0)


class BackoffConfig(BaseModel):
    strategy: BackoffStrategy = Field(default=BackoffStrategy.EXPONENTIAL)
    initial: float = Field(default=0.5, ge=0.0, le=60.0)
    maximum: float = Field(default=30.0, ge=0.0, le=600.0)
    jitter: bool = Field(default=True)

    @model_validator(mode="after")
    def _validate_bounds(self) -> BackoffConfig:
        if self.strategy is BackoffStrategy.NONE:
            return self
        if self.initial < 0.05:
            raise ValueError("initial backoff must be >=0.05 for non-none strategies")
        if self.maximum < 0.1:
            raise ValueError("maximum backoff must be >=0.1 for non-none strategies")
        if self.maximum < self.initial:
            raise ValueError("maximum backoff must be >= initial backoff")
        return self


class ResiliencePolicyConfig(BaseModel):
    """Configuration describing retry, circuit breaker, and rate limiting."""

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(ge=1, le=10)
    timeout_seconds: int = Field(ge=1, le=600)
    backoff: BackoffConfig = Field(default_factory=BackoffConfig)
    circuit_breaker: CircuitBreakerConfig | None = None
    rate_limit: RateLimitConfig | None = None


class ResiliencePolicy(BaseModel):
    """Runtime wrapper around :class:`ResiliencePolicyConfig`."""

    name: str
    config: ResiliencePolicyConfig

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def create_retry(self, stage: str):
        from tenacity import retry, stop_after_attempt
        from tenacity import wait_none
        from tenacity.wait import wait_exponential, wait_fixed, wait_incrementing

        backoff = self.config.backoff
        if backoff.strategy is BackoffStrategy.NONE:
            wait = wait_none()
        elif backoff.strategy is BackoffStrategy.LINEAR:
            wait = wait_incrementing(
                start=backoff.initial,
                increment=backoff.initial,
                max=backoff.maximum,
            )
        else:
            wait = wait_exponential(
                multiplier=backoff.initial,
                max=backoff.maximum,
                exp_base=2,
            )
        if backoff.jitter and hasattr(wait, "with_jitter"):
            wait = wait.with_jitter(0.1)  # type: ignore[attr-defined]

        def _before_sleep(retry_state):
            record_resilience_retry(self.name, stage)

        return retry(
            stop=stop_after_attempt(self.config.max_attempts),
            wait=wait,
            reraise=True,
            before_sleep=_before_sleep,
        )

    def create_circuit_breaker(self, stage: str):
        try:
            from pybreaker import CircuitBreaker, CircuitBreakerListener
        except ModuleNotFoundError:  # pragma: no cover - optional dependency
            logger.warning(
                "resilience.circuit_breaker.disabled policy=%s stage=%s reason=%s",
                self.name,
                stage,
                "pybreaker_missing",
            )
            return None

        cfg = self.config.circuit_breaker
        if cfg is None:
            return None

        class _Listener(CircuitBreakerListener):
            def __init__(self, policy_name: str, stage_name: str) -> None:
                self._policy_name = policy_name
                self._stage_name = stage_name

            def state_change(self, cb, old_state, new_state):  # type: ignore[override]
                record_resilience_circuit_state(
                    self._policy_name, self._stage_name, str(new_state)
                )

        return CircuitBreaker(
            fail_max=cfg.failure_threshold,
            reset_timeout=cfg.recovery_timeout,
            listeners=[_Listener(self.name, stage)],
        )

    def create_rate_limiter(self):
        try:
            from aiolimiter import AsyncLimiter
        except ModuleNotFoundError:  # pragma: no cover - optional dependency
            logger.warning(
                "resilience.rate_limiter.disabled policy=%s reason=%s",
                self.name,
                "aiolimiter_missing",
            )
            return None

        cfg = self.config.rate_limit
        if cfg is None:
            return None
        return AsyncLimiter(cfg.rate_limit_per_second, 1)


def _topological_sort(graph: Mapping[str, Iterable[str]]) -> list[str] | None:
    in_degree: dict[str, int] = {node: 0 for node in graph}
    for deps in graph.values():
        for dep in deps:
            in_degree[dep] = in_degree.get(dep, 0) + 1
    queue = [node for node, degree in in_degree.items() if degree == 0]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for dep in graph.get(node, []):
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)
    if len(order) != len(in_degree):
        return None
    return order


@dataclass(slots=True)
class _CacheEntry:
    config: PipelineTopologyConfig
    mtime: float


class PipelineConfigLoader:
    """Load and cache pipeline topology YAML files."""

    def __init__(self, base_path: str | Path | None = None) -> None:
        self.base_path = Path(base_path or "config/orchestration/pipelines")
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self._watchers: list[Callable[[str, PipelineTopologyConfig], None]] = []
        self._watch_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def load(self, name: str, *, force: bool = False) -> PipelineTopologyConfig:
        path = self.base_path / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Pipeline config '{name}' not found at {path}")
        with self._lock:
            entry = self._cache.get(name)
            mtime = path.stat().st_mtime
            if force or entry is None or entry.mtime < mtime:
                logger.info("pipeline.config.reload", pipeline=name, path=str(path))
                raw = yaml.safe_load(path.read_text()) or {}
                try:
                    config = PipelineTopologyConfig.model_validate(raw)
                except ValidationError as exc:
                    raise ValueError(f"Invalid pipeline config '{name}': {exc}") from exc
                entry = _CacheEntry(config=config, mtime=mtime)
                self._cache[name] = entry
                for watcher in self._watchers:
                    watcher(name, entry.config)
            return entry.config

    def invalidate(self, name: str) -> None:
        with self._lock:
            self._cache.pop(name, None)

    def watch(self, callback: Callable[[str, PipelineTopologyConfig], None], *, interval: float = 2.0) -> None:
        self._watchers.append(callback)
        if self._watch_thread is None:
            self._start_watcher(interval)

    def _start_watcher(self, interval: float) -> None:
        def _run() -> None:
            while not self._stop_event.is_set():
                with self._lock:
                    items = list(self._cache.items())
                for name, entry in items:
                    path = self.base_path / f"{name}.yaml"
                    try:
                        mtime = path.stat().st_mtime
                    except FileNotFoundError:
                        continue
                    if mtime > entry.mtime:
                        self.load(name, force=True)
                time.sleep(interval)

        self._watch_thread = threading.Thread(target=_run, name="pipeline-config-watcher", daemon=True)
        self._watch_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=1.0)


class ResiliencePolicyLoader:
    """Load resilience policy definitions and provide helper factories."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or "config/orchestration/resilience.yaml")
        self._policies: dict[str, ResiliencePolicy] = {}
        self._lock = threading.Lock()
        self._watchers: list[Callable[[str, ResiliencePolicy], None]] = []
        self._watch_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def load(self, *, force: bool = False) -> dict[str, ResiliencePolicy]:
        if not self.path.exists():
            raise FileNotFoundError(f"Resilience policy config not found at {self.path}")
        with self._lock:
            if force or not self._policies:
                raw = yaml.safe_load(self.path.read_text()) or {}
                data = raw.get("policies", {})
                policies: dict[str, ResiliencePolicy] = {}
                for name, payload in data.items():
                    try:
                        config = ResiliencePolicyConfig.model_validate(payload)
                    except ValidationError as exc:
                        raise ValueError(f"Invalid resilience policy '{name}': {exc}") from exc
                    policies[name] = ResiliencePolicy(name=name, config=config)
                self._policies = policies
                for watcher in self._watchers:
                    for name, policy in policies.items():
                        watcher(name, policy)
            return dict(self._policies)

    def get(self, name: str) -> ResiliencePolicy:
        with self._lock:
            if name not in self._policies:
                self.load()
            try:
                return self._policies[name]
            except KeyError as exc:
                raise KeyError(f"Unknown resilience policy '{name}'") from exc

    def apply(self, name: str, stage: str, func: Callable[..., Any]) -> Callable[..., Any]:
        policy = self.get(name)
        if asyncio.iscoroutinefunction(func):
            raise TypeError("ResiliencePolicyLoader.apply only supports synchronous callables")

        retry_decorator = policy.create_retry(stage)
        circuit_breaker = policy.create_circuit_breaker(stage)
        limiter = policy.create_rate_limiter()

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            if limiter is not None:
                start = time.perf_counter()
                try:
                    asyncio.run(limiter.acquire())
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    try:
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(limiter.acquire())
                    finally:
                        asyncio.set_event_loop(None)
                        loop.close()
                waited = time.perf_counter() - start
                if waited:
                    record_resilience_rate_limit_wait(name, stage, waited)
            call = func
            if circuit_breaker is not None:

                def _call_with_breaker(*inner_args: Any, **inner_kwargs: Any) -> Any:
                    return circuit_breaker.call(call, *inner_args, **inner_kwargs)

                call = _call_with_breaker
            return call(*args, **kwargs)

        return retry_decorator(_wrapped)

    def watch(self, callback: Callable[[str, ResiliencePolicy], None], *, interval: float = 2.0) -> None:
        self._watchers.append(callback)
        if self._watch_thread is None:
            self._start_watcher(interval)

    def _start_watcher(self, interval: float) -> None:
        def _run() -> None:
            last_mtime = 0.0
            while not self._stop_event.is_set():
                try:
                    mtime = self.path.stat().st_mtime
                except FileNotFoundError:
                    time.sleep(interval)
                    continue
                if mtime > last_mtime:
                    last_mtime = mtime
                    self.load(force=True)
                time.sleep(interval)

        self._watch_thread = threading.Thread(target=_run, name="resilience-policy-watcher", daemon=True)
        self._watch_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=1.0)


def export_pipeline_schema(path: str | Path) -> None:
    """Write the JSON schema for pipeline topology configs to disk."""

    schema = PipelineTopologyConfig.model_json_schema(by_alias=True)
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(schema, indent=2, sort_keys=True))


__all__ = [
    "BackoffStrategy",
    "GateCondition",
    "GateDefinition",
    "PipelineConfigLoader",
    "PipelineTopologyConfig",
    "ResiliencePolicy",
    "ResiliencePolicyConfig",
    "ResiliencePolicyLoader",
    "export_pipeline_schema",
]
