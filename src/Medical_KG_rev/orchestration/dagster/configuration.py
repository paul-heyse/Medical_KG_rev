"""Configuration models and loaders for Dagster-based orchestration."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from Medical_KG_rev.observability.metrics import (
    record_resilience_circuit_state,
    record_resilience_rate_limit_wait,
    record_resilience_retry,
)
from Medical_KG_rev.utils.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover - hints only
    from aiolimiter import AsyncLimiter
    from pybreaker import CircuitBreaker


class BackoffStrategy(str, Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    NONE = "none"


class GateConditionOperator(str, Enum):
    """Supported operators for gate condition predicates."""

    EQUALS = "equals"
    EXISTS = "exists"
    CHANGED = "changed"
    NOT_EQUALS = "not_equals"
    IN = "in"


class GatePredicate(BaseModel):
    """Single ledger predicate evaluated as part of a gate clause."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    operator: GateConditionOperator = Field(default=GateConditionOperator.EQUALS)
    value: Any = Field(default=True)

    @model_validator(mode="after")
    def _validate_value(cls, model: GatePredicate) -> GatePredicate:
        if model.operator is GateConditionOperator.EXISTS and model.value not in {True, False}:
            raise ValueError("exists operator expects a boolean value")
        if model.operator is GateConditionOperator.CHANGED and model.value not in {True, False}:
            raise ValueError("changed operator expects a boolean value")
        return model


class GateCondition(BaseModel):
    """Compound gate condition supporting AND/OR clauses."""

    model_config = ConfigDict(extra="forbid")

    all: Sequence[GatePredicate] | None = None
    any: Sequence[GatePredicate] | None = None
    description: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_format(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "field" in value:
            predicate: dict[str, Any] = {
                "field": value["field"],
                "operator": value.get("operator", "equals"),
            }
            if "value" in value:
                predicate["value"] = value["value"]
            elif "equals" in value:
                predicate["value"] = value["equals"]
            return {"all": [predicate]}
        return value

    @model_validator(mode="after")
    def _validate_structure(self) -> GateCondition:
        if not self.all and not self.any:
            raise ValueError("gate condition requires at least one 'all' or 'any' clause")
        return self


class GateRetryConfig(BaseModel):
    """Retry configuration for polling gate conditions."""

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=1, ge=1, le=20)
    backoff_seconds: float = Field(default=5.0, ge=0.5, le=300.0)


class GateDefinition(BaseModel):
    """Declarative definition for a pipeline gate."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    resume_stage: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    conditions: Sequence[GateCondition] = Field(default_factory=list)
    timeout_seconds: int | None = Field(default=None, ge=1, le=3600)
    poll_interval_seconds: float = Field(default=5.0, ge=0.5, le=60.0)
    resume_phase: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]+$")
    description: str | None = None
    retry: GateRetryConfig | None = None

    @model_validator(mode="after")
    def _validate_conditions(self) -> GateDefinition:
        if not self.conditions:
            raise ValueError("gate definition must include at least one condition clause")
        return self


class StageDefinition(BaseModel):
    """Declarative stage specification for topology YAML files."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    stage_type: str = Field(alias="type", pattern=r"^[A-Za-z0-9_-]+$")
    policy: str | None = Field(default=None, alias="policy")
    depends_on: list[str] = Field(default_factory=list, alias="depends_on")
    config: dict[str, Any] = Field(default_factory=dict)
    phase: str = Field(default="default", alias="phase", pattern=r"^[A-Za-z0-9_-]+$")
    gate: str | None = Field(default=None, alias="gate", pattern=r"^[A-Za-z0-9_-]+$")

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

    @model_validator(mode="after")
    def _validate_gate_usage(self) -> StageDefinition:
        if self.stage_type != "gate" and self.gate is not None:
            raise ValueError("gate attribute can only be set on gate stages")
        return self


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
    phase_order: list[str] = Field(default_factory=list, alias="phase_order")

    @model_validator(mode="after")
    def _validate_dependencies(self) -> PipelineTopologyConfig:
        stage_names = [stage.name for stage in self.stages]
        if len(stage_names) != len(set(stage_names)):
            duplicates = {name for name in stage_names if stage_names.count(name) > 1}
            raise ValueError(f"duplicate stage names detected: {sorted(duplicates)}")

        stage_map = {stage.name: stage for stage in self.stages}
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

        encountered_phases: list[str] = []
        for stage in self.stages:
            if stage.phase not in encountered_phases:
                encountered_phases.append(stage.phase)
        if self.phase_order:
            normalised: list[str] = []
            for phase in self.phase_order:
                if phase not in normalised:
                    normalised.append(phase)
            missing_phases = [phase for phase in encountered_phases if phase not in normalised]
            if missing_phases:
                raise ValueError(
                    "phase_order is missing phases declared by stages: "
                    + ", ".join(sorted(missing_phases))
                )
            self.phase_order = normalised
        else:
            self.phase_order = encountered_phases

        phase_indices = {phase: index for index, phase in enumerate(self.phase_order)}
        gate_map = {gate.name: gate for gate in self.gates}
        referenced_gates: set[str] = set()

        for stage in self.stages:
            stage_phase_index = phase_indices[stage.phase]
            for dependency in stage.depends_on:
                dep_stage = stage_map[dependency]
                if phase_indices[dep_stage.phase] > stage_phase_index:
                    raise ValueError(
                        f"stage '{stage.name}' depends on future phase stage '{dependency}'"
                    )

            if stage.stage_type == "gate":
                gate_name = stage.gate or stage.config.get("gate") or stage.name
                if gate_name not in gate_map:
                    raise ValueError(
                        f"gate stage '{stage.name}' references unknown gate '{gate_name}'"
                    )
                gate_def = gate_map[gate_name]
                referenced_gates.add(gate_name)
                stage.config.setdefault("gate", gate_name)
                stage.config.setdefault("definition", gate_def.model_dump())
                if gate_def.resume_stage not in stage_map:
                    raise ValueError(
                        f"gate '{gate_name}' resume_stage '{gate_def.resume_stage}' is not a stage"
                    )
                resume_stage = stage_map[gate_def.resume_stage]
                resume_phase = gate_def.resume_phase or resume_stage.phase
                gate_def.resume_phase = resume_phase
                if resume_phase not in phase_indices:
                    raise ValueError(
                        f"gate '{gate_name}' resume_phase '{resume_phase}' is not defined"
                    )
                if phase_indices[resume_phase] <= phase_indices[stage.phase]:
                    raise ValueError(
                        f"gate '{gate_name}' must resume in a phase after '{stage.phase}'"
                    )

        unused_gates = set(gate_map) - referenced_gates
        if unused_gates:
            raise ValueError(
                "pipeline declares gate definitions without corresponding gate stages: "
                + ", ".join(sorted(unused_gates))
            )

        for gate in self.gates:
            if gate.resume_stage not in stage_map:
                raise ValueError(
                    f"gate '{gate.name}' references unknown resume_stage '{gate.resume_stage}'"
                )
            resume_stage = stage_map[gate.resume_stage]
            resume_phase = gate.resume_phase or resume_stage.phase
            if resume_phase not in phase_indices:
                raise ValueError(
                    f"gate '{gate.name}' references unknown resume_phase '{resume_phase}'"
                )
            if phase_indices[resume_phase] <= phase_indices[resume_stage.phase]:
                raise ValueError(
                    f"gate '{gate.name}' resume_phase must be after resume_stage phase"
                )

        return self

    def build_phase_plan(self) -> "PipelinePhasePlan":
        order = _topological_sort({stage.name: stage.depends_on for stage in self.stages}) or []
        stage_map = {stage.name: stage for stage in self.stages}

        phases: tuple[str, ...] = tuple(
            phase
            for phase in self.phase_order
            if any(stage.phase == phase for stage in self.stages)
        )
        phase_to_stages: dict[str, list[str]] = {phase: [] for phase in phases}
        stage_to_phase: dict[str, str] = {}
        stage_positions: dict[str, int] = {}

        for stage_name in order:
            stage = stage_map[stage_name]
            phase = stage.phase
            if phase not in phase_to_stages:
                continue
            position = len(phase_to_stages[phase])
            phase_to_stages[phase].append(stage_name)
            stage_to_phase[stage_name] = phase
            stage_positions[stage_name] = position

        gate_by_stage: dict[str, GateDefinition] = {}
        gate_for_phase: dict[str, GateDefinition | None] = {phase: None for phase in phases}
        gate_map = {gate.name: gate for gate in self.gates}
        for stage in self.stages:
            if stage.stage_type != "gate":
                continue
            gate_name = stage.gate or stage.config.get("gate") or stage.name
            gate = gate_map.get(gate_name)
            if gate is None:
                continue
            gate_by_stage[stage.name] = gate
            resume_stage = stage_map[gate.resume_stage]
            resume_phase = gate.resume_phase or resume_stage.phase
            if resume_phase in gate_for_phase:
                gate_for_phase[resume_phase] = gate

        serialised_phase_stages = {
            phase: tuple(stages)
            for phase, stages in phase_to_stages.items()
        }

        return PipelinePhasePlan(
            phases=phases,
            phase_to_stages=serialised_phase_stages,
            stage_to_phase=stage_to_phase,
            stage_positions=stage_positions,
            gate_by_stage=gate_by_stage,
            gate_for_phase=gate_for_phase,
        )


@dataclass(slots=True)
class PipelinePhasePlan:
    """Derived orchestration plan for phase-aware execution."""

    phases: tuple[str, ...]
    phase_to_stages: dict[str, tuple[str, ...]]
    stage_to_phase: dict[str, str]
    stage_positions: dict[str, int]
    gate_by_stage: dict[str, GateDefinition]
    gate_for_phase: dict[str, GateDefinition | None]
    phase_indices: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase_indices", {phase: idx for idx, phase in enumerate(self.phases)})

    def phase_index(self, phase: str) -> int:
        return self.phase_indices[phase]


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


@dataclass(slots=True)
class StageExecutionHooks:
    """Lifecycle callbacks for stage execution resilience wrappers."""

    on_retry: Callable[[Any], None] | None = None
    on_success: Callable[[int, float], None] | None = None
    on_failure: Callable[[BaseException, int], None] | None = None


class ResiliencePolicy(BaseModel):
    """Runtime wrapper around :class:`ResiliencePolicyConfig`."""

    name: str
    config: ResiliencePolicyConfig

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _rate_limiter: "AsyncLimiter | None" = PrivateAttr(default=None)
    _circuit_breakers: dict[str, "CircuitBreaker"] = PrivateAttr(default_factory=dict)

    def create_retry(self, stage: str, hooks: StageExecutionHooks | None = None):
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
            if hooks and hooks.on_retry:
                hooks.on_retry(retry_state)

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

        if stage in self._circuit_breakers:
            return self._circuit_breakers[stage]

        class _Listener(CircuitBreakerListener):
            def __init__(self, policy_name: str, stage_name: str) -> None:
                self._policy_name = policy_name
                self._stage_name = stage_name

            def state_change(self, cb, old_state, new_state):  # type: ignore[override]
                record_resilience_circuit_state(
                    self._policy_name, self._stage_name, str(new_state)
                )

        breaker = CircuitBreaker(
            fail_max=cfg.failure_threshold,
            reset_timeout=cfg.recovery_timeout,
            listeners=[_Listener(self.name, stage)],
        )
        self._circuit_breakers[stage] = breaker
        return breaker

    def get_rate_limiter(self) -> "AsyncLimiter | None":
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
        if self._rate_limiter is None:
            rate = max(cfg.rate_limit_per_second, 1e-6)
            self._rate_limiter = AsyncLimiter(rate, time_period=1.0)
        return self._rate_limiter


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


class _SyncLimiter:
    """Run an AsyncLimiter behind a dedicated event loop for sync callers."""

    def __init__(self, limiter: "AsyncLimiter") -> None:
        self._limiter = limiter
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="resilience-policy-limiter",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def acquire(self) -> float:
        start = time.perf_counter()
        future = asyncio.run_coroutine_threadsafe(self._limiter.acquire(), self._loop)
        future.result()
        return time.perf_counter() - start

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1.0)

    @property
    def limiter(self) -> "AsyncLimiter":
        return self._limiter


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
        self._sync_limiters: dict[str, _SyncLimiter] = {}

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
                self._prune_limiters(set(policies))
            return dict(self._policies)

    def get(self, name: str) -> ResiliencePolicy:
        with self._lock:
            if name not in self._policies:
                self.load()
            try:
                return self._policies[name]
            except KeyError as exc:
                raise KeyError(f"Unknown resilience policy '{name}'") from exc

    def apply(
        self,
        name: str,
        stage: str,
        func: Callable[..., Any],
        hooks: StageExecutionHooks | None = None,
    ) -> Callable[..., Any]:
        policy = self.get(name)
        if asyncio.iscoroutinefunction(func):
            raise TypeError("ResiliencePolicyLoader.apply only supports synchronous callables")

        hooks = hooks or StageExecutionHooks()
        circuit_breaker = policy.create_circuit_breaker(stage)
        limiter = policy.get_rate_limiter()
        sync_limiter = None
        if limiter is not None:
            sync_limiter = self._sync_limiters.get(name)
            if sync_limiter is None or sync_limiter.limiter is not limiter:
                if sync_limiter is not None:
                    sync_limiter.close()
                sync_limiter = _SyncLimiter(limiter)
                self._sync_limiters[name] = sync_limiter

        def _invoke(*args: Any, **kwargs: Any) -> Any:
            call_state = {"attempts": 0}

            def _call(*inner_args: Any, **inner_kwargs: Any) -> Any:
                call_state["attempts"] += 1
                return func(*inner_args, **inner_kwargs)

            retry_decorator = policy.create_retry(stage, hooks)
            wrapped = retry_decorator(_call)
            start = time.perf_counter()
            try:
                result = wrapped(*args, **kwargs)
            except Exception as exc:
                if hooks.on_failure:
                    hooks.on_failure(exc, call_state["attempts"])
                raise
            duration = time.perf_counter() - start
            if hooks.on_success:
                hooks.on_success(call_state["attempts"], duration)
            return result

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            if sync_limiter is not None:
                waited = sync_limiter.acquire()
                if waited:
                    record_resilience_rate_limit_wait(name, stage, waited)
            call = _invoke
            if circuit_breaker is not None:
                def _call_with_breaker(*inner_args: Any, **inner_kwargs: Any) -> Any:
                    return circuit_breaker.call(call, *inner_args, **inner_kwargs)

                call = _call_with_breaker
            return call(*args, **kwargs)

        return _wrapped

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
        for limiter in list(self._sync_limiters.values()):
            limiter.close()
        self._sync_limiters.clear()

    def _prune_limiters(self, active: set[str]) -> None:
        for key in list(self._sync_limiters.keys()):
            if key not in active:
                limiter = self._sync_limiters.pop(key)
                limiter.close()


def export_pipeline_schema(path: str | Path) -> None:
    """Write the JSON schema for pipeline topology configs to disk."""

    schema = PipelineTopologyConfig.model_json_schema(by_alias=True)
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(schema, indent=2, sort_keys=True))


__all__ = [
    "BackoffStrategy",
    "GateCondition",
    "GateConditionOperator",
    "GateDefinition",
    "GatePredicate",
    "GateRetryConfig",
    "PipelineConfigLoader",
    "PipelinePhasePlan",
    "PipelineTopologyConfig",
    "ResiliencePolicy",
    "ResiliencePolicyConfig",
    "ResiliencePolicyLoader",
    "StageDefinition",
    "export_pipeline_schema",
]
