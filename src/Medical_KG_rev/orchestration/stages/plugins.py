"""Plugin infrastructure for orchestration stages.

This module introduces a ``pluggy`` based discovery and loading system that
allows orchestration stages to be contributed by external packages while the
core runtime remains agnostic of concrete implementations.  The implementation
leans on ``attrs`` for the hot-path data structures, ``pydantic`` for metadata
validation, ``tenacity`` for resilient loading, ``orjson`` for fast
serialisation of diagnostics, and ``prometheus_client`` for observability.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import suppress
from typing import Any, Callable, Iterable, Literal, Sequence

import orjson
import pluggy
import structlog
from attrs import define, evolve, field
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, ConfigDict, Field
from structlog.stdlib import BoundLogger
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from Medical_KG_rev.adapters.plugins.manager import AdapterPluginManager
from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition

__all__ = [
    "STAGE_PLUGIN_NAMESPACE",
    "hookspec",
    "hookimpl",
    "StagePluginMetadata",
    "StagePluginRegistration",
    "StagePlugin",
    "StagePluginHealth",
    "StagePluginResources",
    "StagePluginManager",
    "StagePluginError",
    "StagePluginLookupError",
    "StagePluginBuildError",
]


STAGE_PLUGIN_NAMESPACE = "medical_kg_rev.orchestration.stage"

hookspec = pluggy.HookspecMarker(STAGE_PLUGIN_NAMESPACE)
hookimpl = pluggy.HookimplMarker(STAGE_PLUGIN_NAMESPACE)


class StagePluginMetadata(BaseModel):
    """Typed metadata describing a stage plugin contribution."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    version: str = Field(default="0.0.0")
    stage_type: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    capabilities: tuple[str, ...] = Field(default_factory=tuple)
    dependencies: tuple[str, ...] = Field(default_factory=tuple)

    def serialise(self) -> bytes:
        """Return an ``orjson`` encoded payload for diagnostics."""

        return orjson.dumps(self.model_dump())


@define(slots=True)
class StagePluginRegistration:
    """Registration record returned by plugin implementations."""

    metadata: StagePluginMetadata
    builder: Callable[[StageDefinition, "StagePluginResources"], object]
    provider: "StagePlugin | None" = None

    def bind(self, provider: "StagePlugin") -> "StagePluginRegistration":
        """Return a copy of the registration bound to the provider."""

        if self.provider is provider:
            return self
        return evolve(self, provider=provider)


@define(slots=True)
class StagePluginResources:
    """Shared resources handed to plugin builders during registration."""

    adapter_manager: AdapterPluginManager
    pipeline_resource: Any
    job_ledger: Any | None = None
    extras: dict[str, Any] = field(factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.extras.get(key, default)

    def with_extra(self, **values: Any) -> "StagePluginResources":
        payload = dict(self.extras)
        payload.update(values)
        return StagePluginResources(
            adapter_manager=self.adapter_manager,
            pipeline_resource=self.pipeline_resource,
            job_ledger=self.job_ledger,
            extras=payload,
        )


class StagePluginError(RuntimeError):
    """Base error for stage plugin manager failures."""


class StagePluginLookupError(StagePluginError):
    """Raised when no plugin can satisfy the requested stage type."""


class StagePluginBuildError(StagePluginError):
    """Raised when all plugins for a stage type fail to build an instance."""


@define(slots=True)
class StagePluginHealth:
    """Structured health status returned by plugins."""

    status: Literal["ok", "degraded", "error"] = "ok"
    detail: str | None = None
    timestamp: float = field(factory=time.time)


class StagePlugin(ABC):
    """Base class for stage plugins contributed via :mod:`pluggy`."""

    plugin_name: str | None = None
    version: str = "0.0.0"
    plugin_dependencies: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        plugin_name: str | None = None,
        version: str | None = None,
        dependencies: Sequence[str] | None = None,
    ) -> None:
        self._name = plugin_name or self.plugin_name or self.__class__.__name__.lower()
        self._version = version or self.version
        self._dependencies = tuple(dependencies) if dependencies is not None else tuple(self.plugin_dependencies)
        self._initialized = False
        self._logger = structlog.get_logger(__name__).bind(stage_plugin=self._name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def dependencies(self) -> tuple[str, ...]:
        return self._dependencies

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self, resources: StagePluginResources) -> None:
        """Prepare plugin resources. Override to provide custom setup."""

    def cleanup(self) -> None:
        """Release plugin resources prior to unloading."""

    def health_check(self) -> StagePluginHealth:
        """Return the current plugin health signal."""

        return StagePluginHealth(status="ok")

    def create_registration(
        self,
        *,
        stage_type: str,
        builder: Callable[[StageDefinition, StagePluginResources], object],
        capabilities: Sequence[str] | None = None,
        dependencies: Sequence[str] | None = None,
    ) -> StagePluginRegistration:
        metadata = StagePluginMetadata(
            name=f"{self.name}.{stage_type}",
            version=self._version,
            stage_type=stage_type,
            capabilities=tuple(capabilities or ()),
            dependencies=tuple(dependencies) if dependencies is not None else self.dependencies,
        )
        return StagePluginRegistration(metadata=metadata, builder=builder, provider=self)

    @abstractmethod
    def registrations(
        self, resources: StagePluginResources
    ) -> Sequence[StagePluginRegistration]:
        """Return stage registrations exposed by the plugin."""

    def _ensure_initialized(self, resources: StagePluginResources) -> None:
        if self._initialized:
            return
        self.initialize(resources)
        self._initialized = True
        self._logger.debug("stage.plugin.initialized")

    @hookimpl
    def stage_builders(
        self, resources: StagePluginResources
    ) -> Sequence[StagePluginRegistration]:
        """Pluggy hook entry point delegating to :meth:`registrations`."""

        self._ensure_initialized(resources)
        registrations = self.registrations(resources)
        enriched: list[StagePluginRegistration] = []
        for registration in registrations:
            entry = registration.bind(self)
            metadata = entry.metadata
            payload = metadata
            if metadata.version != self._version:
                payload = metadata.model_copy(update={"version": self._version})
                entry = evolve(entry, metadata=payload)
            if not payload.dependencies and self.dependencies:
                payload = payload.model_copy(update={"dependencies": self.dependencies})
                entry = evolve(entry, metadata=payload)
            enriched.append(entry)
        return tuple(enriched)


class StagePluginSpec:
    """Pluggy hook specification for stage plugins."""

    @hookspec
    def stage_builders(
        self, resources: StagePluginResources
    ) -> Sequence[StagePluginRegistration | dict[str, Any]]:
        """Return registrations for the stages implemented by the plugin."""


_BUILD_COUNTER = Counter(
    "medical_kg_stage_plugin_build_total",
    "Number of stage builds performed by plugin",  # pragma: no cover - metrics wiring
    ("plugin", "stage_type", "status"),
)

_BUILD_LATENCY = Histogram(
    "medical_kg_stage_plugin_build_seconds",
    "Latency for stage plugin instantiation",  # pragma: no cover - metrics wiring
    ("plugin", "stage_type"),
)


@define(slots=True)
class StagePluginManager:
    """Manage discovery, validation, and instantiation of stage plugins."""

    resources: StagePluginResources
    namespace: str = STAGE_PLUGIN_NAMESPACE
    _plugin_manager: pluggy.PluginManager = field(init=False)
    _registry: dict[str, list[StagePluginRegistration]] = field(init=False, factory=dict)
    _states: dict[str, "_StageRegistrationState"] = field(init=False, factory=dict)
    _stage_index: dict[str, set[str]] = field(init=False, factory=dict)
    _logger: BoundLogger = field(init=False)

    def __attrs_post_init__(self) -> None:
        self._plugin_manager = pluggy.PluginManager(self.namespace)
        self._plugin_manager.add_hookspecs(StagePluginSpec)
        self._logger = structlog.get_logger(__name__).bind(component="StagePluginManager")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=5.0))
    def load_entrypoints(self, group: str | None = None) -> None:
        """Load plugins advertised through Python entry points."""

        entrypoint_group = group or self.namespace
        loaded = self._plugin_manager.load_setuptools_entrypoints(entrypoint_group) or []
        count = len(loaded) if isinstance(loaded, Iterable) else 0
        self._logger.info(
            "stage.plugin.entrypoints_loaded",
            group=entrypoint_group,
            count=count,
        )
        self._refresh_registry()

    def register(self, plugin: object) -> None:
        """Register an explicit plugin object with the manager."""

        self._plugin_manager.register(plugin)
        self._logger.debug(
            "stage.plugin.registered",
            plugin=getattr(plugin, "__class__", type(plugin)).__name__,
        )
        self._refresh_registry()

    def unregister(self, plugin: StagePlugin | str) -> None:
        """Unregister a plugin contribution and release any associated resources."""

        provider: StagePlugin | None
        key: str
        if isinstance(plugin, StagePlugin):
            provider = plugin
            key = plugin.name
            target = plugin
        else:
            key = plugin
            state = self._states.get(plugin)
            provider = state.provider if state else None
            target = provider or plugin

        with suppress(Exception):
            self._plugin_manager.unregister(target)
        if provider is not None:
            with suppress(Exception):
                provider.cleanup()

        removal_keys = [
            name
            for name, state in self._states.items()
            if name == key or state.provider_name == key
        ]
        for name in removal_keys:
            self._states.pop(name, None)
            self._stage_index.pop(name, None)
        for stage_type, registrations in list(self._registry.items()):
            filtered = [reg for reg in registrations if reg.metadata.name not in removal_keys]
            if filtered:
                self._registry[stage_type] = filtered
            else:
                self._registry.pop(stage_type, None)
        self._logger.info("stage.plugin.unregistered", plugin=key)

    def available_stage_types(self) -> tuple[str, ...]:
        return tuple(sorted(self._registry))

    def plugin_inventory(self) -> tuple[StagePluginMetadata, ...]:
        entries: list[StagePluginMetadata] = []
        for registrations in self._registry.values():
            entries.extend(entry.metadata for entry in registrations)
        return tuple(entries)

    def describe_plugins(self) -> tuple[dict[str, Any], ...]:
        """Return diagnostic metadata about registered plugin contributions."""

        payload: list[dict[str, Any]] = []
        for name, state in self._states.items():
            payload.append(
                {
                    "name": name,
                    "stage_type": state.stage_type,
                    "provider": state.provider_name,
                    "dependencies": state.dependencies,
                    "status": state.status,
                    "last_error": state.last_error,
                    "last_health": state.last_health,
                }
            )
        return tuple(sorted(payload, key=lambda entry: entry["name"]))

    def check_health(self) -> dict[str, StagePluginHealth]:
        """Execute health checks for providers that expose them."""

        report: dict[str, StagePluginHealth] = {}
        for name, state in self._states.items():
            provider = state.provider
            if provider is None:
                continue
            try:
                health = provider.health_check()
            except Exception as exc:  # pragma: no cover - defensive guard
                message = str(exc)
                state.last_error = message
                health = StagePluginHealth(status="error", detail=message)
            state.last_health = health
            if provider.is_initialized:
                state.status = "initialized"
            report[name] = health
        return report

    def build_stage(self, definition: StageDefinition) -> object:
        """Instantiate a stage for the supplied definition."""

        stage_type = definition.stage_type
        registrations = self._registry.get(stage_type, [])
        if not registrations:
            raise StagePluginLookupError(f"No stage plugin registered for type '{stage_type}'")

        last_error: Exception | None = None
        for registration in registrations:
            metadata = registration.metadata
            start_time = time.perf_counter()
            try:
                instance = self._execute_with_retry(registration.builder, definition)
            except RetryError as exc:
                error = exc.last_attempt.exception() if exc.last_attempt else exc
                last_error = error
                _BUILD_COUNTER.labels(
                    plugin=metadata.name,
                    stage_type=stage_type,
                    status="error",
                ).inc()
                self._logger.error(
                    "stage.plugin.retry_exhausted",
                    plugin=metadata.name,
                    stage_type=stage_type,
                    error=str(error),
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive guard
                last_error = exc
                _BUILD_COUNTER.labels(
                    plugin=metadata.name,
                    stage_type=stage_type,
                    status="error",
                ).inc()
                self._logger.error(
                    "stage.plugin.build_failed",
                    plugin=metadata.name,
                    stage_type=stage_type,
                    error=str(exc),
                )
                continue

            duration = time.perf_counter() - start_time
            _BUILD_COUNTER.labels(
                plugin=metadata.name,
                stage_type=stage_type,
                status="success",
            ).inc()
            _BUILD_LATENCY.labels(plugin=metadata.name, stage_type=stage_type).observe(duration)
            self._logger.debug(
                "stage.plugin.build_succeeded",
                plugin=metadata.name,
                stage_type=stage_type,
                duration_ms=int(duration * 1000),
                metadata=metadata.serialise(),
            )
            return instance

        raise StagePluginBuildError(
            f"All stage plugins failed for type '{stage_type}'"
        ) from last_error

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.1, max=2.0))
    def _execute_with_retry(
        self,
        builder: Callable[[StageDefinition, StagePluginResources], object],
        definition: StageDefinition,
    ) -> object:
        return builder(definition, self.resources)

    def _refresh_registry(self) -> None:
        aggregated: dict[str, list[StagePluginRegistration]] = defaultdict(list)
        states: dict[str, _StageRegistrationState] = {}
        results = self._plugin_manager.hook.stage_builders(resources=self.resources)
        for item in results:
            if not item:
                continue
            for registration in item:
                entry = self._coerce(registration)
                state = states.get(entry.metadata.name)
                if state is None:
                    state = _StageRegistrationState.from_registration(entry)
                else:
                    state.update(entry)
                states[state.name] = state
                aggregated[entry.metadata.stage_type].append(entry)

        order = self._resolve_dependency_order(states)
        if order:
            ranking = {name: index for index, name in enumerate(order)}
            for stage_type, registrations in aggregated.items():
                registrations.sort(key=lambda reg: ranking.get(reg.metadata.name, len(ranking)))

        self._registry = aggregated
        self._states = states
        self._stage_index = {name: state.stage_types for name, state in states.items()}
        self._logger.debug(
            "stage.plugin.registry_refreshed",
            stage_types=sorted(self._registry),
        )

    def _coerce(self, entry: StagePluginRegistration | dict[str, Any]) -> StagePluginRegistration:
        if isinstance(entry, StagePluginRegistration):
            return entry
        if not isinstance(entry, dict):
            raise TypeError(f"Unsupported stage registration payload: {entry!r}")
        metadata_payload = entry.get("metadata")
        if metadata_payload is None:
            raise ValueError("Stage registration missing metadata")
        builder = entry.get("builder")
        if not callable(builder):
            raise TypeError("Stage registration builder must be callable")
        metadata = (
            metadata_payload
            if isinstance(metadata_payload, StagePluginMetadata)
            else StagePluginMetadata.model_validate(metadata_payload)
        )
        provider = entry.get("provider")
        provider_obj = provider if isinstance(provider, StagePlugin) else None
        return StagePluginRegistration(metadata=metadata, builder=builder, provider=provider_obj)

    def _resolve_dependency_order(
        self, states: dict[str, "_StageRegistrationState"]
    ) -> tuple[str, ...]:
        graph = {name: state.dependencies for name, state in states.items()}
        visited: set[str] = set()
        visiting: set[str] = set()
        order: list[str] = []

        def visit(node: str) -> None:
            if node in visited:
                return
            if node in visiting:
                raise StagePluginError(f"Cycle detected in stage plugin dependencies: {node}")
            visiting.add(node)
            for dependency in graph.get(node, ()):  # pragma: no branch - defensive
                if dependency not in graph:
                    self._logger.warning(
                        "stage.plugin.dependency_missing", plugin=node, dependency=dependency
                    )
                    continue
                visit(dependency)
            visiting.remove(node)
            visited.add(node)
            order.append(node)

        for name in graph:
            visit(name)

        return tuple(order)


@define(slots=True)
class _StageRegistrationState:
    """Internal bookkeeping for individual stage registrations."""

    name: str
    stage_type: str
    provider_name: str
    dependencies: tuple[str, ...] = ()
    provider: StagePlugin | None = None
    status: str = "registered"
    last_error: str | None = None
    last_health: StagePluginHealth | None = None
    stage_types: set[str] = field(factory=set)

    @classmethod
    def from_registration(cls, registration: StagePluginRegistration) -> "_StageRegistrationState":
        provider_name = (
            registration.provider.name
            if isinstance(registration.provider, StagePlugin)
            else registration.metadata.name
        )
        state = cls(
            name=registration.metadata.name,
            stage_type=registration.metadata.stage_type,
            provider_name=provider_name,
            dependencies=registration.metadata.dependencies,
            provider=registration.provider if isinstance(registration.provider, StagePlugin) else None,
            status="initialized"
            if isinstance(registration.provider, StagePlugin)
            and registration.provider.is_initialized
            else "registered",
        )
        state.stage_types.add(registration.metadata.stage_type)
        return state

    def update(self, registration: StagePluginRegistration) -> None:
        self.dependencies = registration.metadata.dependencies
        if isinstance(registration.provider, StagePlugin):
            self.provider = registration.provider
            self.provider_name = registration.provider.name
            if registration.provider.is_initialized:
                self.status = "initialized"
        self.stage_type = registration.metadata.stage_type
        self.stage_types.add(registration.metadata.stage_type)

