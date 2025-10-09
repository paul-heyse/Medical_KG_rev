"""Stage plugin infrastructure for Dagster orchestration."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import pluggy
from attrs import define, field
from pydantic import BaseModel, ConfigDict, ValidationError
from tenacity import (  # type: ignore
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import structlog
from Medical_KG_rev.observability.metrics import (
    STAGE_PLUGIN_FAILURES,
    STAGE_PLUGIN_HEALTH,
    STAGE_PLUGIN_REGISTRATIONS,
)

logger = structlog.get_logger(__name__)


PLUGIN_NAMESPACE = "medical_kg_stage_plugins"


hookspec = pluggy.HookspecMarker(PLUGIN_NAMESPACE)
hookimpl = pluggy.HookimplMarker(PLUGIN_NAMESPACE)


class StagePluginError(RuntimeError):
    """Base class for stage plugin errors."""


class StagePluginNotAvailable(StagePluginError):
    """Raised when no plugin can satisfy the requested stage type."""


class StagePluginLoadError(StagePluginError):
    """Raised when plugin discovery or validation fails."""


class StagePluginExecutionError(StagePluginError):
    """Raised when plugin execution fails even after retries."""


class StagePluginMetadata(BaseModel):
    """Structured description of a registered stage plugin."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    stage_types: tuple[str, ...]
    description: str | None = None


@define(slots=True)
class StagePluginContext:
    """Resources made available to plugins during stage construction."""

    resources: Mapping[str, Any]

    def require(self, key: str) -> Any:
        try:
            return self.resources[key]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise StagePluginLoadError(f"Missing resource '{key}' for stage plugin") from exc

    def get(self, key: str, default: Any = None) -> Any:
        """Get resource with optional default value."""
        return self.resources.get(key, default)


@define(slots=True)
class StagePlugin:
    """Base class for stage plugins."""

    metadata: StagePluginMetadata

    def initialise(self, context: StagePluginContext) -> None:
        """Called once when the plugin is registered."""

    def health_check(self, context: StagePluginContext) -> None:
        """Validate plugin health. Raises to mark plugin unhealthy."""

    def cleanup(self, context: StagePluginContext) -> None:
        """Cleanup resources during manager shutdown."""

    def create_stage(self, definition: Any, context: StagePluginContext) -> object:
        """Instantiate a stage implementation for the provided definition."""
        raise NotImplementedError


class StagePluginSpec:
    """pluggy hook specification for third-party plugins."""

    @hookspec
    def stage_plugins(self) -> Iterable[StagePlugin]:  # pragma: no cover - hook specification
        """Return an iterable of StagePlugin instances to register."""


@define(slots=True)
class StagePluginManager:
    """Coordinate registration, health checking, and stage creation."""

    context: StagePluginContext
    namespace: str = PLUGIN_NAMESPACE
    _plugin_manager: pluggy.PluginManager = field(
        factory=lambda: pluggy.PluginManager(PLUGIN_NAMESPACE), init=False
    )
    _registry: dict[str, StagePlugin] = field(factory=dict, init=False)
    _stage_index: dict[str, list[str]] = field(factory=lambda: defaultdict(list), init=False)
    _loaded: bool = field(default=False, init=False)

    def __attrs_post_init__(self) -> None:
        self._plugin_manager.add_hookspecs(StagePluginSpec)

    def load_entrypoints(self) -> None:
        """Discover and register plugins declared via entry points."""
        if self._loaded:
            return
        discovered = self._plugin_manager.load_setuptools_entrypoints(self.namespace)
        logger.debug(
            "orchestration.stage_plugins.entrypoints",
            namespace=self.namespace,
            discovered=len(discovered),
        )
        for plugin in discovered:
            self._plugin_manager.register(plugin)
        for hook in self._plugin_manager.hook.stage_plugins():  # type: ignore[attr-defined]
            for candidate in hook:
                self.register(candidate)
        self._loaded = True

    def register(self, plugin: StagePlugin) -> None:
        """Register a StagePlugin instance and index stage types."""
        metadata = plugin.metadata
        try:
            metadata = StagePluginMetadata.model_validate(metadata.model_dump())
        except ValidationError as exc:  # pragma: no cover - defensive guard
            raise StagePluginLoadError(str(exc)) from exc

        plugin.initialise(self.context)
        for stage_type in metadata.stage_types:
            canonical = stage_type.lower().strip()
            self._registry[metadata.name] = plugin
            self._stage_index[canonical].append(metadata.name)
            STAGE_PLUGIN_REGISTRATIONS.labels(
                plugin=metadata.name, stage_type=canonical
            ).inc()
        logger.info(
            "orchestration.stage_plugins.registered",
            plugin=metadata.name,
            version=metadata.version,
            stage_types=list(metadata.stage_types),
        )
        self._refresh_health(plugin)

    def _refresh_health(self, plugin: StagePlugin) -> None:
        try:
            plugin.health_check(self.context)
        except Exception as exc:
            metadata = plugin.metadata
            STAGE_PLUGIN_FAILURES.labels(
                plugin=metadata.name, stage_type="__health__"
            ).inc()
            STAGE_PLUGIN_HEALTH.labels(plugin=metadata.name).set(0)
            logger.warning(
                "orchestration.stage_plugins.unhealthy",
                plugin=metadata.name,
                error=str(exc),
            )
        else:
            STAGE_PLUGIN_HEALTH.labels(plugin=plugin.metadata.name).set(1)

    def available_stage_types(self) -> list[str]:
        return sorted(self._stage_index)

    def iter_plugins(self) -> Iterable[StagePlugin]:
        return self._registry.values()

    def get(self, name: str) -> StagePlugin | None:
        return self._registry.get(name)

    def shutdown(self) -> None:
        for plugin in list(self._registry.values()):
            try:
                plugin.cleanup(self.context)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning(
                    "orchestration.stage_plugins.cleanup_failed",
                    plugin=plugin.metadata.name,
                    error=str(exc),
                )

    @retry(  # type: ignore[misc]
        retry=retry_if_exception_type(StagePluginExecutionError),
        wait=wait_exponential(multiplier=0.2, min=0.2, max=2.0),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def create_stage(self, definition: Any) -> object:
        """Instantiate a stage for the given definition using registered plugins."""
        stage_type = str(getattr(definition, "stage_type", "")).lower()
        if stage_type not in self._stage_index:
            raise StagePluginNotAvailable(f"No stage plugin registered for '{stage_type}'")

        for plugin_name in self._stage_index[stage_type]:
            plugin = self._registry[plugin_name]
            try:
                stage = plugin.create_stage(definition, self.context)
            except Exception as exc:
                STAGE_PLUGIN_FAILURES.labels(plugin=plugin_name, stage_type=stage_type).inc()
                logger.error(
                    "orchestration.stage_plugins.create_failed",
                    plugin=plugin_name,
                    stage_type=stage_type,
                    error=str(exc),
                )
                raise StagePluginExecutionError(str(exc)) from exc
            if stage is not None:
                logger.debug(
                    "orchestration.stage_plugins.created",
                    plugin=plugin_name,
                    stage_type=stage_type,
                )
                return stage

        raise StagePluginNotAvailable(
            f"Registered plugins declined stage type '{stage_type}'"
        )


__all__ = [
    "StagePlugin",
    "StagePluginContext",
    "StagePluginError",
    "StagePluginExecutionError",
    "StagePluginLoadError",
    "StagePluginManager",
    "StagePluginMetadata",
    "StagePluginNotAvailable",
    "StagePluginSpec",
    "hookimpl",
    "hookspec",
]

