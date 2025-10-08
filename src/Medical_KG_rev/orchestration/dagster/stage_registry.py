"""Stage metadata and plugin registry for Dagster orchestration stages."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from inspect import signature
from importlib import metadata
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import structlog

from Medical_KG_rev.orchestration.dagster.configuration import (
    PipelineTopologyConfig,
    StageDefinition,
)

logger = structlog.get_logger(__name__)


StageBuilder = Callable[[PipelineTopologyConfig | None, StageDefinition], object]


class StageRegistryError(RuntimeError):
    """Raised when stage metadata registration or lookup fails."""


@dataclass(slots=True, frozen=True)
class StageMetadata:
    """Metadata describing how a stage integrates with the runtime state."""

    stage_type: str
    state_key: str | Sequence[str] | None
    output_handler: Callable[[dict[str, Any], str, Any], None]
    output_counter: Callable[[Any], int]
    description: str
    dependencies: Sequence[str] = field(default_factory=tuple)

    _IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __post_init__(self) -> None:
        if not isinstance(self.stage_type, str) or not self.stage_type.strip():
            raise StageRegistryError("Stage type must be a non-empty string")
        if not callable(self.output_handler):
            raise StageRegistryError(
                f"Stage '{self.stage_type}' output_handler must be callable"
            )
        if not callable(self.output_counter):
            raise StageRegistryError(
                f"Stage '{self.stage_type}' output_counter must be callable"
            )
        if not isinstance(self.description, str) or not self.description.strip():
            raise StageRegistryError(
                f"Stage '{self.stage_type}' description must be a non-empty string"
            )
        for dependency in self.dependencies:
            if not isinstance(dependency, str) or not dependency.strip():
                raise StageRegistryError(
                    f"Stage '{self.stage_type}' dependency '{dependency}' is invalid"
                )
        self._validate_state_keys(self.state_key)

    @property
    def state_keys(self) -> Sequence[str] | None:
        if self.state_key is None:
            return None
        if isinstance(self.state_key, str):
            return (self.state_key,)
        return tuple(self.state_key)

    def result_snapshot(self, state: Mapping[str, Any], output: Any) -> Any:
        keys = self.state_keys
        if keys is None:
            return output
        if len(keys) == 1:
            return state.get(keys[0])
        return {key: state.get(key) for key in keys}

    @classmethod
    def _validate_state_keys(cls, state_key: str | Sequence[str] | None) -> None:
        if state_key is None:
            return
        keys = (state_key,) if isinstance(state_key, str) else tuple(state_key)
        if not keys:
            raise StageRegistryError("state_key collection cannot be empty")
        for key in keys:
            if not isinstance(key, str) or not key:
                raise StageRegistryError("state_key entries must be non-empty strings")
            if not cls._IDENTIFIER_PATTERN.match(key):
                raise StageRegistryError(
                    f"Invalid state key '{key}'; must be a valid Python identifier"
                )


@dataclass(slots=True, frozen=True)
class StageRegistration:
    """Combination of metadata and builder used for registration."""

    metadata: StageMetadata
    builder: StageBuilder

    def __post_init__(self) -> None:
        if not callable(self.builder):
            raise StageRegistryError(
                f"Stage '{self.metadata.stage_type}' builder must be callable"
            )


class StagePlugin(Protocol):
    """Protocol for plugin registration callables."""

    def __call__(self) -> StageRegistration | Iterable[StageRegistration]:
        """Return one or more stage registrations."""


def discover_stages(
    group: str = "medical_kg.orchestration.stages",
) -> Iterable[StagePlugin]:
    """Yield plugin callables discovered via entry points."""

    try:
        entry_points = metadata.entry_points()
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("dagster.stage.plugins.discovery_failed", error=str(exc))
        return []
    selected = entry_points.select(group=group) if hasattr(entry_points, "select") else []
    plugins: list[StagePlugin] = []
    for entry_point in selected:
        try:
            loaded = entry_point.load()
        except Exception as exc:  # pragma: no cover - discovery guard
            logger.warning(
                "dagster.stage.plugins.load_failed",
                entry_point=entry_point.name,
                error=str(exc),
            )
            continue
        if not callable(loaded):
            logger.warning(
                "dagster.stage.plugins.invalid",
                entry_point=entry_point.name,
                reason="not callable",
            )
            continue
        plugins.append(loaded)  # type: ignore[return-value]
    return plugins


class StageRegistry:
    """Registry responsible for managing stage metadata and builders."""

    def __init__(
        self,
        *,
        plugin_loader: Callable[[], Iterable[StagePlugin]] | None = None,
    ) -> None:
        self._metadata: dict[str, StageMetadata] = {}
        self._builders: dict[str, StageBuilder] = {}
        self._plugin_loader = plugin_loader or (lambda: discover_stages())

    def register(self, registration: StageRegistration, *, replace: bool = False) -> None:
        stage_type = registration.metadata.stage_type
        if stage_type in self._metadata and not replace:
            raise StageRegistryError(
                f"Stage '{stage_type}' is already registered"
            )
        self._metadata[stage_type] = registration.metadata
        builder = registration.builder
        try:
            param_count = len(signature(builder).parameters)  # type: ignore[arg-type]
        except Exception:
            param_count = 0
        if param_count <= 1:
            original = builder

            def _wrapped(topology: PipelineTopologyConfig | None, definition: StageDefinition):
                return original(definition)  # type: ignore[misc]

            builder = _wrapped  # type: ignore[assignment]
        self._builders[stage_type] = builder
        logger.debug(
            "dagster.stage.registry.registered",
            stage_type=stage_type,
            description=registration.metadata.description,
        )

    def register_stage(
        self,
        *,
        metadata: StageMetadata,
        builder: StageBuilder,
        replace: bool = False,
    ) -> None:
        registration = StageRegistration(metadata=metadata, builder=builder)
        self.register(registration, replace=replace)

    def get_metadata(self, stage_type: str) -> StageMetadata:
        try:
            return self._metadata[stage_type]
        except KeyError as exc:  # pragma: no cover - guard
            raise StageRegistryError(f"Unknown stage type '{stage_type}'") from exc

    def get_builder(self, stage_type: str) -> StageBuilder:
        try:
            return self._builders[stage_type]
        except KeyError as exc:  # pragma: no cover - guard
            raise StageRegistryError(f"Unknown stage type '{stage_type}'") from exc

    def load_plugins(self) -> list[str]:
        loaded: list[str] = []
        for plugin in self._plugin_loader():
            try:
                registrations = plugin()
            except Exception as exc:
                logger.warning(
                    "dagster.stage.plugins.registration_failed",
                    plugin=_plugin_name(plugin),
                    error=str(exc),
                )
                continue
            if isinstance(registrations, StageRegistration):
                registrations = [registrations]
            elif isinstance(registrations, Iterable):
                registrations = list(registrations)
            else:
                logger.warning(
                    "dagster.stage.plugins.invalid_return",
                    plugin=_plugin_name(plugin),
                )
                continue
            for registration in registrations:
                try:
                    self.register(registration)
                except StageRegistryError as exc:
                    logger.warning(
                        "dagster.stage.plugins.registration_conflict",
                        plugin=_plugin_name(plugin),
                        stage_type=registration.metadata.stage_type,
                        error=str(exc),
                    )
                    continue
                loaded.append(registration.metadata.stage_type)
        return loaded

    def stage_types(self) -> list[str]:
        return sorted(self._metadata)


def _plugin_name(plugin: StagePlugin) -> str:
    if hasattr(plugin, "__qualname__"):
        return str(getattr(plugin, "__qualname__"))
    if hasattr(plugin, "__name__"):
        return str(getattr(plugin, "__name__"))
    return plugin.__class__.__name__


__all__ = [
    "StageBuilder",
    "StageMetadata",
    "StagePlugin",
    "StageRegistration",
    "StageRegistry",
    "StageRegistryError",
    "discover_stages",
]
