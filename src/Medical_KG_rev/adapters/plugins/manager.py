"""Adapter plugin manager built on top of Pluggy."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import metadata
from typing import Any, Callable

import pluggy

from Medical_KG_rev.adapters.plugins.models import (
    AdapterMetadata,
    AdapterRequest,
    AdapterResponse,
    AdapterDomain,
    ValidationOutcome,
    AdapterCostEstimate,
)

hookspec = pluggy.HookspecMarker("medical_kg.adapters")
hookimpl = pluggy.HookimplMarker("medical_kg.adapters")


class AdapterHookSpec:
    """Hook specification for adapter plugins."""

    @hookspec
    def get_metadata(self) -> AdapterMetadata:
        """Return metadata describing the adapter."""

    @hookspec
    def fetch(self, request: AdapterRequest) -> AdapterResponse:
        """Fetch raw payloads from an upstream data source."""

    @hookspec
    def parse(self, response: AdapterResponse, request: AdapterRequest) -> AdapterResponse:
        """Parse raw payloads into the canonical response envelope."""

    @hookspec
    def validate(self, response: AdapterResponse, request: AdapterRequest) -> ValidationOutcome:
        """Validate canonical payloads prior to downstream processing."""

    @hookspec
    def health_check(self) -> bool:
        """Return True when the adapter is ready to serve traffic."""

    @hookspec
    def estimate_cost(self, request: AdapterRequest) -> AdapterCostEstimate:
        """Estimate upstream cost (API calls, latency) for the provided request."""


class AdapterPluginError(RuntimeError):
    """Raised when adapter plugin operations fail."""


class AdapterPluginManager:
    """Wrapper around :class:`pluggy.PluginManager` for adapter lifecycle."""

    def __init__(self, project_name: str = "medical_kg.adapters") -> None:
        self._pm = pluggy.PluginManager(project_name)
        self._pm.add_hookspecs(AdapterHookSpec)
        self._adapters: dict[str, Any] = {}
        self._metadata: dict[str, AdapterMetadata] = {}
        self._project_name = project_name

    # ------------------------------------------------------------------
    # Registration & discovery
    # ------------------------------------------------------------------
    def register(self, plugin: Any, name: str | None = None) -> AdapterMetadata:
        """Register a plugin object and cache its metadata."""

        registration_name = name or getattr(plugin, "__name__", plugin.__class__.__name__)
        self._pm.register(plugin, name=registration_name)
        metadata = self._resolve_metadata(plugin)
        adapter_name = metadata.name
        self._adapters[adapter_name] = plugin
        self._metadata[adapter_name] = metadata
        return metadata

    def unregister(self, adapter_name: str) -> None:
        plugin = self._adapters.pop(adapter_name, None)
        if plugin is None:
            raise AdapterPluginError(f"Adapter '{adapter_name}' is not registered")
        self._metadata.pop(adapter_name, None)
        self._pm.unregister(plugin)

    def discover_entry_points(self, group: str = "medical_kg.adapters") -> list[AdapterMetadata]:
        """Discover adapters declared via Python entry points."""

        discovered: list[AdapterMetadata] = []
        for entry_point in metadata.entry_points().select(group=group):
            plugin = entry_point.load()
            plugin_instance = plugin() if callable(plugin) else plugin
            meta = self.register(plugin_instance)
            meta.entry_point = entry_point.value
            discovered.append(meta)
        return discovered

    # ------------------------------------------------------------------
    # Metadata querying
    # ------------------------------------------------------------------
    def get_metadata(self, adapter_name: str) -> AdapterMetadata:
        try:
            return self._metadata[adapter_name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AdapterPluginError(f"Adapter '{adapter_name}' is not registered") from exc

    def list_metadata(self, domain: AdapterDomain | None = None) -> list[AdapterMetadata]:
        values: Iterable[AdapterMetadata] = self._metadata.values()
        if domain is not None:
            values = (meta for meta in values if meta.domain == domain)
        return sorted(values, key=lambda meta: meta.name)

    # ------------------------------------------------------------------
    # Adapter lifecycle helpers
    # ------------------------------------------------------------------
    def run(self, adapter_name: str, request: AdapterRequest) -> AdapterResponse:
        plugin = self._get_plugin(adapter_name)
        response = self._call_single(plugin, "fetch", request)
        response = self._call_single(plugin, "parse", response, request)
        outcome = self._call_single(plugin, "validate", response, request)
        if not outcome.valid:
            raise AdapterPluginError(
                f"Validation failed for adapter '{adapter_name}': {', '.join(outcome.errors)}"
            )
        return response

    def check_health(self, adapter_name: str) -> bool:
        plugin = self._get_plugin(adapter_name)
        return bool(self._call_single(plugin, "health_check"))

    def estimate_cost(self, adapter_name: str, request: AdapterRequest) -> AdapterCostEstimate:
        plugin = self._get_plugin(adapter_name)
        return self._call_single(plugin, "estimate_cost", request)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_metadata(self, plugin: Any) -> AdapterMetadata:
        metadata_getter: Callable[[], AdapterMetadata] | None = getattr(plugin, "get_metadata", None)
        if metadata_getter is None:
            raise AdapterPluginError("Adapter plugins must define a 'get_metadata' method")
        metadata = metadata_getter()
        if not isinstance(metadata, AdapterMetadata):
            raise AdapterPluginError("Adapter metadata must be an AdapterMetadata instance")
        return metadata

    def _get_plugin(self, adapter_name: str) -> Any:
        try:
            return self._adapters[adapter_name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AdapterPluginError(f"Adapter '{adapter_name}' is not registered") from exc

    def _call_single(self, plugin: Any, hook_name: str, *args: Any) -> Any:
        hook = getattr(plugin, hook_name, None)
        if hook is None:
            raise AdapterPluginError(f"Plugin '{plugin}' does not implement hook '{hook_name}'")
        return hook(*args)
