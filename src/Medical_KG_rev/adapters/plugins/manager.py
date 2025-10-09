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
from Medical_KG_rev.adapters.plugins.domains import DomainAdapterRegistry

from .errors import AdapterPluginError
from .pipeline import AdapterExecutionState, AdapterPipelineFactory
from .runtime import AdapterExecutionPlan, AdapterInvocationResult, RegisteredAdapter

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


class AdapterPluginManager:
    """Wrapper around :class:`pluggy.PluginManager` for adapter lifecycle."""

    def __init__(self, project_name: str = "medical_kg.adapters") -> None:
        self._pm = pluggy.PluginManager(project_name)
        self._pm.add_hookspecs(AdapterHookSpec)
        self._adapters: dict[str, RegisteredAdapter] = {}
        self._project_name = project_name
        self._registry = DomainAdapterRegistry()
        self._pipeline_factory = AdapterPipelineFactory()

    # ------------------------------------------------------------------
    # Registration & discovery
    # ------------------------------------------------------------------
    def register(
        self,
        plugin: Any,
        name: str | None = None,
        *,
        entry_point: str | None = None,
    ) -> AdapterMetadata:
        """Register a plugin object and cache its metadata."""

        registration_name = name or getattr(plugin, "__name__", plugin.__class__.__name__)
        self._pm.register(plugin, name=registration_name)
        metadata = self._resolve_metadata(plugin)
        if entry_point:
            metadata = metadata.model_copy(update={"entry_point": entry_point})
        adapter_name = metadata.name
        pipeline = self._pipeline_factory.build(plugin, metadata)
        domain_metadata = self._registry.register(metadata)
        plan = AdapterExecutionPlan(pipeline)
        self._adapters[adapter_name] = RegisteredAdapter(
            plugin=plugin,
            metadata=metadata,
            plan=plan,
            domain_metadata=domain_metadata,
        )
        return metadata

    def unregister(self, adapter_name: str) -> None:
        registered = self._adapters.pop(adapter_name, None)
        if registered is None:
            raise AdapterPluginError(f"Adapter '{adapter_name}' is not registered")
        self._pm.unregister(registered.plugin)
        self._registry.unregister(adapter_name)

    def discover_entry_points(self, group: str = "medical_kg.adapters") -> list[AdapterMetadata]:
        """Discover adapters declared via Python entry points."""

        discovered: list[AdapterMetadata] = []
        for entry_point in metadata.entry_points().select(group=group):
            plugin = entry_point.load()
            plugin_instance = plugin() if callable(plugin) else plugin
            meta = self.register(plugin_instance, entry_point=entry_point.value)
            discovered.append(meta)
        return discovered

    # ------------------------------------------------------------------
    # Metadata querying
    # ------------------------------------------------------------------
    def get_metadata(self, adapter_name: str) -> AdapterMetadata:
        try:
            return self._adapters[adapter_name].metadata
        except KeyError as exc:  # pragma: no cover - defensive
            raise AdapterPluginError(f"Adapter '{adapter_name}' is not registered") from exc

    def list_metadata(self, domain: AdapterDomain | None = None) -> list[AdapterMetadata]:
        if domain is None:
            values: Iterable[AdapterMetadata] = (
                registered.metadata for registered in self._adapters.values()
            )
        else:
            values = self._registry.list(domain)
        return sorted(values, key=lambda meta: meta.name)

    def domains(self) -> dict[AdapterDomain, tuple[str, ...]]:
        """Return mapping of domains to registered adapter names."""

        return {domain: names for domain, names in self._registry.domains().items()}

    # ------------------------------------------------------------------
    # Adapter lifecycle helpers
    # ------------------------------------------------------------------
    def invoke(
        self,
        adapter_name: str,
        request: AdapterRequest,
        *,
        strict: bool = True,
        raise_on_error: bool = False,
    ) -> AdapterInvocationResult:
        registered = self._get_registered(adapter_name)
        context = registered.new_context(request)
        error: AdapterPluginError | None = None
        try:
            context = registered.plan.pipeline.execute(context)
            if strict:
                context.raise_for_validation()
            context.record_success()
        except AdapterPluginError as exc:
            context.record_failure(exc)
            error = exc

        result = registered.build_result(context, strict=strict, error=error)

        if error is not None and raise_on_error:
            raise error

        return result

    def execute(
        self,
        adapter_name: str,
        request: AdapterRequest,
        *,
        strict: bool = True,
    ) -> AdapterExecutionState:
        result = self.invoke(adapter_name, request, strict=strict, raise_on_error=True)
        return result.context

    def run(
        self,
        adapter_name: str,
        request: AdapterRequest,
        *,
        strict: bool = True,
    ) -> AdapterResponse:
        result = self.invoke(adapter_name, request, strict=strict, raise_on_error=True)
        return result.context.ensure_response()

    def check_health(self, adapter_name: str) -> bool:
        registered = self._get_registered(adapter_name)
        return bool(self._call_hook(registered.plugin, "health_check"))

    def estimate_cost(self, adapter_name: str, request: AdapterRequest) -> AdapterCostEstimate:
        registered = self._get_registered(adapter_name)
        return self._call_hook(registered.plugin, "estimate_cost", request)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_metadata(self, plugin: Any) -> AdapterMetadata:
        metadata_getter: Callable[[], AdapterMetadata] | None = getattr(
            plugin, "get_metadata", None
        )
        if metadata_getter is None:
            raise AdapterPluginError("Adapter plugins must define a 'get_metadata' method")
        metadata = metadata_getter()
        if not isinstance(metadata, AdapterMetadata):
            raise AdapterPluginError("Adapter metadata must be an AdapterMetadata instance")
        return metadata

    def _get_registered(self, adapter_name: str) -> RegisteredAdapter:
        try:
            return self._adapters[adapter_name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AdapterPluginError(f"Adapter '{adapter_name}' is not registered") from exc

    def _call_hook(self, plugin: Any, hook_name: str, *args: Any) -> Any:
        hook = getattr(plugin, hook_name, None)
        if hook is None:
            raise AdapterPluginError(f"Plugin '{plugin}' does not implement hook '{hook_name}'")
        return hook(*args)
