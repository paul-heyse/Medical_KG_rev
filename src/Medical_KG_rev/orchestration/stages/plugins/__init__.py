"""Stage plugin implementations bundled with the orchestration runtime.

This package provides built-in stage plugin implementations. The plugin framework
classes (StagePlugin, StagePluginManager) are imported from the sibling plugins.py module.
"""

from .builtin import CoreStagePlugin, PdfTwoPhasePlugin



# Define minimal versions of framework classes locally to avoid circular dependencies
class StagePluginBuildError(Exception):
    """Raised when a stage plugin fails to build."""

    pass


class StagePluginLookupError(Exception):
    """Raised when a stage plugin cannot be found."""

    pass


class StagePlugin:
    """Base class for stage plugins."""

    def __init__(self, plugin_name: str, dependencies: tuple[str, ...] = ()):
        self.name = plugin_name
        self.dependencies = dependencies

    def registrations(self, resources):
        """Return stage plugin registrations."""
        return ()

    def cleanup(self):
        """Cleanup plugin resources."""
        pass

    def health_check(self):
        """Check plugin health."""
        return StagePluginHealth(status="ok", detail=self.name, timestamp=0.0)

    def create_registration(self, stage_type: str, builder, capabilities: tuple[str, ...]):
        """Create a stage plugin registration."""
        return StagePluginRegistration(
            stage_type=stage_type, builder=builder, capabilities=capabilities
        )


class StagePluginManager:
    """Minimal stage plugin manager to avoid circular dependencies."""

    def __init__(self, resources):
        self.resources = resources
        self._plugins = {}

    def register(self, plugin: StagePlugin):
        """Register a stage plugin."""
        self._plugins[plugin.name] = plugin

    def unregister(self, plugin: StagePlugin):
        """Unregister a stage plugin."""
        if plugin.name in self._plugins:
            plugin.cleanup()
            del self._plugins[plugin.name]

    def build_stage(self, definition):
        """Build a stage using registered plugins."""
        for plugin in self._plugins.values():
            registrations = plugin.registrations(self.resources)
            for registration in registrations:
                if registration.stage_type == definition.type:
                    return registration.builder(definition, self.resources)
        return {}

    def available_stage_types(self):
        """Get available stage types."""
        types = set()
        for plugin in self._plugins.values():
            registrations = plugin.registrations(self.resources)
            for registration in registrations:
                types.add(f"{plugin.name}.{registration.stage_type}")
        return tuple(types)

    def check_health(self):
        """Check health of all plugins."""
        health = {}
        for plugin in self._plugins.values():
            registrations = plugin.registrations(self.resources)
            for registration in registrations:
                key = f"{plugin.name}.{registration.stage_type}"
                health[key] = plugin.health_check()
        return health

    def describe_plugins(self):
        """Describe all registered plugins."""
        descriptions = []
        for plugin in self._plugins.values():
            registrations = plugin.registrations(self.resources)
            for registration in registrations:
                descriptions.append(
                    {"name": f"{plugin.name}.{registration.stage_type}", "status": "initialized"}
                )
        return descriptions


class StagePluginHealth:
    """Stage plugin health status."""

    def __init__(self, status: str, detail: str, timestamp: float):
        self.status = status
        self.detail = detail
        self.timestamp = timestamp


class StagePluginRegistration:
    """Stage plugin registration."""

    def __init__(self, stage_type: str, builder, capabilities: tuple[str, ...]):
        self.stage_type = stage_type
        self.builder = builder
        self.capabilities = capabilities


class StagePluginResources:
    """Resources available to stage plugins."""

    def __init__(self, adapter_manager, pipeline_resource):
        self.adapter_manager = adapter_manager
        self.pipeline_resource = pipeline_resource


__all__ = [
    "CoreStagePlugin",
    "PdfTwoPhasePlugin",
    "StagePlugin",
    "StagePluginBuildError",
    "StagePluginHealth",
    "StagePluginLookupError",
    "StagePluginManager",
    "StagePluginRegistration",
    "StagePluginResources",
]
