"""Stage plugin manager for orchestration pipeline."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import pluggy
import structlog

from Medical_KG_rev.observability.metrics import (
    STAGE_PLUGIN_FAILURES,
    STAGE_PLUGIN_HEALTH,
    STAGE_PLUGIN_REGISTRATIONS,
)
from Medical_KG_rev.orchestration.stages.plugins import (
    StagePlugin,
    StagePluginHealth,
    StagePluginRegistration,
)

logger = structlog.get_logger(__name__)


class StagePluginManager:
    """Manages stage plugins for the orchestration pipeline."""

    def __init__(self) -> None:
        """Initialize the plugin manager."""
        self.logger = logger
        self._plugins: dict[str, StagePlugin] = {}
        self._registrations: dict[str, StagePluginRegistration] = {}
        self._plugin_hook = pluggy.PluginManager("stage_plugin")
        self._initialized = False

    def register_plugin(self, plugin: StagePlugin) -> None:
        """Register a stage plugin."""
        try:
            self._plugins[plugin.name] = plugin
            self._plugin_hook.register(plugin)

            # Get registrations from plugin
            registrations = plugin.registrations(None)
            for registration in registrations:
                self._registrations[registration.stage_type] = registration

            self.logger.info(f"Registered stage plugin: {plugin.name}")

        except Exception as exc:
            self.logger.error(f"Failed to register plugin {plugin.name}: {exc}")
            raise exc

    def unregister_plugin(self, plugin_name: str) -> None:
        """Unregister a stage plugin."""
        try:
            if plugin_name in self._plugins:
                plugin = self._plugins[plugin_name]
                self._plugin_hook.unregister(plugin)

                # Remove registrations
                registrations = plugin.registrations(None)
                for registration in registrations:
                    if registration.stage_type in self._registrations:
                        del self._registrations[registration.stage_type]

                del self._plugins[plugin_name]
                self.logger.info(f"Unregistered stage plugin: {plugin_name}")
            else:
                self.logger.warning(f"Plugin not found: {plugin_name}")

        except Exception as exc:
            self.logger.error(f"Failed to unregister plugin {plugin_name}: {exc}")
            raise exc

    def get_plugin(self, plugin_name: str) -> StagePlugin | None:
        """Get a registered plugin by name."""
        return self._plugins.get(plugin_name)

    def list_plugins(self) -> list[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())

    def get_registration(self, stage_type: str) -> StagePluginRegistration | None:
        """Get stage registration by type."""
        return self._registrations.get(stage_type)

    def list_stage_types(self) -> list[str]:
        """List all registered stage types."""
        return list(self._registrations.keys())

    def create_stage(self, stage_type: str, config: dict[str, Any]) -> Any:
        """Create a stage instance."""
        try:
            registration = self.get_registration(stage_type)
            if not registration:
                raise ValueError(f"Stage type not found: {stage_type}")

            # Create stage using builder
            stage = registration.builder(config)

            self.logger.info(f"Created stage: {stage_type}")
            return stage

        except Exception as exc:
            self.logger.error(f"Failed to create stage {stage_type}: {exc}")
            raise exc

    def health_check(self) -> dict[str, StagePluginHealth]:
        """Check health of all registered plugins."""
        health_status = {}

        for plugin_name, plugin in self._plugins.items():
            try:
                health = plugin.health_check()
                health_status[plugin_name] = health

                # Record health metric
                STAGE_PLUGIN_HEALTH.labels(
                    plugin_name=plugin_name,
                    status=health.status,
                ).set(1.0 if health.status == "ok" else 0.0)

            except Exception as exc:
                self.logger.error(f"Health check failed for plugin {plugin_name}: {exc}")
                health_status[plugin_name] = StagePluginHealth(
                    status="error",
                    detail=f"Health check failed: {exc}",
                    timestamp=0.0,
                )

                # Record failure metric
                STAGE_PLUGIN_FAILURES.labels(
                    plugin_name=plugin_name,
                    error_type="health_check_failed",
                ).inc()

        return health_status

    def get_plugin_capabilities(self, plugin_name: str) -> list[str]:
        """Get capabilities of a plugin."""
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            return []

        capabilities = []
        registrations = plugin.registrations(None)
        for registration in registrations:
            capabilities.extend(registration.capabilities)

        return list(set(capabilities))  # Remove duplicates

    def get_stage_capabilities(self, stage_type: str) -> list[str]:
        """Get capabilities of a stage type."""
        registration = self.get_registration(stage_type)
        if not registration:
            return []

        return list(registration.capabilities)

    def validate_stage_config(self, stage_type: str, config: dict[str, Any]) -> bool:
        """Validate stage configuration."""
        try:
            registration = self.get_registration(stage_type)
            if not registration:
                return False

            # Try to create stage to validate config
            registration.builder(config)
            return True

        except Exception as exc:
            self.logger.warning(f"Stage config validation failed for {stage_type}: {exc}")
            return False

    def initialize(self) -> None:
        """Initialize the plugin manager."""
        if self._initialized:
            return

        try:
            # Register built-in plugins
            self._register_builtin_plugins()

            # Initialize plugins
            for plugin in self._plugins.values():
                if hasattr(plugin, 'initialize'):
                    plugin.initialize()

            self._initialized = True
            self.logger.info("Stage plugin manager initialized")

        except Exception as exc:
            self.logger.error(f"Failed to initialize plugin manager: {exc}")
            raise exc

    def _register_builtin_plugins(self) -> None:
        """Register built-in plugins."""
        try:
            from .plugins.builtin import get_builtin_plugins

            plugins = get_builtin_plugins()
            for plugin in plugins:
                self.register_plugin(plugin)

        except ImportError:
            self.logger.warning("Built-in plugins not available")

    def shutdown(self) -> None:
        """Shutdown the plugin manager."""
        try:
            # Cleanup plugins
            for plugin in self._plugins.values():
                if hasattr(plugin, 'cleanup'):
                    plugin.cleanup()

            # Clear registrations
            self._registrations.clear()
            self._plugins.clear()

            self._initialized = False
            self.logger.info("Stage plugin manager shutdown")

        except Exception as exc:
            self.logger.error(f"Failed to shutdown plugin manager: {exc}")

    def get_stats(self) -> dict[str, Any]:
        """Get plugin manager statistics."""
        return {
            "plugin_count": len(self._plugins),
            "stage_type_count": len(self._registrations),
            "initialized": self._initialized,
            "plugins": list(self._plugins.keys()),
            "stage_types": list(self._registrations.keys()),
        }


# Global plugin manager instance
_plugin_manager: StagePluginManager | None = None


def get_plugin_manager() -> StagePluginManager:
    """Get the global plugin manager instance."""
    global _plugin_manager

    if _plugin_manager is None:
        _plugin_manager = StagePluginManager()
        _plugin_manager.initialize()

    return _plugin_manager


def create_plugin_manager() -> StagePluginManager:
    """Create a new plugin manager instance."""
    return StagePluginManager()
