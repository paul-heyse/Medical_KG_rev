"""Helpers for initialising the adapter plugin manager with bundled plugins."""

from __future__ import annotations

import os

from Medical_KG_rev.adapters.plugins.manager import AdapterPluginManager
from Medical_KG_rev.adapters.plugins.models import AdapterDomain

from .domains.biomedical import register_biomedical_plugins
from .domains.financial import FinancialNewsAdapterPlugin
from .domains.legal import LegalPrecedentAdapterPlugin


_MANAGER: AdapterPluginManager | None = None


def plugin_framework_enabled() -> bool:
    """Check feature flag controlling whether the plugin framework is active."""
    value = os.getenv("MK_USE_PLUGIN_FRAMEWORK", "1").lower()
    return value not in {"0", "false", "no"}


def _register_builtin(manager: AdapterPluginManager) -> None:
    register_biomedical_plugins(manager)
    manager.register(FinancialNewsAdapterPlugin())
    manager.register(LegalPrecedentAdapterPlugin())


def get_plugin_manager(refresh: bool = False) -> AdapterPluginManager:
    """Return a singleton plugin manager populated with bundled adapters."""
    global _MANAGER
    if not plugin_framework_enabled():
        raise RuntimeError("Adapter plugin framework is disabled via MK_USE_PLUGIN_FRAMEWORK")
    if _MANAGER is None or refresh:
        _MANAGER = AdapterPluginManager()
        _register_builtin(_MANAGER)
        _MANAGER.discover_entry_points()
    return _MANAGER


def list_adapters_by_domain() -> dict[AdapterDomain, tuple[str, ...]]:
    """Helper returning a mapping of domain to adapter names."""
    manager = get_plugin_manager()
    return manager.domains()


__all__ = [
    "get_plugin_manager",
    "list_adapters_by_domain",
    "plugin_framework_enabled",
]
