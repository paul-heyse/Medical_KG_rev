"""Simplified plugin manager placeholders."""

from __future__ import annotations

from typing import Any


class AdapterPluginError(RuntimeError):
    pass


hookimpl = lambda func: func  # type: ignore
hookspec = lambda func: func  # type: ignore


class AdapterHookSpec:
    pass


class AdapterPluginManager:
    """Minimal plugin manager used during refactoring."""

    def register(self, plugin: Any, name: str | None = None, *, entry_point: str | None = None):
        return getattr(plugin, "metadata", None)

    def unregister(self, adapter_name: str) -> None:
        return None

    def invoke(self, adapter_name: str, request: Any, *, strict: bool = True, raise_on_error: bool = False):
        raise AdapterPluginError("Adapter plugin system is unavailable in this build")


__all__ = [
    "AdapterPluginError",
    "AdapterHookSpec",
    "AdapterPluginManager",
    "hookimpl",
    "hookspec",
]
