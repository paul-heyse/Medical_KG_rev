"""Simple registry for adapter discovery."""

from __future__ import annotations

from collections.abc import Iterable

from .base import BaseAdapter


class AdapterRegistry:
    """Singleton-style registry."""

    def __init__(self) -> None:
        self._registry: dict[str, type[BaseAdapter]] = {}

    def register(self, adapter_cls: type[BaseAdapter]) -> None:
        if adapter_cls.__name__ in self._registry:
            raise ValueError(f"Adapter '{adapter_cls.__name__}' already registered")
        self._registry[adapter_cls.__name__] = adapter_cls

    def create(self, name: str, *args, **kwargs) -> BaseAdapter:
        try:
            cls = self._registry[name]
        except KeyError as exc:
            raise KeyError(f"Adapter '{name}' is not registered") from exc
        return cls(*args, **kwargs)

    def registered(self) -> Iterable[str]:
        return sorted(self._registry.keys())


registry = AdapterRegistry()
