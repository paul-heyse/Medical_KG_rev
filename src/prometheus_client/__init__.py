"""Minimal Prometheus client shim for environments without the real dependency."""

from __future__ import annotations

from typing import Any

CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


class _MetricHandle:
    def __init__(self, store: dict[tuple[Any, ...], float], key: tuple[Any, ...]) -> None:
        self._store = store
        self._key = key

    def inc(self, amount: float = 1.0) -> None:
        self._store[self._key] = self._store.get(self._key, 0.0) + float(amount)

    def observe(self, value: float) -> None:
        self._store[self._key] = float(value)

    def set(self, value: float) -> None:
        self._store[self._key] = float(value)


class _Metric:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._store: dict[tuple[Any, ...], float] = {}

    def labels(self, *args: Any, **kwargs: Any) -> _MetricHandle:
        key = tuple(args) + tuple(kwargs.items())
        return _MetricHandle(self._store, key)

    def inc(self, amount: float = 1.0) -> None:
        self._store[(None,)] = self._store.get((None,), 0.0) + float(amount)

    def observe(self, value: float) -> None:
        self._store[(None,)] = float(value)

    def set(self, value: float) -> None:
        self._store[(None,)] = float(value)


class Counter(_Metric):
    pass


class Gauge(_Metric):
    pass


class Histogram(_Metric):
    pass


class CollectorRegistry:
    """Minimal collector registry shim."""

    def __init__(self) -> None:
        self._collectors: dict[str, Any] = {}

    def register(self, collector: Any) -> None:
        """Register a collector."""
        pass

    def unregister(self, collector: Any) -> None:
        """Unregister a collector."""
        pass


def generate_latest() -> bytes:
    return b""
