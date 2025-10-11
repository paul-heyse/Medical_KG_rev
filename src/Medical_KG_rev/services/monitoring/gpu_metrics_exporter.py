"""GPU metrics exporter placeholder."""

from __future__ import annotations

from typing import Any, Dict


class GPUMetricsExporter:
    """Store metrics in memory so callers can inspect them."""

    def __init__(self) -> None:
        self._latest: Dict[str, Any] = {}

    def export(self, metrics: Dict[str, Any]) -> None:
        self._latest = dict(metrics)

    def latest(self) -> Dict[str, Any]:
        return dict(self._latest)


__all__ = ["GPUMetricsExporter"]
