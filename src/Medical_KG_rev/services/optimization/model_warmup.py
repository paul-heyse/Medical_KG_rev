"""Model warm-up manager placeholder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class WarmupReport:
    requests: int
    successes: int
    failures: int


class ModelWarmupManager:
    """Provide a minimal API compatible with the legacy warm-up manager."""

    def __init__(self, config: Any | None = None) -> None:
        self.config = config

    def run(self, requests: List[Any] | None = None) -> WarmupReport:
        count = len(requests or [])
        return WarmupReport(requests=count, successes=count, failures=0)


__all__ = ["ModelWarmupManager", "WarmupReport"]
