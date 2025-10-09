"""Simple circuit breaker for rerankers."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from time import monotonic


@dataclass(slots=True)
class CircuitState:
    failures: int = 0
    opened_at: float | None = None


@dataclass(slots=True)
class CircuitBreaker:
    failure_threshold: int = 5
    reset_timeout: float = 30.0
    _state: MutableMapping[str, CircuitState] = field(default_factory=dict)

    def record_success(self, key: str) -> None:
        self._state.pop(key, None)

    def record_failure(self, key: str) -> None:
        state = self._state.setdefault(key, CircuitState())
        state.failures += 1
        if state.failures >= self.failure_threshold:
            state.opened_at = monotonic()

    def can_execute(self, key: str) -> bool:
        state = self._state.get(key)
        if state is None or state.opened_at is None:
            return True
        if monotonic() - state.opened_at > self.reset_timeout:
            self._state.pop(key, None)
            return True
        return False

    def state(self, key: str) -> str:
        if self.can_execute(key):
            return "closed"
        return "open"
