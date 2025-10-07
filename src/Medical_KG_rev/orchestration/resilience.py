"""Resilience helpers for orchestration pipelines."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

import structlog

from Medical_KG_rev.observability.metrics import (
    record_timeout_breach,
    set_orchestration_circuit_state,
)

from .pipeline import StageFailure


logger = structlog.get_logger(__name__)


class CircuitState(str, Enum):
    """Finite state machine for circuit breakers."""

    CLOSED = "closed"
    HALF_OPEN = "half_open"
    OPEN = "open"


@dataclass(slots=True)
class CircuitBreaker:
    """Simple circuit breaker implementation with half-open recovery."""

    service: str
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    half_open_max_calls: int = 1

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _half_open_calls: int = field(default=0, init=False)

    def _transition(self, state: CircuitState) -> None:
        if self._state == state:
            return
        logger.info(
            "orchestration.circuit.transition",
            service=self.service,
            previous_state=self._state.value,
            next_state=state.value,
        )
        self._state = state
        if state is CircuitState.OPEN:
            self._opened_at = time.time()
        elif state is CircuitState.CLOSED:
            self._failure_count = 0
            self._opened_at = None
            self._half_open_calls = 0
        set_orchestration_circuit_state(self.service, state.value)

    def record_success(self) -> None:
        self._transition(CircuitState.CLOSED)

    def record_failure(self) -> None:
        self._failure_count += 1
        logger.warning(
            "orchestration.circuit.failure",
            service=self.service,
            failure_count=self._failure_count,
            threshold=self.failure_threshold,
        )
        if self._failure_count >= self.failure_threshold:
            self._transition(CircuitState.OPEN)

    def _ready_for_half_open(self) -> bool:
        if self._opened_at is None:
            return False
        return (time.time() - self._opened_at) >= self.recovery_timeout_seconds

    def _on_half_open_call(self, stage: str) -> None:
        self._half_open_calls += 1
        if self._half_open_calls > self.half_open_max_calls:
            raise StageFailure(
                "Circuit breaker half-open limit reached",
                status=503,
                stage=stage,
                error_type="circuit",
                retriable=True,
            )

    def before_call(self, stage: str) -> None:
        if self._state is CircuitState.OPEN and not self._ready_for_half_open():
            raise StageFailure(
                "Circuit breaker open",
                status=503,
                stage=stage,
                error_type="circuit",
                retriable=True,
            )
        if self._state is CircuitState.OPEN and self._ready_for_half_open():
            self._transition(CircuitState.HALF_OPEN)
        if self._state is CircuitState.HALF_OPEN:
            self._on_half_open_call(stage)

    @contextmanager
    def guard(self, stage: str) -> Iterator[None]:
        """Context manager protecting an operation with the breaker."""

        self.before_call(stage)
        try:
            yield
        except Exception:
            self.record_failure()
            raise
        else:
            self.record_success()


@dataclass(slots=True)
class TimeoutManager:
    """Helper to enforce per-stage timeout policies."""

    tolerance_ms: float = 0.0

    def ensure(
        self,
        *,
        operation: str,
        stage: str,
        duration_seconds: float,
        timeout_ms: int | None,
    ) -> None:
        if timeout_ms is None:
            return
        allowed = timeout_ms + self.tolerance_ms
        elapsed_ms = duration_seconds * 1000
        if elapsed_ms <= allowed:
            return
        logger.warning(
            "orchestration.timeout",
            operation=operation,
            stage=stage,
            timeout_ms=timeout_ms,
            duration_ms=round(elapsed_ms, 2),
        )
        record_timeout_breach(operation, stage, duration_seconds)
        raise StageFailure(
            f"Stage '{stage}' exceeded timeout",
            status=504,
            stage=stage,
            error_type="timeout",
            retriable=True,
        )


__all__ = ["CircuitBreaker", "CircuitState", "TimeoutManager"]
