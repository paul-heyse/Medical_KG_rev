"""Circuit breaker implementation for the MinerU vLLM client."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from enum import Enum

try:  # pragma: no cover - fallback if observability logging dependencies missing
    from Medical_KG_rev.utils.logging import get_logger
except Exception:  # pragma: no cover - fallback to stdlib logging
    import logging

    class _FallbackLogger:
        def __init__(self, name: str) -> None:
            self._logger = logging.getLogger(name)

        def _format(self, message: str, details: dict[str, object]) -> str:
            if not details:
                return message
            suffix = ", ".join(f"{key}={value}" for key, value in sorted(details.items()))
            return f"{message} | {suffix}"

        def debug(self, message: str, **kwargs: object) -> None:
            self._logger.debug(self._format(message, kwargs))

        def info(self, message: str, **kwargs: object) -> None:
            self._logger.info(self._format(message, kwargs))

        def warning(self, message: str, **kwargs: object) -> None:
            self._logger.warning(self._format(message, kwargs))

        def error(self, message: str, **kwargs: object) -> None:
            self._logger.error(self._format(message, kwargs))

    def get_logger(name: str) -> _FallbackLogger:  # type: ignore[override]
        return _FallbackLogger(name)


try:  # pragma: no cover - metrics import may pull optional deps
    from Medical_KG_rev.observability.metrics import MINERU_VLLM_CIRCUIT_BREAKER_STATE
except Exception:  # pragma: no cover - fallback gauge when metrics unavailable

    class _FallbackGauge:
        def set(self, value: float) -> None:  # type: ignore[override]
            return None

    MINERU_VLLM_CIRCUIT_BREAKER_STATE = _FallbackGauge()

logger = get_logger(__name__)


class CircuitState(Enum):
    """Enumeration of supported circuit breaker states."""

    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is open and execution is blocked."""


class CircuitBreaker:
    """Asynchronous circuit breaker guarding vLLM client invocations."""

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be > 0")
        if success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        self._lock = asyncio.Lock()

        MINERU_VLLM_CIRCUIT_BREAKER_STATE.set(self.state.value)
        logger.info(
            "mineru.vllm.circuit_breaker.initialised",
            state=self.state.name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
        )

    async def can_execute(self) -> bool:
        """Return ``True`` if execution is permitted."""

        async with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                if self.last_failure_time is None:
                    return False

                elapsed = datetime.now(timezone.utc) - self.last_failure_time
                if elapsed >= timedelta(seconds=self.recovery_timeout):
                    self._transition_to_half_open()
                    return True

                return False

            # HALF_OPEN
            return True

    async def record_success(self) -> None:
        """Record a successful execution."""

        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                logger.debug(
                    "mineru.vllm.circuit_breaker.success",
                    success_count=self.success_count,
                    threshold=self.success_threshold,
                )
                if self.success_count >= self.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0

    async def record_failure(self) -> None:
        """Record a failed execution."""

        async with self._lock:
            self.last_failure_time = datetime.now(timezone.utc)

            if self.state == CircuitState.CLOSED:
                self.failure_count += 1
                logger.warning(
                    "mineru.vllm.circuit_breaker.failure",
                    failure_count=self.failure_count,
                    threshold=self.failure_threshold,
                )
                if self.failure_count >= self.failure_threshold:
                    self._transition_to_open()

            elif self.state == CircuitState.HALF_OPEN:
                logger.warning("mineru.vllm.circuit_breaker.reopen")
                self._transition_to_open()

    def _transition_to_open(self) -> None:
        self.state = CircuitState.OPEN
        self.success_count = 0
        MINERU_VLLM_CIRCUIT_BREAKER_STATE.set(self.state.value)
        logger.error(
            "mineru.vllm.circuit_breaker.opened",
            failure_count=self.failure_count,
            recovery_timeout=self.recovery_timeout,
        )

    def _transition_to_half_open(self) -> None:
        self.state = CircuitState.HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        MINERU_VLLM_CIRCUIT_BREAKER_STATE.set(self.state.value)
        logger.info("mineru.vllm.circuit_breaker.half_open")

    def _transition_to_closed(self) -> None:
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        MINERU_VLLM_CIRCUIT_BREAKER_STATE.set(self.state.value)
        logger.info("mineru.vllm.circuit_breaker.closed")


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitState",
]
