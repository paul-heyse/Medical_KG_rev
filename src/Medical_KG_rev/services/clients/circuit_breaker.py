"""Circuit breaker implementation for resilient service calls."""

from collections.abc import Callable
from enum import Enum
from typing import Any, Optional, TypeVar
import time

from prometheus_client import Counter, Gauge


T = TypeVar("T")

# Circuit breaker metrics
CIRCUIT_BREAKER_STATE_CHANGES = Counter(
    "circuit_breaker_state_changes_total",
    "Total number of circuit breaker state changes",
    ["service", "from_state", "to_state"],
)

CIRCUIT_BREAKER_REQUESTS = Counter(
    "circuit_breaker_requests_total",
    "Total number of circuit breaker requests",
    ["service", "state", "result"],
)

CIRCUIT_BREAKER_STATE_GAUGE = Gauge(
    "circuit_breaker_state", "Current circuit breaker state", ["service"]
)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker implementation."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        name: str = "default",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name

        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = CircuitBreakerState.CLOSED

        CIRCUIT_BREAKER_STATE_GAUGE.labels(service=name).set(0)  # CLOSED = 0

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection."""
        if self._state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._state = CircuitBreakerState.HALF_OPEN
                CIRCUIT_BREAKER_STATE_GAUGE.labels(service=self.name).set(2)  # HALF_OPEN = 2
            else:
                CIRCUIT_BREAKER_REQUESTS.labels(
                    service=self.name, state="open", result="rejected"
                ).inc()
                raise Exception(f"Circuit breaker is OPEN for {self.name}")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            CIRCUIT_BREAKER_REQUESTS.labels(
                service=self.name, state=self._state.value, result="success"
            ).inc()
            return result
        except self.expected_exception as e:
            self._on_failure()
            CIRCUIT_BREAKER_REQUESTS.labels(
                service=self.name, state=self._state.value, result="failure"
            ).inc()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful call."""
        if self._state == CircuitBreakerState.HALF_OPEN:
            self._state = CircuitBreakerState.CLOSED
            CIRCUIT_BREAKER_STATE_GAUGE.labels(service=self.name).set(0)  # CLOSED = 0
            CIRCUIT_BREAKER_STATE_CHANGES.labels(
                service=self.name, from_state="half_open", to_state="closed"
            ).inc()

        self._failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self.failure_threshold:
            old_state = self._state.value
            self._state = CircuitBreakerState.OPEN
            CIRCUIT_BREAKER_STATE_GAUGE.labels(service=self.name).set(1)  # OPEN = 1
            CIRCUIT_BREAKER_STATE_CHANGES.labels(
                service=self.name, from_state=old_state, to_state="open"
            ).inc()

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state
