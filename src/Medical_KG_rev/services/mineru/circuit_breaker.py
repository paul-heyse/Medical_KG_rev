"""Circuit breaker implementation for the MinerU vLLM client.

This module provides an asynchronous circuit breaker pattern implementation
for protecting vLLM client calls from cascading failures. It monitors
success/failure rates and automatically opens/closes based on configured
thresholds to prevent resource exhaustion.

Key Components:
    - CircuitBreaker: Main circuit breaker implementation
    - CircuitState: Enumeration of circuit states (CLOSED, OPEN, HALF_OPEN)
    - CircuitBreakerOpenError: Exception raised when circuit is open
    - Fallback implementations: For environments without observability

Responsibilities:
    - Monitor vLLM client call success/failure rates
    - Automatically open circuit on failure threshold
    - Transition to half-open state after recovery timeout
    - Close circuit after success threshold in half-open state
    - Provide thread-safe state management
    - Record metrics and logs for observability

Collaborators:
    - vLLM client for protected calls
    - Metrics collection for observability
    - Logging system for debugging

Side Effects:
    - Updates circuit breaker state and metrics
    - Logs state transitions and operations
    - May block execution when circuit is open

Thread Safety:
    - Thread-safe: Uses asyncio.Lock for state protection
    - All state changes are atomic and protected

Performance Characteristics:
    - Fast state checks with minimal overhead
    - Automatic recovery prevents permanent failures
    - Configurable thresholds for different environments
    - Metrics collection for monitoring

Example:
    >>> breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
    >>> if await breaker.can_execute():
    ...     try:
    ...         # Make vLLM call
    ...         await breaker.record_success()
    ...     except Exception:
    ...         await breaker.record_failure()
    ... else:
    ...     raise CircuitBreakerOpenError("Circuit is open")
"""

from __future__ import annotations

# ==============================================================================
# IMPORTS
# ==============================================================================
import asyncio
from datetime import datetime, timedelta, timezone
from enum import Enum

# ==============================================================================
# OPTIONAL DEPENDENCIES
# ==============================================================================

try:  # pragma: no cover - fallback if observability logging dependencies missing
    from Medical_KG_rev.utils.logging import get_logger
except Exception:  # pragma: no cover - fallback to stdlib logging
    import logging

    class _FallbackLogger:
        """Fallback logger implementation for environments without observability.

        This class provides a simple logging interface that formats messages
        with structured data and delegates to Python's standard logging
        module when observability dependencies are unavailable.

        Attributes:
            _logger: Underlying Python logger instance

        Example:
            >>> logger = _FallbackLogger("test")
            >>> logger.info("Test message", key="value")
        """

        def __init__(self, name: str) -> None:
            """Initialize fallback logger.

            Args:
                name: Logger name for identification
            """
            self._logger = logging.getLogger(name)

        def _format(self, message: str, details: dict[str, object]) -> str:
            """Format message with structured details.

            Args:
                message: Base message text
                details: Structured data to append

            Returns:
                Formatted message string
            """
            if not details:
                return message
            suffix = ", ".join(f"{key}={value}" for key, value in sorted(details.items()))
            return f"{message} | {suffix}"

        def debug(self, message: str, **kwargs: object) -> None:
            """Log debug message with structured data.

            Args:
                message: Debug message text
                **kwargs: Structured data to include
            """
            self._logger.debug(self._format(message, kwargs))

        def info(self, message: str, **kwargs: object) -> None:
            """Log info message with structured data.

            Args:
                message: Info message text
                **kwargs: Structured data to include
            """
            self._logger.info(self._format(message, kwargs))

        def warning(self, message: str, **kwargs: object) -> None:
            """Log warning message with structured data.

            Args:
                message: Warning message text
                **kwargs: Structured data to include
            """
            self._logger.warning(self._format(message, kwargs))

        def error(self, message: str, **kwargs: object) -> None:
            """Log error message with structured data.

            Args:
                message: Error message text
                **kwargs: Structured data to include
            """
            self._logger.error(self._format(message, kwargs))

    def get_logger(name: str) -> _FallbackLogger:  # type: ignore[override]
        """Fallback logger factory.

        Args:
            name: Logger name for identification

        Returns:
            Fallback logger instance
        """
        return _FallbackLogger(name)

try:  # pragma: no cover - metrics import may pull optional deps
    from Medical_KG_rev.observability.metrics import MINERU_VLLM_CIRCUIT_BREAKER_STATE
except Exception:  # pragma: no cover - fallback gauge when metrics unavailable
    class _FallbackGauge:
        """Fallback gauge implementation for environments without metrics.

        This class provides a no-op gauge interface when metrics
        dependencies are unavailable, preventing import errors.

        Example:
            >>> gauge = _FallbackGauge()
            >>> gauge.set(1.0)  # No-op
        """

        def set(self, value: float) -> None:  # type: ignore[override]
            """Set gauge value (no-op).

            Args:
                value: Gauge value to set
            """
            return None

    MINERU_VLLM_CIRCUIT_BREAKER_STATE = _FallbackGauge()

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

logger = get_logger(__name__)


# ==============================================================================
# DATA MODELS
# ==============================================================================

class CircuitState(Enum):
    """Enumeration of supported circuit breaker states.

    This enum defines the three possible states of a circuit breaker:
    - CLOSED: Normal operation, allowing all requests
    - HALF_OPEN: Testing state, allowing limited requests to test recovery
    - OPEN: Failure state, blocking all requests

    Attributes:
        CLOSED: Circuit is closed, allowing all operations
        HALF_OPEN: Circuit is half-open, allowing limited operations
        OPEN: Circuit is open, blocking all operations

    Example:
        >>> state = CircuitState.CLOSED
        >>> assert state.value == 0
    """

    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is open and execution is blocked.

    This exception is raised when an operation is attempted while the
    circuit breaker is in the OPEN state, preventing execution to
    protect against cascading failures.

    Example:
        >>> breaker = CircuitBreaker()
        >>> if not await breaker.can_execute():
        ...     raise CircuitBreakerOpenError("Circuit is open")
    """

    pass


# ==============================================================================
# CIRCUIT BREAKER IMPLEMENTATION
# ==============================================================================

class CircuitBreaker:
    """Asynchronous circuit breaker guarding vLLM client invocations.

    This class implements the circuit breaker pattern to protect vLLM
    client calls from cascading failures. It monitors success/failure
    rates and automatically opens/closes based on configured thresholds.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before trying half-open
        success_threshold: Number of successes needed to close circuit
        state: Current circuit breaker state
        failure_count: Number of consecutive failures
        success_count: Number of consecutive successes
        last_failure_time: Timestamp of last failure

    Invariants:
        - failure_count >= 0
        - success_count >= 0
        - recovery_timeout > 0
        - failure_threshold >= 1
        - success_threshold >= 1

    Thread Safety:
        - Thread-safe: Uses asyncio.Lock for state protection
        - All state changes are atomic and protected

    Example:
        >>> breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
        >>> if await breaker.can_execute():
        ...     try:
        ...         # Make vLLM call
        ...         await breaker.record_success()
        ...     except Exception:
        ...         await breaker.record_failure()
        ... else:
        ...     raise CircuitBreakerOpenError("Circuit is open")
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying half-open
            success_threshold: Number of successes needed to close circuit

        Raises:
            ValueError: If thresholds or timeout are invalid

        Example:
            >>> breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
            >>> assert breaker.state == CircuitState.CLOSED
        """
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
        """Check if circuit breaker allows execution.

        Returns:
            True if circuit is closed or half-open, False if open

        Note:
            This method is thread-safe and can be called concurrently.

        Example:
            >>> breaker = CircuitBreaker()
            >>> can_execute = await breaker.can_execute()
            >>> assert can_execute is True  # Initially closed
        """
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
        """Record a successful operation.

        This method should be called after a successful vLLM operation.
        It updates the success count and may transition the circuit
        breaker to the CLOSED state if in HALF_OPEN state.

        Note:
            This method is thread-safe and can be called concurrently.

        Example:
            >>> breaker = CircuitBreaker()
            >>> await breaker.record_success()
            >>> assert breaker.success_count == 0  # Not in half-open state
        """
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
        """Record a failed operation.

        This method should be called after a failed vLLM operation.
        It updates the failure count and may transition the circuit
        breaker to the OPEN state if thresholds are exceeded.

        Note:
            This method is thread-safe and can be called concurrently.

        Example:
            >>> breaker = CircuitBreaker(failure_threshold=2)
            >>> await breaker.record_failure()
            >>> await breaker.record_failure()
            >>> assert breaker.state == CircuitState.OPEN
        """
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
        """Transition circuit breaker to open state.

        This method is called when the failure threshold is exceeded
        or when a failure occurs in the HALF_OPEN state. It resets
        the success count and logs the state transition.

        Note:
            This method assumes the lock is already held.
        """
        self.state = CircuitState.OPEN
        self.success_count = 0
        MINERU_VLLM_CIRCUIT_BREAKER_STATE.set(self.state.value)
        logger.error(
            "mineru.vllm.circuit_breaker.opened",
            failure_count=self.failure_count,
            recovery_timeout=self.recovery_timeout,
        )

    def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to half-open state.

        This method is called when the recovery timeout has passed
        and the circuit breaker is ready to test if the service
        has recovered. It resets the success count and logs the
        state transition.

        Note:
            This method assumes the lock is already held.
        """
        self.state = CircuitState.HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        MINERU_VLLM_CIRCUIT_BREAKER_STATE.set(self.state.value)
        logger.info("mineru.vllm.circuit_breaker.half_open")

    def _transition_to_closed(self) -> None:
        """Transition circuit breaker to closed state.

        This method is called when the success threshold is reached
        in the HALF_OPEN state, indicating that the service has
        recovered. It resets all counters and logs the state transition.

        Note:
            This method assumes the lock is already held.
        """
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        MINERU_VLLM_CIRCUIT_BREAKER_STATE.set(self.state.value)
        logger.info("mineru.vllm.circuit_breaker.closed")


# ==============================================================================
# EXPORTS
# ==============================================================================


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitState",
]
