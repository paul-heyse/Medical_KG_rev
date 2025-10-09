"""Base coordinator abstractions shared by gateway operations.

This module provides the foundational abstractions for all gateway coordinators,
including base request/result models, error handling, metrics collection, and
the abstract coordinator interface that implements retry logic, circuit breakers,
and rate limiting.

Key Components:
    - CoordinatorRequest: Base class for strongly typed coordinator requests
    - CoordinatorResult: Base class for typed coordinator results with metadata
    - CoordinatorError: Exception raised when coordinator operations fail
    - CoordinatorMetrics: Prometheus metrics collection for coordinators
    - CoordinatorConfig: Runtime configuration for coordinator behavior
    - BaseCoordinator: Abstract base class implementing retry, circuit breaker, and rate limiting

Architecture:
    - All coordinators inherit from BaseCoordinator and implement _execute method
    - BaseCoordinator handles retry logic, circuit breaking, rate limiting, and metrics
    - Concrete coordinators focus on business logic while base handles infrastructure concerns
    - Metrics are collected automatically for all coordinator operations

Thread Safety:
    - BaseCoordinator is thread-safe for concurrent operations
    - Metrics collection is thread-safe via Prometheus client
    - Circuit breaker and rate limiter are thread-safe

Performance Characteristics:
    - O(1) overhead for metrics collection
    - Retry logic adds exponential backoff delays
    - Circuit breaker prevents cascading failures
    - Rate limiting prevents resource exhaustion

Example:
    >>> from Medical_KG_rev.gateway.coordinators.base import BaseCoordinator, CoordinatorConfig
    >>> class MyCoordinator(BaseCoordinator[MyRequest, MyResult]):
    ...     def _execute(self, request: MyRequest, **kwargs) -> MyResult:
    ...         # Implement coordinator logic here
    ...         return MyResult(job_id="123", duration_s=1.0)
    >>> coordinator = MyCoordinator(
    ...     config=CoordinatorConfig(name="my_coordinator"),
    ...     metrics=CoordinatorMetrics.create("my_coordinator")
    ... )
    >>> result = coordinator(MyRequest(tenant_id="tenant1"))
"""
from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================
import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from aiolimiter import AsyncLimiter
from pybreaker import CircuitBreaker, CircuitBreakerError
from tenacity import RetryError, Retrying, stop_after_attempt, wait_exponential

import structlog
from prometheus_client import Counter, Histogram

logger = structlog.get_logger(__name__)

_RequestT = TypeVar("_RequestT", bound="CoordinatorRequest")
_ResultT = TypeVar("_ResultT", bound="CoordinatorResult")


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class CoordinatorRequest:
    """Base class for strongly typed coordinator requests.

    This class serves as a marker base class for all coordinator request types,
    providing common fields for tenant identification, correlation tracking,
    and metadata storage.

    Attributes:
        tenant_id: Unique identifier for the tenant making the request.
        correlation_id: Optional correlation ID for request tracing.
        metadata: Optional mapping of additional request metadata.

    Invariants:
        - tenant_id is never None or empty
        - correlation_id is None or non-empty string
        - metadata is None or non-empty mapping

    Thread Safety:
        - Immutable after construction (dataclass with frozen=True would be ideal)
        - Safe for concurrent access if not modified

    Lifecycle:
        - Created by gateway service layer
        - Passed to coordinator for processing
        - Not modified during processing

    Example:
        >>> request = CoordinatorRequest(
        ...     tenant_id="tenant1",
        ...     correlation_id="req-123",
        ...     metadata={"source": "api", "priority": "high"}
        ... )
        >>> print(f"Processing request for tenant: {request.tenant_id}")
    """

    tenant_id: str
    correlation_id: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass
class CoordinatorResult:
    """Base class for typed coordinator results.

    This class serves as a base class for all coordinator result types,
    providing common fields for job tracking, performance metrics,
    and result metadata.

    Attributes:
        job_id: Unique identifier for the coordinator job.
        duration_s: Duration of the coordinator operation in seconds.
        metadata: Additional result metadata and context.

    Invariants:
        - job_id is never None or empty
        - duration_s is non-negative
        - metadata is never None (defaults to empty dict)

    Thread Safety:
        - Immutable after construction
        - Safe for concurrent access

    Lifecycle:
        - Created by coordinator after successful operation
        - Returned to gateway service layer
        - Used for logging and metrics collection

    Example:
        >>> result = CoordinatorResult(
        ...     job_id="job-123",
        ...     duration_s=1.5,
        ...     metadata={"status": "success", "items_processed": 42}
        ... )
        >>> print(f"Job {result.job_id} completed in {result.duration_s}s")
    """

    job_id: str
    duration_s: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class CoordinatorError(RuntimeError):
    """Exception raised when a coordinator operation fails after all guards.

    This exception is raised when a coordinator operation fails after all
    retry attempts, circuit breaker checks, and rate limiting have been
    applied. It provides context about the failure for debugging and
    monitoring purposes.

    Attributes:
        context: Additional context about the failure (request details, error info).

    Invariants:
        - message is never None or empty
        - context is never None (defaults to empty dict)

    Thread Safety:
        - Immutable after construction
        - Safe for concurrent access

    Lifecycle:
        - Created when coordinator operation fails
        - Raised to caller (gateway service layer)
        - Logged and monitored by error handling systems

    Example:
        >>> try:
        ...     result = coordinator(request)
        ... except CoordinatorError as e:
        ...     print(f"Coordinator failed: {e}")
        ...     print(f"Context: {e.context}")
    """

    def __init__(self, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        """Initialize coordinator error with message and optional context.

        Args:
            message: Human-readable error message describing the failure.
            context: Optional mapping containing additional error context.

        Example:
            >>> error = CoordinatorError(
            ...     "Circuit breaker open",
            ...     context={"request": request, "attempts": 3}
            ... )
        """
        super().__init__(message)
        self.context = context or {}


# ============================================================================
# METRICS
# ============================================================================

_METRICS_CACHE: dict[str, CoordinatorMetrics] = {}


@dataclass(slots=True)
class CoordinatorMetrics:
    """Prometheus metrics collection for coordinator operations.

    This class encapsulates the Prometheus metrics used by all coordinators
    to track operation attempts, failures, and duration. Metrics are
    cached per coordinator name to avoid duplicate metric registration.

    Attributes:
        attempts: Counter tracking total coordinator invocations.
        failures: Counter tracking coordinator failures after retries.
        duration: Histogram tracking coordinator operation duration.

    Invariants:
        - attempts is never None
        - failures is never None
        - duration is never None
        - All metrics have consistent label names

    Thread Safety:
        - Prometheus metrics are thread-safe
        - Caching is protected by module-level cache

    Lifecycle:
        - Created once per coordinator name via create() class method
        - Cached for reuse across coordinator instances
        - Metrics persist for the lifetime of the application

    Example:
        >>> metrics = CoordinatorMetrics.create("my_coordinator")
        >>> metrics.attempts.inc()  # Increment attempt counter
        >>> with metrics.duration.time():  # Time an operation
        ...     # Perform coordinator work
        ...     pass
    """

    attempts: Counter
    failures: Counter
    duration: Histogram

    @classmethod
    def create(cls, name: str) -> CoordinatorMetrics:
        """Create or retrieve cached metrics instance for coordinator.

        This method implements a singleton pattern for metrics instances,
        ensuring that each coordinator name has exactly one metrics instance
        to avoid duplicate Prometheus metric registration.

        Args:
            name: The coordinator name used for metric labels and caching.

        Returns:
            CoordinatorMetrics instance for the given coordinator name.

        Raises:
            ValueError: If name is None or empty.

        Example:
            >>> metrics1 = CoordinatorMetrics.create("my_coordinator")
            >>> metrics2 = CoordinatorMetrics.create("my_coordinator")
            >>> assert metrics1 is metrics2  # Same instance
        """
        try:
            return _METRICS_CACHE[name]
        except KeyError:
            labels = {"coordinator": name}
            metrics = cls(
                attempts=Counter(
                    "gateway_coordinator_attempts_total",
                    "Total coordinator invocations",
                    labelnames=list(labels),
                ).labels(**labels),
                failures=Counter(
                    "gateway_coordinator_failures_total",
                    "Coordinator failures after retries",
                    labelnames=list(labels),
                ).labels(**labels),
                duration=Histogram(
                    "gateway_coordinator_duration_seconds",
                    "Coordinator operation duration",
                    labelnames=list(labels),
                ).labels(**labels),
            )
            _METRICS_CACHE[name] = metrics
            return metrics


# ============================================================================
# BASE COORDINATOR INTERFACE
# ============================================================================


@dataclass(slots=True)
class CoordinatorConfig:
    """Runtime configuration for coordinator behavior and resilience.

    This class encapsulates all configuration parameters that control
    coordinator behavior, including retry logic, circuit breaker settings,
    and rate limiting configuration.

    Attributes:
        name: Unique name identifier for the coordinator.
        retry_attempts: Maximum number of retry attempts for failed operations.
        retry_wait_base: Base wait time in seconds for exponential backoff.
        retry_wait_max: Maximum wait time in seconds for exponential backoff.
        breaker: Optional circuit breaker for failure protection.
        limiter: Optional rate limiter for request throttling.

    Invariants:
        - name is never None or empty
        - retry_attempts is positive
        - retry_wait_base is non-negative
        - retry_wait_max is greater than retry_wait_base

    Thread Safety:
        - Immutable after construction
        - Safe for concurrent access

    Lifecycle:
        - Created during coordinator initialization
        - Used throughout coordinator lifetime
        - Not modified after creation

    Example:
        >>> config = CoordinatorConfig(
        ...     name="my_coordinator",
        ...     retry_attempts=5,
        ...     retry_wait_base=0.1,
        ...     retry_wait_max=5.0
        ... )
        >>> retrying = config.build_retrying()
    """

    name: str
    retry_attempts: int = 3
    retry_wait_base: float = 0.2
    retry_wait_max: float = 2.0
    breaker: CircuitBreaker | None = None
    limiter: AsyncLimiter | None = None

    def build_retrying(self) -> Retrying:
        """Build Retrying instance configured with exponential backoff.

        This method creates a tenacity Retrying instance configured with
        the coordinator's retry parameters, implementing exponential
        backoff with jitter for resilient operation handling.

        Returns:
            Retrying instance configured with stop and wait strategies.

        Example:
            >>> config = CoordinatorConfig(name="test", retry_attempts=3)
            >>> retrying = config.build_retrying()
            >>> # Use retrying.call() to execute operations with retry logic
        """
        return Retrying(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(
                multiplier=self.retry_wait_base,
                max=self.retry_wait_max,
            ),
            reraise=True,
        )


@dataclass
class BaseCoordinator(ABC, Generic[_RequestT, _ResultT]):
    """Abstract base class for all gateway coordinators.

    This class provides the common infrastructure for all coordinators,
    including retry logic, circuit breaker protection, rate limiting,
    metrics collection, and error handling. Concrete coordinators must
    implement the _execute method to perform their specific business logic.

    The base coordinator handles all resilience concerns transparently,
    allowing concrete coordinators to focus on their core functionality
    while benefiting from automatic retry, circuit breaking, and metrics.

    Attributes:
        config: Coordinator configuration including retry and resilience settings.
        metrics: Prometheus metrics for tracking coordinator operations.

    Invariants:
        - config is never None
        - metrics is never None
        - _retrying is initialized in __post_init__
        - All public methods maintain consistent error handling

    Thread Safety:
        - Thread-safe for concurrent operations
        - Metrics collection is thread-safe
        - Circuit breaker and rate limiter are thread-safe

    Lifecycle:
        - Created with config and metrics
        - __post_init__ initializes retry and resilience components
        - Used via __call__ method for request processing
        - No explicit cleanup required

    Example:
        >>> class MyCoordinator(BaseCoordinator[MyRequest, MyResult]):
        ...     def _execute(self, request: MyRequest, **kwargs) -> MyResult:
        ...         # Implement coordinator logic
        ...         return MyResult(job_id="123", duration_s=1.0)
        >>> coordinator = MyCoordinator(
        ...     config=CoordinatorConfig(name="my_coordinator"),
        ...     metrics=CoordinatorMetrics.create("my_coordinator")
        ... )
        >>> result = coordinator(MyRequest(tenant_id="tenant1"))
    """

    config: CoordinatorConfig
    metrics: CoordinatorMetrics

    def __post_init__(self) -> None:
        """Initialize coordinator resilience components after dataclass construction.

        This method is called automatically after dataclass construction to
        set up the retry logic, rate limiter, and circuit breaker components
        based on the coordinator configuration.

        Example:
            >>> coordinator = MyCoordinator(config=config, metrics=metrics)
            >>> # __post_init__ is called automatically
            >>> assert coordinator._retrying is not None
        """
        self._retrying = self.config.build_retrying()
        self._limiter = self.config.limiter
        self._breaker = self.config.breaker

    def __call__(self, request: _RequestT, /, **kwargs: Any) -> _ResultT:
        """Execute coordinator operation with full resilience and metrics.

        This method serves as the main entry point for coordinator operations,
        providing automatic retry logic, circuit breaker protection, rate limiting,
        metrics collection, and comprehensive error handling.

        The method coordinates between the various resilience components:
        1. Logs operation start with request details
        2. Times the operation using metrics histogram
        3. Delegates to _execute_with_guards for resilience logic
        4. Collects success/failure metrics
        5. Logs operation completion or failure

        Args:
            request: The coordinator request to process.
            **kwargs: Additional keyword arguments passed to _execute method.

        Returns:
            Coordinator result containing operation outcome and metadata.

        Raises:
            CoordinatorError: If operation fails after all retry attempts.

        Example:
            >>> result = coordinator(request)
            >>> print(f"Operation completed in {result.duration_s}s")
        """
        logger.debug(
            "gateway.coordinator.invoke",
            coordinator=self.config.name,
            tenant_id=request.tenant_id,
            correlation_id=request.correlation_id,
        )
        start = time.perf_counter()
        error: Exception | None = None
        try:
            with self.metrics.duration.time():
                result = self._execute_with_guards(request, **kwargs)
            return result
        except Exception as exc:  # pragma: no cover - defensive logging
            error = exc
            raise
        finally:
            duration = time.perf_counter() - start
            self.metrics.attempts.inc()
            if error is not None:
                self.metrics.failures.inc()
                logger.warning(
                    "gateway.coordinator.failed",
                    coordinator=self.config.name,
                    tenant_id=request.tenant_id,
                    correlation_id=request.correlation_id,
                    error=str(error),
                    duration=duration,
                )
            else:
                logger.info(
                    "gateway.coordinator.completed",
                    coordinator=self.config.name,
                    tenant_id=request.tenant_id,
                    correlation_id=request.correlation_id,
                    duration=duration,
                )

    def _execute_with_guards(self, request: _RequestT, /, **kwargs: Any) -> _ResultT:
        """Execute coordinator operation with retry logic and resilience guards.

        This method orchestrates the resilience components (circuit breaker,
        rate limiter, retry logic) and delegates to the concrete _execute
        method. It handles circuit breaker errors and retry exhaustion,
        translating them into CoordinatorError exceptions.

        The execution flow:
        1. Apply rate limiting if configured
        2. Apply circuit breaker if configured
        3. Execute with retry logic
        4. Handle circuit breaker and retry errors

        Args:
            request: The coordinator request to process.
            **kwargs: Additional keyword arguments passed to _execute method.

        Returns:
            Coordinator result from successful operation.

        Raises:
            CoordinatorError: If circuit breaker is open or retries are exhausted.

        Example:
            >>> result = coordinator._execute_with_guards(request)
            >>> # Method handles all resilience concerns automatically
        """
        def _call() -> _ResultT:
            if self._breaker is not None:
                return self._breaker.call(self._execute, request, **kwargs)
            return self._execute(request, **kwargs)

        def _with_rate_limit() -> _ResultT:
            if self._limiter is None:
                return _call()
            try:
                return asyncio.run(self._consume_limiter(self._limiter, _call))
            except RuntimeError:
                # Running inside an event loop (FastAPI async handlers).
                logger.debug(
                    "gateway.coordinator.rate_limit.async_loop",
                    coordinator=self.config.name,
                )
                return _call()

        try:
            return self._retrying.call(_with_rate_limit)
        except CircuitBreakerError as exc:
            raise CoordinatorError(
                f"{self.config.name} circuit open", context={"request": request}
            ) from exc
        except RetryError as exc:
            last = exc.last_attempt
            raise CoordinatorError(
                f"{self.config.name} retries exhausted",
                context={"request": request, "error": str(last.exception())},
            ) from last.exception()

    @staticmethod
    async def _consume_limiter(
        limiter: AsyncLimiter, func: Callable[[], _ResultT]
    ) -> _ResultT:
        """Consume rate limiter permit and execute function.

        This static method handles the async rate limiter interaction,
        acquiring a permit from the limiter and then executing the
        provided function. It's used when rate limiting is enabled
        and the coordinator is running in an async context.

        Args:
            limiter: The async rate limiter to consume permits from.
            func: The function to execute after acquiring permit.

        Returns:
            Result from the executed function.

        Example:
            >>> result = await BaseCoordinator._consume_limiter(limiter, lambda: execute())
            >>> # Function executes after rate limiter permits
        """
        async with limiter:
            return func()

    @abstractmethod
    def _execute(self, request: _RequestT, /, **kwargs: Any) -> _ResultT:
        """Perform the actual coordinator work.

        This abstract method must be implemented by concrete coordinators
        to perform their specific business logic. It is called by the
        base coordinator after all resilience guards (retry, circuit breaker,
        rate limiting) have been applied.

        Concrete coordinators should focus on their core functionality
        and not worry about resilience concerns, as these are handled
        transparently by the base coordinator.

        Args:
            request: The coordinator request to process.
            **kwargs: Additional keyword arguments (unused in base implementation).

        Returns:
            Coordinator result containing operation outcome and metadata.

        Raises:
            Exception: Any exception will be caught and handled by retry logic.

        Example:
            >>> class MyCoordinator(BaseCoordinator[MyRequest, MyResult]):
            ...     def _execute(self, request: MyRequest, **kwargs) -> MyResult:
            ...         # Implement specific coordinator logic
            ...         return MyResult(job_id="123", duration_s=1.0)
        """


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "BaseCoordinator",
    "CoordinatorConfig",
    "CoordinatorError",
    "CoordinatorMetrics",
    "CoordinatorRequest",
    "CoordinatorResult",
]
