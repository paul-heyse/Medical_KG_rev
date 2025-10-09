"""Base coordinator abstractions shared by gateway operations."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import structlog
from aiolimiter import AsyncLimiter
from prometheus_client import Counter, Histogram
from pybreaker import CircuitBreaker, CircuitBreakerError
from tenacity import RetryError, Retrying, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

_RequestT = TypeVar("_RequestT", bound="CoordinatorRequest")
_ResultT = TypeVar("_ResultT", bound="CoordinatorResult")


@dataclass
class CoordinatorRequest:
    """Marker base class for strongly typed coordinator requests."""

    tenant_id: str
    correlation_id: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass
class CoordinatorResult:
    """Base class for typed coordinator results."""

    job_id: str
    duration_s: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class CoordinatorError(RuntimeError):
    """Raised when a coordinator operation fails after all guards."""

    def __init__(self, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}


_METRICS_CACHE: dict[str, "CoordinatorMetrics"] = {}


@dataclass(slots=True)
class CoordinatorMetrics:
    """Prometheus metrics shared by all coordinators."""

    attempts: Counter
    failures: Counter
    duration: Histogram

    @classmethod
    def create(cls, name: str) -> "CoordinatorMetrics":
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


@dataclass(slots=True)
class CoordinatorConfig:
    """Runtime configuration applied to a coordinator instance."""

    name: str
    retry_attempts: int = 3
    retry_wait_base: float = 0.2
    retry_wait_max: float = 2.0
    breaker: CircuitBreaker | None = None
    limiter: AsyncLimiter | None = None

    def build_retrying(self) -> Retrying:
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
    """Common wrapper orchestrating retries, rate limits, and metrics."""

    config: CoordinatorConfig
    metrics: CoordinatorMetrics

    def __post_init__(self) -> None:
        self._retrying = self.config.build_retrying()
        self._limiter = self.config.limiter
        self._breaker = self.config.breaker

    def __call__(self, request: _RequestT, /, **kwargs: Any) -> _ResultT:
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
    async def _consume_limiter(limiter: AsyncLimiter, func: Callable[[], _ResultT]) -> _ResultT:
        async with limiter:
            return func()

    @abstractmethod
    def _execute(self, request: _RequestT, /, **kwargs: Any) -> _ResultT:
        """Perform the actual coordinator work."""


__all__ = [
    "BaseCoordinator",
    "CoordinatorConfig",
    "CoordinatorError",
    "CoordinatorMetrics",
    "CoordinatorRequest",
    "CoordinatorResult",
]
