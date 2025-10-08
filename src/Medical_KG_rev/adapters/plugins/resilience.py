"""Unified resilience utilities shared by adapter plugins."""

from __future__ import annotations

import time
from collections import deque
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar, cast

import httpx
from aiolimiter import AsyncLimiter
from pybreaker import CircuitBreaker, CircuitBreakerError
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

from pydantic import BaseModel, Field, NonNegativeFloat, PositiveInt

try:  # pragma: no cover - optional dependency
    from prometheus_client import Counter, Gauge
except Exception:  # pragma: no cover - optional dependency
    Counter = Gauge = None  # type: ignore


class BackoffStrategy(str, Enum):
    """Supported retry backoff strategies."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    JITTER = "jitter"


class CircuitState(str, Enum):
    """Normalised circuit breaker states exposed to adapter callers."""

    CLOSED = "closed"
    HALF_OPEN = "half-open"
    OPEN = "open"

    @classmethod
    def from_pybreaker(cls, state: Any) -> "CircuitState":
        """Coerce pybreaker state objects/strings into a ``CircuitState``."""

        if isinstance(state, CircuitState):
            return state
        name = getattr(state, "name", None) or str(state)
        normalised = name.replace("_", "-").lower()
        if normalised in {"closed"}:
            return cls.CLOSED
        if normalised in {"half-open", "halfopen"}:
            return cls.HALF_OPEN
        return cls.OPEN

    @property
    def is_open(self) -> bool:
        return self is CircuitState.OPEN


class ResilienceConfig(BaseModel):
    """Configuration for retry, rate limiting and circuit breaking."""

    max_attempts: PositiveInt = Field(3, description="Maximum retry attempts before failing.")
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_multiplier: NonNegativeFloat = Field(1.0, description="Multiplier applied to backoff intervals.")
    backoff_max_seconds: NonNegativeFloat = Field(60.0, description="Maximum backoff duration in seconds.")
    rate_limit_per_second: NonNegativeFloat = Field(5.0, description="Token bucket fill rate.")
    rate_limit_capacity: PositiveInt = Field(10, description="Maximum tokens in the bucket.")
    circuit_breaker_failure_threshold: PositiveInt = Field(
        5, description="Failures required before opening the circuit."
    )
    circuit_breaker_reset_timeout: NonNegativeFloat = Field(
        30.0, description="Seconds to wait before allowing a trial request."
    )


FuncT = TypeVar("FuncT", bound=Callable[..., Any])
AsyncFuncT = TypeVar("AsyncFuncT", bound=Callable[..., Awaitable[Any]])


def _build_wait_strategy(config: ResilienceConfig):
    if config.backoff_strategy is BackoffStrategy.EXPONENTIAL:
        return wait_exponential(
            multiplier=max(config.backoff_multiplier, 0.0),
            max=max(config.backoff_max_seconds, 0.0),
        )
    if config.backoff_strategy is BackoffStrategy.JITTER:
        base = wait_exponential(
            multiplier=max(config.backoff_multiplier, 0.0) or 1.0,
            max=max(config.backoff_max_seconds, 0.0) or None,
        )
        if hasattr(base, "with_jitter"):
            return base.with_jitter(0.1)
        return base
    return wait_fixed(config.backoff_multiplier)


def retry_on_failure(
    config: ResilienceConfig,
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[FuncT], FuncT]:
    """Create a retry decorator backed by Tenacity."""

    wait_strategy = _build_wait_strategy(config)

    def before_sleep(retry_state: RetryCallState) -> None:  # pragma: no cover - logging hook
        if Counter is not None:
            adapter = retry_state.kwargs.get("adapter", "unknown")
            RETRY_ATTEMPTS.labels(adapter=adapter).inc()

    def decorator(func: FuncT) -> FuncT:
        wrapped = retry(
            stop=stop_after_attempt(config.max_attempts),
            wait=wait_strategy,
            reraise=True,
            retry=retry_if_exception_type(retry_exceptions),
            before_sleep=before_sleep,
        )(func)
        return cast(FuncT, wrapped)

    return decorator


def _create_rate_limiter(config: ResilienceConfig) -> AsyncLimiter | None:
    if config.rate_limit_per_second <= 0:
        return None
    capacity = max(1, int(config.rate_limit_capacity))
    fill_rate = max(config.rate_limit_per_second, 1e-6)
    period = max(capacity / fill_rate, 1e-6)
    return AsyncLimiter(capacity, period)


def rate_limit(config: ResilienceConfig) -> Callable[[AsyncFuncT], AsyncFuncT]:
    limiter = _create_rate_limiter(config)

    def decorator(func: AsyncFuncT) -> AsyncFuncT:
        if limiter is None:
            return func

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any):
            start = time.perf_counter()
            async with limiter:
                waited = time.perf_counter() - start
                if Counter is not None and waited > 0:
                    RETRY_LATENCY.labels(operation="rate_limit_wait").inc(waited)
                return await func(*args, **kwargs)

        return cast(AsyncFuncT, wrapper)

    return decorator


async def _call_with_breaker(
    breaker: CircuitBreaker,
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    **kwargs: Any,
) -> Any:
    with breaker._lock:  # type: ignore[attr-defined]
        state = breaker.state
        state.before_call(func, *args, **kwargs)
        for listener in breaker.listeners:
            listener.before_call(breaker, func, *args, **kwargs)
    try:
        result = await func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - state transitions tested separately
        with breaker._lock:  # type: ignore[attr-defined]
            breaker.state._handle_error(exc)
        raise
    else:
        with breaker._lock:  # type: ignore[attr-defined]
            breaker.state._handle_success()
        return result


def circuit_breaker(config: ResilienceConfig) -> Callable[[AsyncFuncT], AsyncFuncT]:
    breaker = CircuitBreaker(
        fail_max=config.circuit_breaker_failure_threshold,
        reset_timeout=config.circuit_breaker_reset_timeout,
    )

    def decorator(func: AsyncFuncT) -> AsyncFuncT:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any):
            adapter = kwargs.get("adapter", "unknown")
            try:
                result = await _call_with_breaker(breaker, func, *args, **kwargs)
            except CircuitBreakerError:
                if Gauge is not None:
                    CIRCUIT_STATE.labels(adapter=adapter).set(1)
                raise
            except Exception:
                if Gauge is not None and CircuitState.from_pybreaker(breaker.current_state).is_open:
                    CIRCUIT_STATE.labels(adapter=adapter).set(1)
                raise
            else:
                if Gauge is not None:
                    CIRCUIT_STATE.labels(adapter=adapter).set(0)
                return result

        return cast(AsyncFuncT, wrapper)

    return decorator


class ResilientHTTPClient:
    """HTTP client wrapper applying retry and rate limiting."""

    def __init__(
        self,
        config: ResilienceConfig | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config or ResilienceConfig()
        self._client = client or httpx.AsyncClient()
        self._history: deque[float] = deque(maxlen=20)
        self._retry = retry_on_failure(self.config)
        self._limiter = _create_rate_limiter(self.config)
        self._breaker = CircuitBreaker(
            fail_max=self.config.circuit_breaker_failure_threshold,
            reset_timeout=self.config.circuit_breaker_reset_timeout,
        )

    @property
    def circuit_state(self) -> CircuitState:
        """Return the current state of the internal circuit breaker."""

        return CircuitState.from_pybreaker(self._breaker.current_state)

    async def get(self, url: str, *, adapter_name: str = "http", **kwargs: Any) -> httpx.Response:
        async def _request() -> httpx.Response:
            response = await self._client.get(url, **kwargs)
            response.raise_for_status()
            return response

        start = time.perf_counter()
        call = self._retry(_request)

        if self._limiter is not None:
            previous = call

            async def limited_call() -> httpx.Response:
                async with self._limiter:
                    return await previous()

            call = limited_call

        previous_call = call

        async def guarded_call() -> httpx.Response:
            try:
                result = await _call_with_breaker(self._breaker, previous_call)
            except CircuitBreakerError:
                if Gauge is not None:
                    CIRCUIT_STATE.labels(adapter=adapter_name).set(1)
                raise
            except Exception:
                if Gauge is not None and self.circuit_state.is_open:
                    CIRCUIT_STATE.labels(adapter=adapter_name).set(1)
                raise
            else:
                if Gauge is not None:
                    CIRCUIT_STATE.labels(adapter=adapter_name).set(0)
                return result

        response = await guarded_call()
        await self._record_latency(time.perf_counter() - start)
        return response

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "ResilientHTTPClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def _record_latency(self, elapsed: float) -> None:
        self._history.append(elapsed)
        if Counter is not None:
            RETRY_LATENCY.labels(operation="get").inc(elapsed)


if Counter is not None:  # pragma: no cover - optional metrics registration
    RETRY_ATTEMPTS = Counter(
        "adapter_retry_attempts_total",
        "Retry attempts performed by adapter resilience layer",
        labelnames=("adapter",),
    )
    RETRY_LATENCY = Counter(
        "adapter_retry_latency_seconds_total",
        "Total time spent inside adapter retry logic",
        labelnames=("operation",),
    )
    CIRCUIT_STATE = Gauge(
        "adapter_circuit_state",
        "State of adapter circuit breaker (1=open)",
        labelnames=("adapter",),
    )
else:  # pragma: no cover - fallback placeholders
    RETRY_ATTEMPTS = RETRY_LATENCY = CIRCUIT_STATE = None  # type: ignore
