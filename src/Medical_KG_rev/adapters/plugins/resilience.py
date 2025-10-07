"""Unified resilience utilities shared by adapter plugins."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Coroutine, TypeVar

import httpx
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


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Simple circuit breaker implementation."""

    failure_threshold: int
    reset_timeout: float
    _failures: int = 0
    _state: CircuitState = CircuitState.CLOSED
    _opened_at: float | None = None

    def record_success(self) -> None:
        self._failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()

    def can_execute(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            assert self._opened_at is not None
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.reset_timeout:
                self._state = CircuitState.HALF_OPEN
                return True
            return False
        # Half-open allows a single trial request
        return True

    def on_trial_result(self, success: bool) -> None:
        if success:
            self.record_success()
        else:
            self.record_failure()

    @property
    def state(self) -> CircuitState:
        return self._state


RateLimiterCallable = TypeVar("RateLimiterCallable", bound=Callable[..., Any])


class TokenBucket:
    def __init__(self, capacity: int, fill_rate: float) -> None:
        self.capacity = capacity
        self.tokens = capacity
        self.fill_rate = fill_rate
        self.timestamp = time.monotonic()
        self.lock = asyncio.Lock()

    async def consume(self, tokens: float = 1.0) -> None:
        async with self.lock:
            self._add_new_tokens()
            while self.tokens < tokens:
                await asyncio.sleep(1 / max(self.fill_rate, 1e-6))
                self._add_new_tokens()
            self.tokens -= tokens

    def _add_new_tokens(self) -> None:
        now = time.monotonic()
        delta = now - self.timestamp
        self.timestamp = now
        self.tokens = min(self.capacity, self.tokens + delta * self.fill_rate)


def retry_on_failure(config: ResilienceConfig, retry_exceptions: tuple[type[Exception], ...] = (Exception,)):
    """Create a retry decorator using Tenacity with the provided configuration."""

    if config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
        wait = wait_exponential(multiplier=config.backoff_multiplier, max=config.backoff_max_seconds)
    else:
        wait = wait_fixed(config.backoff_multiplier)

    def before_sleep(retry_state: RetryCallState) -> None:  # pragma: no cover - logging hook
        if Counter is not None:
            RETRY_ATTEMPTS.labels(adapter=retry_state.kwargs.get("adapter", "unknown")).inc()

    def decorator(func: RateLimiterCallable) -> RateLimiterCallable:
        wrapped = retry(
            stop=stop_after_attempt(config.max_attempts),
            wait=wait,
            reraise=True,
            retry=retry_if_exception_type(retry_exceptions),
            before_sleep=before_sleep,
        )(func)
        return wrapped  # type: ignore[return-value]

    return decorator


def rate_limit(config: ResilienceConfig):
    bucket = TokenBucket(config.rate_limit_capacity, config.rate_limit_per_second)

    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]):
        async def wrapper(*args: Any, **kwargs: Any):
            await bucket.consume()
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def circuit_breaker(config: ResilienceConfig):
    breaker = CircuitBreaker(
        failure_threshold=config.circuit_breaker_failure_threshold,
        reset_timeout=config.circuit_breaker_reset_timeout,
    )

    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]):
        async def wrapper(*args: Any, **kwargs: Any):
            if not breaker.can_execute():
                if Gauge is not None:
                    CIRCUIT_STATE.labels(adapter=kwargs.get("adapter", "unknown")).set(1)
                raise RuntimeError("Circuit breaker is open")
            try:
                result = await func(*args, **kwargs)
            except Exception:
                breaker.record_failure()
                if Gauge is not None:
                    CIRCUIT_STATE.labels(adapter=kwargs.get("adapter", "unknown")).set(1)
                raise
            else:
                breaker.on_trial_result(True)
                if Gauge is not None:
                    CIRCUIT_STATE.labels(adapter=kwargs.get("adapter", "unknown")).set(0)
                return result

        return wrapper

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

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        async def _request() -> httpx.Response:
            response = await self._client.get(url, **kwargs)
            response.raise_for_status()
            return response

        start = time.perf_counter()
        wrapped = retry_on_failure(self.config)(_request)
        response = await wrapped()
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
