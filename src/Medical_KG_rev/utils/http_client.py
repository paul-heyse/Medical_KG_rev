from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import Enum
import time

import httpx
from aiolimiter import AsyncLimiter
from opentelemetry import trace
from pybreaker import CircuitBreaker, CircuitBreakerError
from tenacity import AsyncRetrying, RetryCallState, Retrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential, wait_fixed, wait_incrementing


class BackoffStrategy(str, Enum):
    """Supported retry backoff strategies for the HTTP clients."""

    EXPONENTIAL = "exponential"
    FIXED = "fixed"
    LINEAR = "linear"


@dataclass(frozen=True)
class HTTPResilienceConfig:
    """Configuration describing retry, circuit breaker, and rate limiting."""

    max_attempts: int = 3
    timeout: float = 10.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_initial: float = 0.5
    backoff_max: float = 30.0
    backoff_jitter: bool = True
    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)
    circuit_failure_threshold: int = 5
    circuit_reset_timeout: float = 60.0
    rate_limit_per_second: float | None = None
    rate_limit_capacity: int | None = None


class RetryAfterError(Exception):
    """Raised to signal that a response requested a Retry-After delay."""

    def __init__(self, response: httpx.Response, retry_after: float) -> None:
        super().__init__(f"Retry after {retry_after:.2f}s")
        self.response = response
        self.retry_after = max(retry_after, 0.0)


def _build_wait_strategy(config: HTTPResilienceConfig):
    if config.backoff_strategy is BackoffStrategy.FIXED:
        wait = wait_fixed(max(config.backoff_initial, 0.0))
    elif config.backoff_strategy is BackoffStrategy.LINEAR:
        wait = wait_incrementing(
            start=max(config.backoff_initial, 0.0),
            increment=max(config.backoff_initial, 0.0) or 0.1,
            max=max(config.backoff_max, 0.0) or None,
        )
    else:
        wait = wait_exponential(
            multiplier=max(config.backoff_initial, 0.0) or 0.1,
            max=max(config.backoff_max, 0.0) or None,
        )
    if config.backoff_jitter and hasattr(wait, "with_jitter"):
        wait = wait.with_jitter(0.1)  # type: ignore[attr-defined]

    def _wait(retry_state: RetryCallState) -> float:
        value = float(wait(retry_state))
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exc, RetryAfterError):
            return max(value, exc.retry_after)
        return value

    return _wait


def _create_async_limiter(config: HTTPResilienceConfig) -> AsyncLimiter | None:
    if not config.rate_limit_per_second or config.rate_limit_per_second <= 0:
        return None
    capacity = int(config.rate_limit_capacity or max(1, round(config.rate_limit_per_second)))
    period = max(capacity / config.rate_limit_per_second, 1e-6)
    return AsyncLimiter(max_rate=capacity, time_period=period)


async def _call_with_breaker(
    breaker: CircuitBreaker,
    func,
    *args,
    **kwargs,
):
    with breaker._lock:  # type: ignore[attr-defined]
        state = breaker.state
        state.before_call(func, *args, **kwargs)
        for listener in breaker.listeners:
            listener.before_call(breaker, func, *args, **kwargs)
    try:
        result = await func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - state transitions tested elsewhere
        with breaker._lock:  # type: ignore[attr-defined]
            breaker.state._handle_error(exc)
        raise
    else:
        with breaker._lock:  # type: ignore[attr-defined]
            breaker.state._handle_success()
        return result


class SyncLimiter:
    """Synchronous facade around :class:`aiolimiter.AsyncLimiter`."""

    def __init__(self, limiter: AsyncLimiter) -> None:
        self._limiter = limiter

    def acquire(self) -> float:
        start = time.perf_counter()
        try:
            asyncio.run(self._limiter.acquire())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._limiter.acquire())
            finally:
                asyncio.set_event_loop(None)
                loop.close()
        return time.perf_counter() - start


class HttpClient:
    """Wrapper around :class:`httpx.Client` with resilience helpers."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        resilience: HTTPResilienceConfig | None = None,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = resilience or HTTPResilienceConfig()
        client_kwargs: dict[str, object] = {"timeout": self._config.timeout, "transport": transport}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = client or httpx.Client(**client_kwargs)
        self._retry_statuses = set(self._config.retry_statuses)
        self._tracer = trace.get_tracer(__name__)
        self._wait_strategy = _build_wait_strategy(self._config)
        limiter = _create_async_limiter(self._config)
        self._limiter = SyncLimiter(limiter) if limiter is not None else None
        self._breaker = (
            CircuitBreaker(
                fail_max=self._config.circuit_failure_threshold,
                reset_timeout=self._config.circuit_reset_timeout,
            )
            if self._config.circuit_failure_threshold > 0
            else None
        )

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        retrying = Retrying(
            stop=stop_after_attempt(self._config.max_attempts),
            wait=self._wait_strategy,
            retry=retry_if_exception_type((RetryAfterError, httpx.HTTPError)),
            reraise=True,
        )

        def _perform_request() -> httpx.Response:
            if self._limiter is not None:
                self._limiter.acquire()
            with self._tracer.start_as_current_span("http.request") as span:
                span.set_attribute("http.method", method)
                span.set_attribute("http.url", url)
                response = self._client.request(method, url, **kwargs)
            if response.status_code in self._retry_statuses:
                raise RetryAfterError(response, _compute_retry_after(response))
            response.raise_for_status()
            return response

        def _send() -> httpx.Response:
            if self._breaker is not None:
                return self._breaker.call(_perform_request)
            return _perform_request()

        try:
            for attempt in retrying:
                with attempt:
                    return _send()
        except RetryAfterError as exc:
            exc.response.raise_for_status()
        finally:
            retrying.statistics.clear()
        raise RuntimeError("Unexpected retry exit")

    def close(self) -> None:
        self._client.close()

    @contextmanager
    def lifespan(self) -> Iterator[HttpClient]:
        try:
            yield self
        finally:
            self.close()


class AsyncHttpClient:
    """Async variant of :class:`HttpClient` with matching semantics."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        resilience: HTTPResilienceConfig | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = resilience or HTTPResilienceConfig()
        client_kwargs: dict[str, object] = {"timeout": self._config.timeout, "transport": transport}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = client or httpx.AsyncClient(**client_kwargs)
        self._retry_statuses = set(self._config.retry_statuses)
        self._tracer = trace.get_tracer(__name__)
        self._wait_strategy = _build_wait_strategy(self._config)
        self._limiter = _create_async_limiter(self._config)
        self._breaker = (
            CircuitBreaker(
                fail_max=self._config.circuit_failure_threshold,
                reset_timeout=self._config.circuit_reset_timeout,
            )
            if self._config.circuit_failure_threshold > 0
            else None
        )

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._config.max_attempts),
            wait=self._wait_strategy,
            retry=retry_if_exception_type((RetryAfterError, httpx.HTTPError)),
            reraise=True,
        )

        async def _send() -> httpx.Response:
            async def _perform() -> httpx.Response:
                with self._tracer.start_as_current_span("http.request") as span:
                    span.set_attribute("http.method", method)
                    span.set_attribute("http.url", url)
                    response = await self._client.request(method, url, **kwargs)
                if response.status_code in self._retry_statuses:
                    raise RetryAfterError(response, _compute_retry_after(response))
                response.raise_for_status()
                return response

            if self._breaker is not None:
                guarded = lambda: _call_with_breaker(self._breaker, _perform)
            else:
                guarded = _perform

            if self._limiter is not None:
                async with self._limiter:
                    return await guarded()
            return await guarded()

        try:
            async for attempt in retrying:
                with attempt:
                    return await _send()
        except RetryAfterError as exc:
            exc.response.raise_for_status()
        finally:
            retrying.statistics.clear()
        raise RuntimeError("Unexpected retry exit")

    async def aclose(self) -> None:
        await self._client.aclose()

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[AsyncHttpClient]:
        try:
            yield self
        finally:
            await self.aclose()


def _compute_retry_after(response: httpx.Response) -> float:
    header = response.headers.get("Retry-After")
    if not header:
        return 0.0
    try:
        return float(header)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(header)
        except (TypeError, ValueError):
            return 0.0
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        delta = (retry_at - now).total_seconds()
        return max(delta, 0.0)


__all__ = [
    "AsyncHttpClient",
    "BackoffStrategy",
    "CircuitBreakerError",
    "HTTPResilienceConfig",
    "HttpClient",
    "RetryAfterError",
    "SyncLimiter",
]
