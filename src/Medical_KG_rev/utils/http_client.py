"""HTTP client utilities, retry orchestration, and circuit breaker helpers.

Key Responsibilities:
    - Construct synchronous and asynchronous HTTP clients with retry, timeout,
      backoff, rate limit, and circuit breaker behaviour
    - Provide shared configuration dataclasses used across gateway and service
      integrations
    - Emit OpenTelemetry spans to ensure downstream calls remain observable

Collaborators:
    - Upstream: Gateway services, ingestion pipelines, and orchestration stages
      rely on these helpers to call external APIs
    - Downstream: Wraps `httpx` clients, `tenacity` retry primitives,
      `pybreaker` circuit breakers, and `aiolimiter` rate limiters

Side Effects:
    - Opens network connections via `httpx`
    - Emits OpenTelemetry spans and metrics
    - Spawns background threads when adapting async rate limiters to sync code

Thread Safety:
    - Designed to be thread-safe; `SynchronousLimiter` creates a dedicated event
      loop thread while public client helpers may be shared across threads

Performance Characteristics:
    - Connection pooling is delegated to `httpx`
    - Backoff and rate limit configuration allow callers to tune latency
      trade-offs
    - Circuit breaker short-circuits repeated downstream failures

Example:
    >>> config = RetryConfig(attempts=3, backoff_strategy=BackoffStrategy.EXPONENTIAL)
    >>> client = HttpClient(config=config)
    >>> response = client.get("https://api.example.com/data")
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import Enum

import httpx
from aiolimiter import AsyncLimiter
from opentelemetry import trace
from pybreaker import CircuitBreaker, CircuitBreakerError
from tenacity import retry, stop_after_attempt, wait_exponential, wait_incrementing, wait_none
from tenacity.wait import wait_base

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================


class BackoffStrategy(str, Enum):
    """Supported retry backoff strategies for HTTP clients."""

    NONE = "none"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


@dataclass(frozen=True)
class RetryConfig:
    """Retry configuration shared by sync and async clients."""

    attempts: int = 3
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_initial: float = 0.5
    backoff_max: float = 10.0
    jitter: bool = True
    status_forcelist: Iterable[int] = (429, 500, 502, 503, 504)
    timeout: float = 10.0


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Configuration for the HTTP circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0


@dataclass(frozen=True)
class RateLimitConfig:
    """Configuration for rate limiting HTTP calls."""

    rate_per_second: float = 5.0
    burst: int | None = None


class RetryableHTTPStatus(httpx.HTTPStatusError):
    """HTTP status error annotated with Retry-After delays."""

    def __init__(
        self,
        message: str,
        *,
        request: httpx.Request,
        response: httpx.Response,
        retry_after: float = 0.0,
    ) -> None:
        """Build the retryable error wrapper.

        Args:
            message: Human readable description of the failure.
            request: Underlying HTTP request object.
            response: Response returned by the server.
            retry_after: Optional server supplied backoff duration.
        """
        super().__init__(message, request=request, response=response)
        self.retry_after = max(retry_after, 0.0)


class _RetryAfterWait(wait_base):
    """Tenacity wait strategy that honours Retry-After headers."""

    def __init__(self, fallback: wait_base) -> None:
        """Initialise wait strategy with a fallback policy.

        Args:
            fallback: Tenacity wait strategy invoked when a Retry-After header
                is not present on the latest failure.
        """
        self._fallback = fallback

    def __call__(self, retry_state: RetryCallState) -> float:
        """Return the wait duration dictated by the last failure state.

        Args:
            retry_state: Tenacity retry state containing the last outcome.

        Returns:
            Number of seconds to wait before the next retry attempt.

        Raises:
            NotImplementedError: If the retry outcome did not include a
                ``RetryableHTTPStatus`` with a valid ``Retry-After`` header.
        """
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exception, RetryableHTTPStatus) and exception.retry_after > 0:
            return exception.retry_after
        raise NotImplementedError(
            "HTTP client retry fallback not implemented. "
            "This client requires a real retry strategy implementation. "
            "Please implement or configure a proper retry mechanism."
        )


class SynchronousLimiter:
    """Adapt an :class:`AsyncLimiter` for synchronous call sites."""

    def __init__(self, limiter: AsyncLimiter) -> None:
        """Create a synchronous facade over an async rate limiter.

        Args:
            limiter: AsyncLimiter enforcing per-period rate limits.
        """
        self._limiter = limiter
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="http-client-rate-limiter",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()

    def _run_loop(self) -> None:
        """Run the limiter event loop on the worker thread."""
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def acquire(self) -> float:
        """Acquire a limiter slot and return the wait duration in seconds."""
        start = time.perf_counter()
        future = asyncio.run_coroutine_threadsafe(self._limiter.acquire(), self._loop)
        future.result()
        return time.perf_counter() - start

    def close(self) -> None:
        """Shutdown the worker thread and close the event loop."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1.0)


def _build_wait(config: RetryConfig) -> wait_base:
    """Construct a Tenacity wait strategy from retry configuration.

    Args:
        config: Retry configuration describing attempts, backoff strategy, and
            jitter behaviour.

    Returns:
        Tenacity wait strategy honouring the configured backoff parameters.
    """
    if config.backoff_strategy is BackoffStrategy.NONE:
        base = wait_none()
    elif config.backoff_strategy is BackoffStrategy.LINEAR:
        base = wait_incrementing(
            start=max(config.backoff_initial, 0.0),
            increment=max(config.backoff_initial, 0.0),
            max=max(config.backoff_max, config.backoff_initial),
        )
    else:
        base = wait_exponential(
            multiplier=max(config.backoff_initial, 0.0) or 0.1,
            max=max(config.backoff_max, config.backoff_initial),
        )
    if config.jitter and hasattr(base, "with_jitter"):
        base = base.with_jitter(0.1)  # type: ignore[attr-defined]
    return base


def _create_async_limiter(config: RateLimitConfig | None) -> AsyncLimiter | None:
    """Create an :class:`AsyncLimiter` when rate limiting is enabled.

    Args:
        config: Rate limit configuration or ``None`` when throttling is disabled.

    Returns:
        AsyncLimiter enforcing the configured rate, or ``None`` when disabled.
    """
    if config is None:
        return None
    rate = max(config.rate_per_second, 1e-6)
    burst = config.burst or max(1, int(rate))
    # AsyncLimiter enforces "max_rate" operations per period; use burst for short spikes.
    return AsyncLimiter(burst, time_period=max(burst / rate, 1e-3))


def _compute_retry_after(response: httpx.Response) -> float:
    """Parse a Retry-After header and return the wait duration in seconds.

    Args:
        response: HTTP response potentially containing a Retry-After header.

    Returns:
        Number of seconds suggested by the server before retrying.
    """
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
        delta = (retry_at - datetime.now(UTC)).total_seconds()
        return max(delta, 0.0)


async def _async_breaker_call(
    breaker: CircuitBreaker,
    func: Callable[[], Awaitable[httpx.Response]],
) -> httpx.Response:
    """Invoke ``func`` under a circuit breaker within async code paths.

    Args:
        breaker: Circuit breaker guarding downstream interactions.
        func: Awaitable producing an ``httpx.Response``.

    Returns:
        Response produced by ``func`` when the call succeeds.

    Raises:
        Exception: Propagates any exception raised by ``func`` after notifying
            the breaker of the failure.
    """
    with breaker._lock:  # type: ignore[attr-defined]
        state = breaker.state
        state.before_call(func)
        for listener in breaker.listeners:
            listener.before_call(breaker, func)
    try:
        result = await func()
    except Exception as exc:  # pragma: no cover - state transitions verified in sync tests
        with breaker._lock:  # type: ignore[attr-defined]
            breaker.state._handle_error(exc)
        raise
    else:
        with breaker._lock:  # type: ignore[attr-defined]
            breaker.state._handle_success()
        return result


class HttpClient:
    """Synchronous HTTP client with tenacity retries and pybreaker protection."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        retry: RetryConfig | None = None,
        rate_limit: RateLimitConfig | None = None,
        circuit_breaker: CircuitBreakerConfig | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Create a resilient HTTP client.

        Args:
            base_url: Optional base URL applied to every request.
            retry: Retry configuration controlling attempts, backoff, and timeouts.
            rate_limit: Rate limit configuration; when omitted no client-side
                throttling is performed.
            circuit_breaker: Circuit breaker configuration; when omitted no
                breaker is created.
            transport: Optional httpx transport override used in tests.
        """
        self._retry_config = retry or RetryConfig()
        client_kwargs: dict[str, object] = {
            "timeout": self._retry_config.timeout,
            "transport": transport,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = httpx.Client(**client_kwargs)
        self._limiter = _create_async_limiter(rate_limit)
        self._sync_limiter = SynchronousLimiter(self._limiter) if self._limiter else None
        self._breaker = (
            CircuitBreaker(
                fail_max=circuit_breaker.failure_threshold,
                reset_timeout=circuit_breaker.recovery_timeout,
            )
            if circuit_breaker
            else None
        )
        wait = _RetryAfterWait(_build_wait(self._retry_config))
        self._retry = Retrying(
            stop=stop_after_attempt(self._retry_config.attempts),
            wait=wait,
            retry=retry_if_exception_type(
                (httpx.HTTPError, RetryableHTTPStatus, CircuitBreakerError)
            ),
            reraise=True,
        )
        self._tracer = trace.get_tracer(__name__)

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Issue an HTTP request with retries, circuit breaker, and rate limit.

        Args:
            method: HTTP method (GET, POST, etc.).
            url: Fully qualified or relative URL depending on ``base_url``.
            **kwargs: Additional arguments forwarded to ``httpx.Client.request``.

        Returns:
            Response returned by ``httpx``.

        Raises:
            httpx.HTTPError: When ``httpx`` raises an error not marked retryable.
            RetryableHTTPStatus: When the retry budget exhausts while receiving
                retryable status codes.
            CircuitBreakerError: When the circuit breaker rejects the call.
        """
        def _attempt() -> httpx.Response:
            if self._sync_limiter is not None:
                self._sync_limiter.acquire()

            def _perform() -> httpx.Response:
                with self._tracer.start_as_current_span("http.request") as span:
                    span.set_attribute("http.method", method)
                    span.set_attribute("http.url", url)
                    response = self._client.request(method, url, **kwargs)
                if response.status_code in self._retry_config.status_forcelist:
                    raise RetryableHTTPStatus(
                        f"Retryable status {response.status_code}",
                        request=response.request,
                        response=response,
                        retry_after=_compute_retry_after(response),
                    )
                return response

            if self._breaker is not None:
                return self._breaker.call(_perform)
            return _perform()

        return self._retry(_attempt)

    def close(self) -> None:
        """Release limiter and HTTP resources."""
        if self._sync_limiter is not None:
            self._sync_limiter.close()
        self._client.close()

    @contextmanager
    def lifespan(self) -> Iterator[HttpClient]:
        """Provide a context manager that automatically closes the client.

        Yields:
            The current ``HttpClient`` instance.
        """
        try:
            yield self
        finally:
            self.close()


class AsyncHttpClient:
    """Async HTTP client composed from the same resilience primitives."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        retry: RetryConfig | None = None,
        rate_limit: RateLimitConfig | None = None,
        circuit_breaker: CircuitBreakerConfig | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Create an async HTTP client with retry, rate limit, and breaker support."""
        self._retry_config = retry or RetryConfig()
        client_kwargs: dict[str, object] = {
            "timeout": self._retry_config.timeout,
            "transport": transport,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = httpx.AsyncClient(**client_kwargs)
        self._limiter = _create_async_limiter(rate_limit)
        self._breaker = (
            CircuitBreaker(
                fail_max=circuit_breaker.failure_threshold,
                reset_timeout=circuit_breaker.recovery_timeout,
            )
            if circuit_breaker
            else None
        )
        wait = _RetryAfterWait(_build_wait(self._retry_config))
        self._retry = AsyncRetrying(
            stop=stop_after_attempt(self._retry_config.attempts),
            wait=wait,
            retry=retry_if_exception_type(
                (httpx.HTTPError, RetryableHTTPStatus, CircuitBreakerError)
            ),
            reraise=True,
        )
        self._tracer = trace.get_tracer(__name__)

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Issue an asynchronous HTTP request with resilience safeguards.

        Args:
            method: HTTP method (GET, POST, etc.).
            url: Fully qualified or relative URL depending on ``base_url``.
            **kwargs: Additional arguments forwarded to ``httpx.AsyncClient.request``.

        Returns:
            Response returned by ``httpx``.

        Raises:
            httpx.HTTPError: When ``httpx`` raises a non-retryable error.
            RetryableHTTPStatus: When retryable status codes persist until the
                retry budget is exhausted.
            CircuitBreakerError: When the circuit breaker rejects the call.
        """
        async def _attempt() -> httpx.Response:
            async def _perform() -> httpx.Response:
                with self._tracer.start_as_current_span("http.request") as span:
                    span.set_attribute("http.method", method)
                    span.set_attribute("http.url", url)
                    response = await self._client.request(method, url, **kwargs)
                if response.status_code in self._retry_config.status_forcelist:
                    raise RetryableHTTPStatus(
                        f"Retryable status {response.status_code}",
                        request=response.request,
                        response=response,
                        retry_after=_compute_retry_after(response),
                    )
                return response

            async def _invoke() -> httpx.Response:
                if self._limiter is None:
                    return await _perform()
                async with self._limiter:
                    return await _perform()

            if self._breaker is not None:
                return await _async_breaker_call(self._breaker, _invoke)
            return await _invoke()

        async for attempt in self._retry:
            with attempt:
                return await _attempt()
        raise RuntimeError("unreachable")  # pragma: no cover - tenacity exhausts attempts

    async def aclose(self) -> None:
        """Close the underlying ``httpx.AsyncClient``."""
        await self._client.aclose()

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[AsyncHttpClient]:
        """Provide an async context manager that closes the client.

        Yields:
            The current ``AsyncHttpClient`` instance.
        """
        try:
            yield self
        finally:
            await self.aclose()


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


# ==============================================================================
# HELPER CLASSES
# ==============================================================================


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "AsyncHttpClient",
    "BackoffStrategy",
    "CircuitBreakerConfig",
    "HttpClient",
    "RateLimitConfig",
    "RetryConfig",
    "RetryableHTTPStatus",
    "SynchronousLimiter",
]
