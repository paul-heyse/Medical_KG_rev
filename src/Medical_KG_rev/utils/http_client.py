"""Shared HTTP client with retry, backoff and rate limiting."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Iterable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx
from opentelemetry import trace


@dataclass(frozen=True)
class RetryConfig:
    attempts: int = 3
    backoff_factor: float = 0.5
    status_forcelist: Iterable[int] = (429, 500, 502, 503, 504)
    timeout: float = 10.0


class CircuitBreakerOpen(RuntimeError):
    """Raised when a request is attempted while the circuit is open."""


class CircuitBreaker:
    """Simple circuit breaker implementation."""

    def __init__(self, failure_threshold: int = 5, recovery_time: float = 60.0) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_time = recovery_time
        self._state_lock = threading.Lock()
        self._failure_count = 0
        self._opened_at: float | None = None

    def before_request(self) -> None:
        with self._state_lock:
            if self._opened_at is None:
                return
            if time.monotonic() - self._opened_at >= self._recovery_time:
                self._opened_at = None
                self._failure_count = 0
                return
            raise CircuitBreakerOpen("Circuit breaker is open")

    def record_success(self) -> None:
        with self._state_lock:
            self._failure_count = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._state_lock:
            self._failure_count += 1
            if self._failure_count >= self._failure_threshold:
                self._opened_at = time.monotonic()


class SyncRateLimiter:
    """Synchronous token bucket rate limiter."""

    def __init__(self, rate_per_second: float, burst: int | None = None) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be greater than zero")
        self.rate = rate_per_second
        self.capacity = burst or max(1, int(rate_per_second))
        self._tokens = float(self.capacity)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            self._refill()
            while self._tokens < 1:
                wait_time = max(1.0 / self.rate, 0.01)
                time.sleep(wait_time)
                self._refill()
            self._tokens -= 1

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated
        self._updated = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)


class RateLimiter:
    """Token bucket style rate limiter."""

    def __init__(self, rate_per_second: float, burst: int | None = None) -> None:
        self.rate = rate_per_second
        self.capacity = burst or max(1, int(rate_per_second))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            await self._refill()
            while self._tokens < 1:
                wait_time = max(1.0 / self.rate, 0.01)
                await asyncio.sleep(wait_time)
                await self._refill()
            self._tokens -= 1

    async def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated
        self._updated = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)


class HttpClient:
    """Wrapper around ``httpx.Client`` with retry logic."""

    def __init__(
        self,
        base_url: str | None = None,
        retry: RetryConfig | None = None,
        transport: httpx.BaseTransport | None = None,
        rate_limiter: SyncRateLimiter | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        client_kwargs: dict[str, object] = {
            "timeout": retry.timeout if retry else 10.0,
            "transport": transport,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = httpx.Client(**client_kwargs)
        self._retry = retry or RetryConfig()
        self._tracer = trace.get_tracer(__name__)
        self._rate_limiter = rate_limiter
        self._circuit_breaker = circuit_breaker

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        for attempt in range(1, self._retry.attempts + 1):
            if self._rate_limiter:
                self._rate_limiter.acquire()
            if self._circuit_breaker:
                self._circuit_breaker.before_request()
            with self._tracer.start_as_current_span("http.request") as span:
                span.set_attribute("http.method", method)
                span.set_attribute("http.url", url)
                try:
                    response = self._client.request(method, url, **kwargs)
                except httpx.HTTPError:
                    if self._circuit_breaker:
                        self._circuit_breaker.record_failure()
                    if attempt == self._retry.attempts:
                        raise
                    time.sleep(self._retry.backoff_factor * attempt)
                    continue
                if response.status_code in self._retry.status_forcelist:
                    if self._circuit_breaker:
                        self._circuit_breaker.record_failure()
                    retry_after = _compute_retry_after(response)
                    if attempt == self._retry.attempts:
                        response.raise_for_status()
                    sleep_time = max(retry_after, self._retry.backoff_factor * attempt)
                    time.sleep(sleep_time)
                    continue
                if self._circuit_breaker:
                    self._circuit_breaker.record_success()
                return response
        raise RuntimeError("Exhausted retry attempts")

    def close(self) -> None:
        self._client.close()

    @contextmanager
    def lifespan(self) -> Iterator[HttpClient]:
        try:
            yield self
        finally:
            self.close()


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


class AsyncHttpClient:
    """Async variant with rate limiting support."""

    def __init__(
        self,
        base_url: str | None = None,
        retry: RetryConfig | None = None,
        rate_limiter: RateLimiter | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        client_kwargs: dict[str, object] = {
            "timeout": retry.timeout if retry else 10.0,
            "transport": transport,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = httpx.AsyncClient(**client_kwargs)
        self._retry = retry or RetryConfig()
        self._tracer = trace.get_tracer(__name__)
        self._rate_limiter = rate_limiter

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        for attempt in range(1, self._retry.attempts + 1):
            if self._rate_limiter:
                await self._rate_limiter.acquire()
            with self._tracer.start_as_current_span("http.request") as span:
                span.set_attribute("http.method", method)
                span.set_attribute("http.url", url)
                try:
                    response = await self._client.request(method, url, **kwargs)
                except httpx.HTTPError:
                    if attempt == self._retry.attempts:
                        raise
                    await asyncio.sleep(self._retry.backoff_factor * attempt)
                    continue
                if response.status_code in self._retry.status_forcelist:
                    if attempt == self._retry.attempts:
                        response.raise_for_status()
                    await asyncio.sleep(self._retry.backoff_factor * attempt)
                    continue
                return response
        raise RuntimeError("Exhausted retry attempts")

    async def aclose(self) -> None:
        await self._client.aclose()

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[AsyncHttpClient]:
        try:
            yield self
        finally:
            await self.aclose()
