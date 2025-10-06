"""Shared HTTP client with retry, backoff and rate limiting."""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Iterable, Iterator, Optional

import httpx
from opentelemetry import trace


@dataclass(frozen=True)
class RetryConfig:
    attempts: int = 3
    backoff_factor: float = 0.5
    status_forcelist: Iterable[int] = (429, 500, 502, 503, 504)
    timeout: float = 10.0


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
        base_url: Optional[str] = None,
        retry: RetryConfig | None = None,
        transport: httpx.BaseTransport | None = None,
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

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        for attempt in range(1, self._retry.attempts + 1):
            with self._tracer.start_as_current_span("http.request") as span:
                span.set_attribute("http.method", method)
                span.set_attribute("http.url", url)
                try:
                    response = self._client.request(method, url, **kwargs)
                except httpx.HTTPError as exc:
                    if attempt == self._retry.attempts:
                        raise
                    time.sleep(self._retry.backoff_factor * attempt)
                    continue
                if response.status_code in self._retry.status_forcelist:
                    if attempt == self._retry.attempts:
                        response.raise_for_status()
                    time.sleep(self._retry.backoff_factor * attempt)
                    continue
                return response
        raise RuntimeError("Exhausted retry attempts")

    def close(self) -> None:
        self._client.close()

    @contextmanager
    def lifespan(self) -> Iterator["HttpClient"]:
        try:
            yield self
        finally:
            self.close()


class AsyncHttpClient:
    """Async variant with rate limiting support."""

    def __init__(
        self,
        base_url: Optional[str] = None,
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
    async def lifespan(self) -> AsyncIterator["AsyncHttpClient"]:
        try:
            yield self
        finally:
            await self.aclose()
