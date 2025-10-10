import time

import httpx
import pytest
from pybreaker import CircuitBreakerError

from Medical_KG_rev.utils.http_client import (
    AsyncHttpClient,
    BackoffStrategy,
    CircuitBreakerConfig,
    HttpClient,
    RateLimitConfig,
    RetryConfig,
)


def test_http_client_retries():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": True})

    client = HttpClient(
        retry=RetryConfig(attempts=3, backoff_strategy=BackoffStrategy.NONE, jitter=False),
        transport=httpx.MockTransport(handler),
    )
    response = client.request("GET", "https://example.com")
    assert response.json()["ok"] is True
    assert calls["count"] == 2
    client.close()


def test_http_client_exhausts_retries():
    client = HttpClient(
        retry=RetryConfig(attempts=1, backoff_strategy=BackoffStrategy.NONE, jitter=False),
        transport=httpx.MockTransport(lambda _: httpx.Response(503)),
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.request("GET", "https://example.com")
    client.close()


def test_http_client_respects_retry_after(monkeypatch):
    calls = {"count": 0}
    sleeps: list[float] = []

    def handler(_: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, json={"ok": True})

    client = HttpClient(
        retry=RetryConfig(attempts=2, backoff_strategy=BackoffStrategy.NONE, jitter=False),
        transport=httpx.MockTransport(handler),
    )
    monkeypatch.setattr("tenacity._utils.sleep", lambda seconds: sleeps.append(seconds))
    response = client.request("GET", "https://example.com")
    assert response.json()["ok"] is True
    assert pytest.approx(sleeps[0], rel=1e-3) == 1.0
    client.close()


def test_http_client_circuit_breaker_opens():
    transport = httpx.MockTransport(lambda _: httpx.Response(500))
    client = HttpClient(
        retry=RetryConfig(attempts=1, backoff_strategy=BackoffStrategy.NONE, jitter=False),
        transport=transport,
        circuit_breaker=CircuitBreakerConfig(failure_threshold=2, recovery_timeout=60.0),
    )

    with pytest.raises(httpx.HTTPStatusError):
        client.request("GET", "https://example.com")
    with pytest.raises(httpx.HTTPStatusError):
        client.request("GET", "https://example.com")
    with pytest.raises(CircuitBreakerError):
        client.request("GET", "https://example.com")
    client.close()


def test_http_client_uses_rate_limiter(monkeypatch):
    class StubLimiter:
        def __init__(self) -> None:
            self.calls: list[float] = []

        async def acquire(self) -> None:
            self.calls.append(time.perf_counter())

    limiter = StubLimiter()
    monkeypatch.setattr(
        "Medical_KG_rev.utils.http_client._create_async_limiter",
        lambda config: limiter,
    )

    client = HttpClient(
        retry=RetryConfig(attempts=1, backoff_strategy=BackoffStrategy.NONE, jitter=False),
        transport=httpx.MockTransport(lambda _: httpx.Response(200)),
        rate_limit=RateLimitConfig(rate_per_second=5),
    )

    client.request("GET", "https://example.com")
    client.request("GET", "https://example.com")
    assert len(limiter.calls) == 2
    client.close()


@pytest.mark.anyio("asyncio")
async def test_async_http_client_rate_limiter(monkeypatch):
    class StubAsyncLimiter:
        def __init__(self) -> None:
            self.entries = 0

        async def __aenter__(self) -> "StubAsyncLimiter":
            self.entries += 1
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    limiter = StubAsyncLimiter()
    monkeypatch.setattr(
        "Medical_KG_rev.utils.http_client._create_async_limiter",
        lambda config: limiter,
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    client = AsyncHttpClient(
        retry=RetryConfig(attempts=1, backoff_strategy=BackoffStrategy.NONE, jitter=False),
        transport=httpx.MockTransport(handler),
        rate_limit=RateLimitConfig(rate_per_second=5),
    )

    response = await client.request("GET", "https://example.com")
    assert response.status_code == 200
    assert limiter.entries == 1
    await client.aclose()
