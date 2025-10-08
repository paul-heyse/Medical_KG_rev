import asyncio
from concurrent.futures import Future

import httpx
import pytest
from tenacity import RetryCallState, Retrying, retry_if_exception_type, stop_after_attempt

from Medical_KG_rev.utils.http_client import (
    AsyncHttpClient,
    BackoffStrategy,
    HTTPResilienceConfig,
    HttpClient,
    RetryAfterError,
    SyncLimiter,
)
from pybreaker import CircuitBreakerError


def test_http_client_retries():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": True})

    config = HTTPResilienceConfig(
        max_attempts=3,
        backoff_strategy=BackoffStrategy.FIXED,
        backoff_initial=0.0,
        backoff_max=0.0,
        backoff_jitter=False,
    )
    client = HttpClient(resilience=config, transport=httpx.MockTransport(handler))
    response = client.request("GET", "https://example.com")
    assert response.json()["ok"] is True
    assert calls["count"] == 2
    with client.lifespan() as instance:
        assert instance is client


def test_http_client_exhausts_retries():
    config = HTTPResilienceConfig(
        max_attempts=1,
        backoff_strategy=BackoffStrategy.FIXED,
        backoff_initial=0.0,
        backoff_max=0.0,
        backoff_jitter=False,
    )
    client = HttpClient(resilience=config, transport=httpx.MockTransport(lambda _: httpx.Response(503)))
    with pytest.raises(httpx.HTTPStatusError):
        client.request("GET", "https://example.com")


@pytest.mark.anyio("asyncio")
async def test_async_http_client_rate_limiter():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    config = HTTPResilienceConfig(
        max_attempts=1,
        rate_limit_per_second=2.0,
        rate_limit_capacity=1,
    )
    client = AsyncHttpClient(resilience=config, transport=httpx.MockTransport(handler))
    async with client.lifespan() as instance:
        response = await instance.request("GET", "https://example.com")
        assert response.status_code == 200


def test_http_client_respects_retry_after(monkeypatch):
    calls = {"count": 0}
    sleeps: list[float] = []

    def handler(_: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, json={"ok": True})

    config = HTTPResilienceConfig(
        max_attempts=2,
        backoff_strategy=BackoffStrategy.FIXED,
        backoff_initial=0.0,
        backoff_max=0.0,
        backoff_jitter=False,
    )
    monkeypatch.setattr("tenacity.nap.sleep", lambda seconds: sleeps.append(seconds) or None)
    client = HttpClient(resilience=config, transport=httpx.MockTransport(handler))
    response = client.request("GET", "https://example.com")
    assert response.json()["ok"] is True
    assert calls["count"] == 2
    retrying = Retrying(
        stop=stop_after_attempt(2),
        wait=client._wait_strategy,
        retry=retry_if_exception_type((RetryAfterError, httpx.HTTPError)),
        reraise=True,
    )
    state = RetryCallState(retrying, lambda: None, (), {})
    future = Future()
    future.set_exception(RetryAfterError(httpx.Response(429, headers={"Retry-After": "1"}), 1.0))
    state.outcome = future
    assert pytest.approx(client._wait_strategy(state), rel=1e-3) == 1.0


def test_http_client_circuit_breaker_opens():
    calls = {"count": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise httpx.ConnectError("boom")

    config = HTTPResilienceConfig(
        max_attempts=3,
        backoff_strategy=BackoffStrategy.FIXED,
        backoff_initial=0.0,
        backoff_max=0.0,
        backoff_jitter=False,
        circuit_failure_threshold=1,
        circuit_reset_timeout=60.0,
        retry_statuses=(429,),
    )
    client = HttpClient(resilience=config, transport=httpx.MockTransport(handler))

    with pytest.raises(CircuitBreakerError):
        client.request("GET", "https://example.com")
    assert calls["count"] == 1


def test_sync_limiter_fallback(monkeypatch):
    class StubLimiter:
        async def acquire(self):
            calls.append("acquired")

    calls: list[str] = []
    real_new_loop = asyncio.new_event_loop

    def fail_run(coro):
        raise RuntimeError("loop running")

    class DummyLoop:
        def __init__(self) -> None:
            self.invoked = False
            self.closed = False

        def run_until_complete(self, coro) -> None:
            self.invoked = True
            assert asyncio.iscoroutine(coro)
            inner_loop = real_new_loop()
            try:
                inner_loop.run_until_complete(coro)
            finally:
                inner_loop.close()

        def close(self) -> None:
            self.closed = True

    dummy_loop = DummyLoop()

    monkeypatch.setattr("Medical_KG_rev.utils.http_client.asyncio.run", fail_run)
    monkeypatch.setattr(
        "Medical_KG_rev.utils.http_client.asyncio.new_event_loop", lambda: dummy_loop
    )
    monkeypatch.setattr(
        "Medical_KG_rev.utils.http_client.asyncio.set_event_loop", lambda _: None
    )

    limiter = SyncLimiter(StubLimiter())
    limiter.acquire()
    assert calls == ["acquired"]
    assert dummy_loop.invoked
    assert dummy_loop.closed
