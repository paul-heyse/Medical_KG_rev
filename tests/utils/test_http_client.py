import httpx
import pytest

from Medical_KG_rev.utils.http_client import (
    AsyncHttpClient,
    CircuitBreaker,
    CircuitBreakerOpen,
    HttpClient,
    RateLimiter,
    RetryConfig,
    SyncRateLimiter,
)


def test_http_client_retries():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": True})

    client = HttpClient(retry=RetryConfig(attempts=3, backoff_factor=0), transport=httpx.MockTransport(handler))
    response = client.request("GET", "https://example.com")
    assert response.json()["ok"] is True
    assert calls["count"] == 2
    with client.lifespan() as c:
        assert c is client


def test_http_client_exhausts_retries():
    client = HttpClient(retry=RetryConfig(attempts=1, backoff_factor=0), transport=httpx.MockTransport(lambda _: httpx.Response(503)))
    with pytest.raises(httpx.HTTPStatusError):
        client.request("GET", "https://example.com")


@pytest.mark.anyio("asyncio")
async def test_async_http_client_rate_limiter():
    rate_limiter = RateLimiter(rate_per_second=10)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    client = AsyncHttpClient(
        retry=RetryConfig(attempts=1),
        transport=httpx.MockTransport(handler),
        rate_limiter=rate_limiter,
    )
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

    client = HttpClient(retry=RetryConfig(attempts=2, backoff_factor=0), transport=httpx.MockTransport(handler))
    monkeypatch.setattr("Medical_KG_rev.utils.http_client.time.sleep", lambda seconds: sleeps.append(seconds))
    response = client.request("GET", "https://example.com")
    assert response.json()["ok"] is True
    assert pytest.approx(sleeps[0], rel=1e-3) == 1.0


def test_http_client_circuit_breaker_opens():
    transport = httpx.MockTransport(lambda _: httpx.Response(500))
    breaker = CircuitBreaker(failure_threshold=2, recovery_time=60)
    client = HttpClient(
        retry=RetryConfig(attempts=1, backoff_factor=0),
        transport=transport,
        circuit_breaker=breaker,
    )

    with pytest.raises(httpx.HTTPStatusError):
        client.request("GET", "https://example.com")
    with pytest.raises(httpx.HTTPStatusError):
        client.request("GET", "https://example.com")
    with pytest.raises(CircuitBreakerOpen):
        client.request("GET", "https://example.com")


def test_sync_rate_limiter_blocks(monkeypatch):
    limiter = SyncRateLimiter(rate_per_second=1, burst=1)
    current_time = {"value": 0.0}
    sleeps: list[float] = []

    def fake_monotonic() -> float:
        return current_time["value"]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        current_time["value"] += seconds

    monkeypatch.setattr("Medical_KG_rev.utils.http_client.time.monotonic", fake_monotonic)
    monkeypatch.setattr("Medical_KG_rev.utils.http_client.time.sleep", fake_sleep)

    limiter.acquire()
    limiter.acquire()
    assert sleeps
