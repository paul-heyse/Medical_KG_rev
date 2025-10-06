import httpx
import pytest

from Medical_KG_rev.utils.http_client import AsyncHttpClient, HttpClient, RateLimiter, RetryConfig


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


@pytest.mark.asyncio
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
