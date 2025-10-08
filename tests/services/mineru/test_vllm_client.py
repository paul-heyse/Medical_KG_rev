"""Unit tests for the :mod:`Medical_KG_rev.services.mineru.vllm_client` module."""

import asyncio

import pytest

pytest.importorskip("respx")
httpx = pytest.importorskip("httpx")

from Medical_KG_rev.services.mineru.vllm_client import (
    VLLMClient,
    VLLMServerError,
    VLLMTimeoutError,
)


def test_vllm_client_init() -> None:
    client = VLLMClient(base_url="http://localhost:8000")
    assert client.base_url == "http://localhost:8000"
    asyncio.run(client.close())


def test_chat_completion_success(respx_mock) -> None:
    respx_mock.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "4"}},
                ],
                "usage": {"total_tokens": 20},
            },
        )
    )

    async def _run() -> None:
        async with VLLMClient(base_url="http://localhost:8000") as client:
            response = await client.chat_completion(
                messages=[{"role": "user", "content": "What is 2+2?"}]
            )
            assert response["choices"][0]["message"]["content"] == "4"

    asyncio.run(_run())


def test_chat_completion_timeout(respx_mock) -> None:
    respx_mock.post("http://localhost:8000/v1/chat/completions").mock(
        side_effect=httpx.TimeoutException("timeout"),
    )

    async def _run() -> None:
        async with VLLMClient(base_url="http://localhost:8000") as client:
            with pytest.raises(VLLMTimeoutError):
                await client.chat_completion(messages=[{"role": "user", "content": "ping"}])

    asyncio.run(_run())


def test_chat_completion_server_error(respx_mock) -> None:
    respx_mock.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="Internal Server Error"),
    )

    async def _run() -> None:
        async with VLLMClient(base_url="http://localhost:8000") as client:
            with pytest.raises(VLLMServerError):
                await client.chat_completion(messages=[{"role": "user", "content": "ping"}])

    asyncio.run(_run())


def test_health_check_success(respx_mock) -> None:
    respx_mock.get("http://localhost:8000/health").mock(return_value=httpx.Response(200))

    async def _run() -> None:
        async with VLLMClient(base_url="http://localhost:8000") as client:
            assert await client.health_check() is True

    asyncio.run(_run())
